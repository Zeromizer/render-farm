"""API-format ComfyUI graph for an LTX 2.5 clean plate: the clip regenerated with every vehicle
and person taken out (Lightricks' Clean-Plate IC-LoRA), for clip_edit's move mode. It replaces
the masked temporal median, which only worked on a locked-off camera and needed every other
car tracked and excluded; the IC-LoRA follows the camera and takes out cars and their shadows
itself.

Follows vlo's vlo_ltx2_5_clean_plate (github.com/PxTicks/vlo) and Lightricks'
LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled, on CORE nodes only (no LTXVideo or KJNodes pack):

  LoadVideo(source) -> LTXVAddGuide(whole clip at frame 0)   [the LTXVideo pack's IC-LoRA guide
                                                              is this same call when the LoRA's
                                                              reference_downscale_factor is 1]
  UNETLoader(dev int8) -> LoRA(distilled 450) -> LoRA(clean plate)
  -> CFGGuider(cfg 1) + ManualSigmas(the distilled 8 steps) -> SamplerCustomAdvanced
  -> LTXVSeparateAVLatent -> LTXVCropGuides -> VAEDecodeTiled -> SaveVideo

The graph encodes audio (LTX is an audio-video model), so the source needs an audio track:
clip_move's prep writes the windows with a silent one.

Spike (2026-10-11, farm branch spike/ltx-clean-plate): 1024x576x121 in ~155 s, VRAM 15.6 of
16 GB, system RAM peak 28.4 of 31 GB, so windows stay at that size. On the crossroads clip it
ties the median plate; on a drone shot that follows a car it removes the car and keeps the
camera move.
"""

REQUIRED_NODES = ("LoadVideo", "GetVideoComponents", "UNETLoader", "LoraLoaderModelOnly", "CLIPLoader",
                  "VAELoader", "EmptyLTXVLatentVideo", "LTXVAddGuide", "LTXVConditioning", "LTXVAudioVAEEncode",
                  "LTXVConcatAVLatent", "ManualSigmas", "CFGGuider", "SamplerCustomAdvanced",
                  "LTXVSeparateAVLatent", "LTXVCropGuides", "VAEDecodeTiled", "CreateVideo", "SaveVideo")

UNET = "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
PLATE_LORA = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
# (node class, input, file): what the PC must have for the plate (checked against /object_info)
MODEL_FILES = (("UNETLoader", "unet_name", UNET), ("LoraLoaderModelOnly", "lora_name", DISTILLED_LORA),
               ("LoraLoaderModelOnly", "lora_name", PLATE_LORA), ("CLIPLoader", "clip_name", TEXT_ENCODER),
               ("VAELoader", "vae_name", VIDEO_VAE), ("VAELoader", "vae_name", AUDIO_VAE))
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # LTX distilled 8-step
FPS = 24.0
# One window: the size the spike ran on the 16 GB card with ~3 GB of system RAM to spare.
WINDOW_PIXELS = 1024 * 576
WINDOW_FRAMES = 121


def prompts(noun):
    """Positive and negative prompts. The card: describe the empty result (the LoRA was
    trained on captions of the clean scene) and name what must go in BOTH prompts."""
    noun = (noun or "car").strip()
    pos = ("An empty clean plate of the exact same location: identical roads, lane markings, pavements, buildings, "
           "trees and lighting as the source video, with no vehicles on the roads. "
           f"No cars, no {noun}, no vans, no vehicles anywhere in the frame. No cast shadows from vehicles on the "
           "road. Photorealistic footage with exactly the same camera movement, natural light, high detail. "
           "Keep the roads, markings, trees and buildings intact.")
    neg = (f"car, cars, {noun}, vehicle, vehicles, van, car shadow, leftover cars, ghosting, semi-transparent "
           "remnants, blurry, soft, distorted, inconsistent motion, jitter, worst quality, cartoon, video game")
    return pos, neg


def missing_models(info):
    """The model files this graph needs that the server's loaders do not list."""
    out = []
    for cls, inp, name in MODEL_FILES:
        spec = (info.get(cls) or {}).get("input", {})
        s = spec.get("required", {}).get(inp) or spec.get("optional", {}).get(inp)
        opts = None
        if s:
            opts = s[0] if isinstance(s[0], list) else ((s[1] or {}).get("options") if len(s) > 1 else None)
        if not opts or name not in opts:
            out.append(name)
    return out


def build(video_name, noun, width, height, frames, seed, prefix):
    """Return (graph, meta). The source video uploaded at exactly width x height (32-aligned)
    and `frames` frames (8k+1), with an audio track."""
    if width % 32 or height % 32:
        raise ValueError(f"LTX canvas must be 32-aligned, got {width}x{height}")
    if (frames - 1) % 8:
        raise ValueError(f"LTX frame count must be 8k+1, got {frames}")
    pos, neg = prompts(noun)
    g = {
        "src_v": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "src": {"class_type": "GetVideoComponents", "inputs": {"video": ["src_v", 0]}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "dist": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": DISTILLED_LORA, "strength_model": 1.0}},
        "plate": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["dist", 0], "lora_name": PLATE_LORA, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "ltxv", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": pos}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": neg}},
        "latent": {"class_type": "EmptyLTXVLatentVideo",
                   "inputs": {"width": int(width), "height": int(height), "length": int(frames), "batch_size": 1}},
        "guide": {"class_type": "LTXVAddGuide",
                  "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "latent": ["latent", 0],
                             "image": ["src", 0], "frame_idx": 0, "strength": 1.0}},
        "cond": {"class_type": "LTXVConditioning",
                 "inputs": {"positive": ["guide", 0], "negative": ["guide", 1], "frame_rate": FPS}},
        "aenc": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["src", 1], "audio_vae": ["avae", 0]}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["guide", 2], "audio_latent": ["aenc", 0]}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "guider": {"class_type": "CFGGuider",
                   "inputs": {"model": ["plate", 0], "positive": ["cond", 0], "negative": ["cond", 1], "cfg": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sigmas", 0], "latent_image": ["av", 0]}},
        "split": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sample", 0]}},
        "crop": {"class_type": "LTXVCropGuides",
                 "inputs": {"positive": ["cond", 0], "negative": ["cond", 1], "latent": ["split", 0]}},
        "dec": {"class_type": "VAEDecodeTiled",
                "inputs": {"samples": ["crop", 2], "vae": ["vae", 0], "tile_size": 512, "overlap": 64,
                           "temporal_size": 4096, "temporal_overlap": 8}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": FPS}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix,
                                                       "format": "auto", "codec": "auto"}},
    }
    return g, {"width": int(width), "height": int(height), "frames": int(frames), "seed": int(seed)}
