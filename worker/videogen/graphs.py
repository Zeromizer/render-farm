"""API-format ComfyUI graphs for MiniMax H3 (t2v / i2v / r2v).

Node ids and input names come straight from the ComfyUI checkout
(comfy_extras/nodes_minimax_h3.py, nodes_video.py, nodes_custom_sampler.py),
mirroring the official video_minimax_h3_*.json templates:

  UNETLoader -> [LoraLoaderModelOnly turbo] -> BasicGuider/BasicScheduler
  CLIPLoader(type=minimax) + VAELoader x2 -> MiniMaxH3ImageToVideo | MiniMaxH3ReferenceToVideo
  RandomNoise + KSamplerSelect(res_multistep) -> SamplerCustomAdvanced
  -> VAEDecode + VAEDecodeAudio -> CreateVideo(24 fps, audio muxed) -> SaveVideo(mp4)

T2V is MiniMaxH3ImageToVideo with no frames connected; there is no separate
T2V node. H3 is guidance-distilled: BasicGuider, no negative prompt, no CFG.

Autogrow (variadic) inputs are addressed in API format by a dotted path,
"<group>.<prefix><i>" (comfy_api/latest/_io.py finalize_prefix), e.g.
"ref_images.ref_image_0". A flat "ref_image_0" is rejected at execute().
"""
import math

FPS = 24

CHECKPOINTS = {
    "fl2va": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",   # t2v + i2v (first/last frame)
    "ref2va": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",  # reference-to-video
}
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
# Turbo LoRAs: distilled step counts. The 8-step FL2V one is what the official
# t2v/i2v templates ship with; ref2v only has a 4-step distillation.
TURBO_LORAS = {
    "fl2va": {8: "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
              4: "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"},
    "ref2va": {4: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"},
}
DEFAULT_TURBO_STEPS = {"fl2va": 8, "ref2va": 4}
DEFAULT_FULL_STEPS = 20

# Short edge per preset. 768p is the model's native canvas (1344x768 at 16:9);
# 480p is where a 16 GB card should live.
SHORT_EDGE = {"480p": 480, "768p": 768}
RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}

MODES = ("t2v", "i2v", "r2v", "upscale", "turntable")   # turntable = studio/turntable.py flow, not a graph
GEN_MODES = ("t2v", "i2v", "r2v")

# SeedVR2 (ByteDance, Apache 2.0): one-step video restoration/upscale, native in
# ComfyUI >= 0.28. 3B fp8 is the 16 GB-card variant (int8_convrot wants a CUDA-13
# torch build, same as H3). Wiring mirrors the official template
# utility_seedvr2_3b_int8_upscale_video.json.
SEEDVR2_MODEL = "seedvr2_3b_fp8_e4m3fn.safetensors"
SEEDVR2_VAE = "seedvr2_ema_vae_fp16.safetensors"
UPSCALE_COLOR_METHODS = ("lab", "wavelet", "adain", "none")
MAX_REF_IMAGES, MAX_REF_VIDEOS, MAX_REF_AUDIOS = 9, 3, 3


def dims(resolution, ratio):
    if resolution not in SHORT_EDGE:
        raise ValueError(f"resolution must be one of {sorted(SHORT_EDGE)}, got {resolution!r}")
    if ratio not in RATIOS:
        raise ValueError(f"ratio must be one of {sorted(RATIOS)}, got {ratio!r}")
    short = SHORT_EDGE[resolution]
    r = RATIOS[ratio]
    # floor, not round: the official canvases are 1344x768 / 832x480 at 16:9
    long_ = int(short * max(r, 1 / r) // 32) * 32
    return (long_, short) if r >= 1 else (short, long_)


def frames_for(duration_s):
    """Snap seconds to H3's 17k+5 frame grid at 24 fps (5, 22, 39, ..., 124 ~ 5 s)."""
    d = float(duration_s)
    if not 1 <= d <= 15:
        raise ValueError(f"duration_s must be 1-15, got {duration_s}")
    n = int(round(d * FPS))
    return max(5, int(math.ceil((n - 5) / 17)) * 17 + 5)


def _family(mode):
    return "ref2va" if mode == "r2v" else "fl2va"


def build(mode, p, inputs, filename_prefix):
    """Return (api_prompt_dict, meta).

    mode      t2v | i2v | r2v
    p         params.video_gen (prompt, duration_s, resolution, ratio, seed,
              turbo, steps, shift_video, shift_audio, ref_image_size)
    inputs    ComfyUI input names already uploaded via /upload/image:
              {"first_frame": name, "last_frame": name,
               "ref_images": [name...], "ref_videos": [name...], "ref_audios": [name...]}
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("video_gen needs a non-empty prompt (H3 requires a text item)")

    fam = _family(mode)
    width, height = dims(p.get("resolution") or "480p", p.get("ratio") or "16:9")
    length = frames_for(p.get("duration_s", 5))
    turbo = p.get("turbo", True) is not False
    seed = int(p.get("seed") if p.get("seed") is not None else 0)

    g = {}
    g["unet"] = {"class_type": "UNETLoader",
                 "inputs": {"unet_name": CHECKPOINTS[fam], "weight_dtype": "default"}}
    model_ref = ["unet", 0]
    if turbo:
        steps = int(p.get("steps") or DEFAULT_TURBO_STEPS[fam])
        lora = TURBO_LORAS[fam].get(steps) or TURBO_LORAS[fam][DEFAULT_TURBO_STEPS[fam]]
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model_ref, "lora_name": lora, "strength_model": 1.0}}
        model_ref = ["lora", 0]
    else:
        steps = int(p.get("steps") or DEFAULT_FULL_STEPS)
    if p.get("shift_video") is not None or p.get("shift_audio") is not None:
        g["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                      "inputs": {"model": model_ref,
                                 "shift_video": float(p.get("shift_video", 12.0)),
                                 "shift_audio": float(p.get("shift_audio", 3.0))}}
        model_ref = ["shift", 0]

    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}

    cond_inputs = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                   "width": width, "height": height, "length": length}
    if mode in ("t2v", "i2v"):
        if mode == "i2v" and not inputs.get("first_frame") and not inputs.get("last_frame"):
            raise ValueError("i2v needs first_frame and/or last_frame")
        for key in ("first_frame", "last_frame"):
            name = inputs.get(key)
            if name:
                g[f"load_{key}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
                cond_inputs[key] = [f"load_{key}", 0]
        g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}
    else:
        imgs = list(inputs.get("ref_images") or [])[:MAX_REF_IMAGES]
        vids = list(inputs.get("ref_videos") or [])[:MAX_REF_VIDEOS]
        auds = list(inputs.get("ref_audios") or [])[:MAX_REF_AUDIOS]
        if not (imgs or vids or auds):
            raise ValueError("r2v needs at least one of ref_images / ref_videos / ref_audios")
        cond_inputs["audio_vae"] = ["avae", 0]
        cond_inputs["ref_image_size"] = p.get("ref_image_size") or "match"
        for i, name in enumerate(imgs):
            g[f"load_ref_image_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond_inputs[f"ref_images.ref_image_{i}"] = [f"load_ref_image_{i}", 0]
        for i, name in enumerate(vids):
            g[f"load_ref_video_{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
            g[f"ref_video_parts_{i}"] = {"class_type": "GetVideoComponents",
                                         "inputs": {"video": [f"load_ref_video_{i}", 0]}}
            cond_inputs[f"ref_videos.ref_video_{i}"] = [f"ref_video_parts_{i}", 0]              # images
            cond_inputs[f"ref_video_audios.ref_video_audio_{i}"] = [f"ref_video_parts_{i}", 1]  # audio
        for i, name in enumerate(auds):
            g[f"load_ref_audio_{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            cond_inputs[f"ref_audios.ref_audio_{i}"] = [f"load_ref_audio_{i}", 0]
        g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}

    g["guider"] = {"class_type": "BasicGuider",
                   "inputs": {"model": model_ref, "conditioning": ["cond", 0]}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["cond", 1]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo",
                  "inputs": {"images": ["dec", 0], "fps": float(FPS), "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": filename_prefix, "format": "mp4"}}

    meta = {"family": fam, "width": width, "height": height, "length": length,
            "seconds": round(length / FPS, 2), "steps": steps, "turbo": turbo, "seed": seed}
    return g, meta


def build_upscale(video_name, u, filename_prefix):
    """SeedVR2 upscale of a clip already in ComfyUI/input. Returns (graph, meta).

    video_name  input-relative name from comfy_client.upload_input (any mp4/mov/webm)
    u           params.video_gen.upscale:
                  factor            scale multiplier (default 2.0); or
                  shorter_size      target short edge in px (wins over factor)
                  color_correction  wavelet | lab | adain | none   (default wavelet:
                                    measured on the 4080, lab costs ~3 s/frame on top,
                                    wavelet ~0.25 s/frame, for a near-identical match)
                  temporal_overlap  latent frames cross-faded between chunks (default 1)
                  frames_per_chunk  4n+1 pixel frames per chunk; absent = auto from free VRAM
                  seed              default 0
                  blend             0..1, share of the SeedVR2 result in the output (default 0.5).
                                    Below 1 the frame is mixed with the plain lanczos resize:
                                    SeedVR2 is a restoration model and on clean, soft AI footage
                                    (H3 768p -> 1080p) it etches fur and invents speckle; the
                                    2026-09-06 chihuahua A/B picked 0.5 (1.0 = raw SeedVR2).
    Audio and fps are carried over from the source clip; output is one mp4.
    """
    u = u or {}
    color = u.get("color_correction") or "wavelet"
    if color not in UPSCALE_COLOR_METHODS:
        raise ValueError(f"upscale.color_correction must be one of {UPSCALE_COLOR_METHODS}, got {color!r}")
    overlap = int(u.get("temporal_overlap", 1))
    blend = float(u.get("blend", 0.5))
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"upscale.blend must be in [0, 1], got {blend}")
    fpc = u.get("frames_per_chunk")
    if fpc is not None and (int(fpc) < 1 or (int(fpc) - 1) % 4):
        raise ValueError(f"upscale.frames_per_chunk must be 4n+1 (5, 9, 13, ...), got {fpc}")

    if u.get("shorter_size"):
        resize = {"resize_type": "scale shorter dimension",
                  "resize_type.shorter_size": int(u["shorter_size"])}
        how = f"short edge -> {int(u['shorter_size'])}px"
    else:
        factor = float(u.get("factor", 2.0))
        if not 1.0 < factor <= 4.0:
            raise ValueError(f"upscale.factor must be in (1, 4], got {factor}")
        resize = {"resize_type": "scale by multiplier", "resize_type.multiplier": factor}
        how = f"x{factor:g}"

    tiles = {"tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}
    g = {}
    g["load"] = {"class_type": "LoadVideo", "inputs": {"file": video_name}}
    g["parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}}
    g["resize"] = {"class_type": "ResizeImageMaskNode",
                   "inputs": {"input": ["parts", 0], "scale_method": "lanczos", **resize}}
    g["pre"] = {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["resize", 0]}}
    g["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": SEEDVR2_MODEL, "weight_dtype": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": SEEDVR2_VAE}}
    g["enc"] = {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0], **tiles}}
    chunk_inputs = {"latent": ["enc", 0], "temporal_overlap": overlap}
    if fpc is None:
        chunk_inputs["chunking_mode"] = "auto"
    else:
        chunk_inputs["chunking_mode"] = "manual"
        chunk_inputs["chunking_mode.frames_per_chunk"] = int(fpc)
    g["chunk"] = {"class_type": "SeedVR2TemporalChunk", "inputs": chunk_inputs}
    g["cond"] = {"class_type": "SeedVR2Conditioning",
                 "inputs": {"model": ["unet", 0], "vae_conditioning": ["chunk", 0]}}
    # One-step model: steps=1, cfg=1, euler/simple, exactly as the official template.
    g["ks"] = {"class_type": "KSampler",
               "inputs": {"model": ["unet", 0], "seed": int(u.get("seed", 0)), "steps": 1, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                          "positive": ["cond", 0], "negative": ["cond", 1], "latent_image": ["chunk", 0]}}
    g["merge"] = {"class_type": "SeedVR2TemporalMerge",
                  "inputs": {"latents": ["ks", 0], "temporal_overlap": ["chunk", 1]}}
    g["dec"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0], **tiles}}
    g["post"] = {"class_type": "SeedVR2PostProcessing",
                 "inputs": {"images": ["dec", 0], "original_resized_images": ["resize", 0],
                            "color_correction_method": color}}
    final = "post"
    if blend < 1.0:
        # Core ImageBlend (normal): image1 * (1 - f) + image2 * f.
        g["blend"] = {"class_type": "ImageBlend",
                      "inputs": {"image1": ["resize", 0], "image2": ["post", 0],
                                 "blend_factor": blend, "blend_mode": "normal"}}
        final = "blend"
    g["video"] = {"class_type": "CreateVideo",
                  "inputs": {"images": [final, 0], "fps": ["parts", 2], "audio": ["parts", 1]}}
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": filename_prefix, "format": "mp4"}}
    meta = {"upscale": how, "color_correction": color, "temporal_overlap": overlap,
            "frames_per_chunk": fpc or "auto", "blend": blend}
    return g, meta
