"""API-format ComfyUI graph for MiniMax H3 + vlo Time-to-Move: a clip that follows a rough
cut-and-drag animation (the car cut out and dragged along a new path), for clip_edit's
move mode. Mirrors vlo's vlo_minimax_h3_ttm workflow (github.com/PxTicks/vlo) minus its UI
plumbing:

  LoadVideo(reference) -> VAEEncode ------------------------------\\
  LoadVideo(mask) -> ImageToMask -> ThresholdMask -> vloTimeToMove(model, reference, mask)
  LoadImage(first frame) -> MiniMaxH3ImageToVideo (fl2va) -> BasicGuider -> SamplerCustomAdvanced
  -> VAEDecode + VAEDecodeAudio -> SaveVideo

The reference seeds the sampler at start_step and the masked subject is held to it until
end_step; after that the whole frame denoises freely (TTM replaces any noise mask), so the
caller pastes back everything but the subject. Spike results (2026-10-10, 832x480 and
1344x768): turbo 8-step with ttm (1, 2)/(1, 3) tracks the path with a locked camera; the full
20-step model lags it; without TTM H3 invents a camera move.
"""
from videogen import graphs

REQUIRED_NODES = ("vloTimeToMove", "MiniMaxH3ImageToVideo", "VAEEncode", "LoadVideo", "GetVideoComponents",
                  "ImageToMask", "ThresholdMask", "SamplerCustomAdvanced")


def build(ref_name, mask_name, first_name, prompt, width, height, length, seed, prefix,
          steps=8, ttm=(1, 3)):
    """Return (graph, meta). width/height 32-aligned, length 17k+5; the reference and mask
    videos already uploaded at exactly width x height x length."""
    if width % 32 or height % 32:
        raise ValueError(f"canvas must be 32-aligned, got {width}x{height}")
    if (length - 5) % 17:
        raise ValueError(f"length must be on H3's 17k+5 grid, got {length}")
    if not (prompt or "").strip():
        raise ValueError("the move needs a prompt")
    start, end = int(ttm[0]), int(ttm[1])
    if not 1 <= start < int(steps):
        raise ValueError(f"ttm start step must be 1..{int(steps) - 1}")
    fam = "fl2va"
    lora = graphs.TURBO_LORAS[fam].get(int(steps)) or graphs.TURBO_LORAS[fam][graphs.DEFAULT_TURBO_STEPS[fam]]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": graphs.CHECKPOINTS[fam], "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": lora, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": graphs.TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": graphs.VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": graphs.AUDIO_VAE}},
        "ref_v": {"class_type": "LoadVideo", "inputs": {"file": ref_name}},
        "ref_p": {"class_type": "GetVideoComponents", "inputs": {"video": ["ref_v", 0]}},
        "m_v": {"class_type": "LoadVideo", "inputs": {"file": mask_name}},
        "m_p": {"class_type": "GetVideoComponents", "inputs": {"video": ["m_v", 0]}},
        "mask": {"class_type": "ImageToMask", "inputs": {"image": ["m_p", 0], "channel": "red"}},
        "mthr": {"class_type": "ThresholdMask", "inputs": {"mask": ["mask", 0], "value": 0.5}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["ref_p", 0], "vae": ["vae", 0]}},
        "first": {"class_type": "LoadImage", "inputs": {"image": first_name}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
                            "height": int(height), "length": int(length), "first_frame": ["first", 0]}},
        "ttm": {"class_type": "vloTimeToMove",
                "inputs": {"model": ["lora", 0], "reference_latents": ["enc", 0], "mask": ["mthr", 0],
                           "start_step": start, "end_step": end}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["ttm", 0], "conditioning": ["cond", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler",
                  "inputs": {"model": ["lora", 0], "scheduler": "simple", "steps": int(steps), "denoise": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["cond", 1]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": float(graphs.FPS), "audio": ["adec", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}},
    }
    return g, {"steps": int(steps), "ttm": [start, end], "seed": int(seed), "family": fam, "turbo": True}
