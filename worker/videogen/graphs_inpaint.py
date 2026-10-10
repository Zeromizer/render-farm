"""API-format ComfyUI graph for MiniMax H3 masked inpainting of an existing clip.

Mirrors vlo's vlo_minimax_h3_inpaint workflow (github.com/PxTicks/vlo) minus its UI
plumbing, Spectrum and attention-backend nodes:

  LoadVideo(src) -> VAEEncode + VAEEncodeAudio -> LTXVConcatAVLatent
  -> vloMaskToLatentMask -> SetLatentNoiseMask            (video: regenerate masked area only)
  -> vloSetAudioLatentBinaryMasks (all-zero mask = keep the audio)
  -> vloLatentCompositeMasked(source = blank latent)      (masked area starts from nothing)
  -> vloFeatherAudioLatentMask
  -> SamplerCustomAdvanced(BasicGuider, MiniMaxH3ReferenceToVideo conditioning) -> decode

Needs the ComfyUI-vlo custom node pack (PxTicks/ComfyUI-vlo @ 8a7092a, no pip deps) in
C:\\ComfyUI\\custom_nodes. REQUIRED_NODES lets the runner fail with a clear message
when it is missing instead of a /prompt 400.
"""
from videogen import graphs

FPS = graphs.FPS
REQUIRED_NODES = ("vloMaskToLatentMask", "vloLatentCompositeMasked", "vloSetAudioLatentBinaryMasks",
                  "vloFeatherAudioLatentMask", "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo",
                  "LTXVConcatAVLatent")


def build(src_name, mask_name, amask_name, prompt, width, height, length, seed, prefix,
          steps=None, turbo=True, family="ref2va", first_frame=None, last_frame=None):
    """Return the API prompt dict. width/height must be multiples of 32, length 17k+5,
    and the three inputs already uploaded at exactly width x height x length.

    family  ref2va (default: reference-to-video, what region edits use) or fl2va (the
            first/last-frame model, vlo's flf2va inpaint: made for in-betweens, so
            bridges and extensions); fl2va may also take first_frame / last_frame
            image names as extra conditioning."""
    if width % 32 or height % 32:
        raise ValueError(f"canvas must be 32-aligned, got {width}x{height}")
    if (length - 5) % 17:
        raise ValueError(f"length must be on H3's 17k+5 grid, got {length}")
    if not (prompt or "").strip():
        raise ValueError("inpaint needs a prompt")
    if family not in graphs.CHECKPOINTS:
        raise ValueError(f"family must be one of {sorted(graphs.CHECKPOINTS)}")
    fam = family
    g = {}
    g["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": graphs.CHECKPOINTS[fam], "weight_dtype": "default"}}
    model = ["unet", 0]
    if turbo:
        steps = int(steps or graphs.DEFAULT_TURBO_STEPS[fam])
        lora = graphs.TURBO_LORAS[fam].get(steps) or graphs.TURBO_LORAS[fam][graphs.DEFAULT_TURBO_STEPS[fam]]
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model, "lora_name": lora, "strength_model": 1.0}}
        model = ["lora", 0]
    else:
        steps = int(steps or graphs.DEFAULT_FULL_STEPS)
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": graphs.TEXT_ENCODER, "type": "minimax", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": graphs.VIDEO_VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": graphs.AUDIO_VAE}}

    for key, name in (("src", src_name), ("vm", mask_name), ("am", amask_name)):
        g[f"{key}_v"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        g[f"{key}_p"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"{key}_v", 0]}}
    g["vmask"] = {"class_type": "ImageToMask", "inputs": {"image": ["vm_p", 0], "channel": "red"}}
    g["amask"] = {"class_type": "ImageToMask", "inputs": {"image": ["am_p", 0], "channel": "red"}}

    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["src_p", 0], "vae": ["vae", 0]}}
    g["aenc"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["src_p", 1], "vae": ["avae", 0]}}
    g["av"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["enc", 0], "audio_latent": ["aenc", 0]}}
    g["latmask"] = {"class_type": "vloMaskToLatentMask",
                    "inputs": {"latent": ["av", 0], "vae": ["vae", 0], "masks": ["vmask", 0],
                               "pooling_method": "max", "resize_mode": "bilinear"}}
    g["noisemask"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["av", 0], "mask": ["latmask", 0]}}
    g["amasked"] = {"class_type": "vloSetAudioLatentBinaryMasks",
                    "inputs": {"audio_latent": ["noisemask", 0], "masks": ["amask", 0], "mask_fps": 0.0,
                               "threshold": 0.5, "resize_mode": "nearest", "existing_mask_mode": "overwrite",
                               "audio_vae": ["avae", 0], "layout_override": "auto", "audio_latent_rate": 0.0}}
    g["blank"] = {"class_type": "LatentMultiply", "inputs": {"samples": ["amasked", 0], "multiplier": 0.0}}
    g["cleared"] = {"class_type": "vloLatentCompositeMasked",
                    "inputs": {"destination": ["amasked", 0], "source": ["blank", 0],
                               "clear_mask": False, "force_binary_mask": True}}
    g["feather"] = {"class_type": "vloFeatherAudioLatentMask",
                    "inputs": {"audio_latent": ["cleared", 0], "mode": "outer", "lead_ramp": 0.15, "tail_ramp": 0.15,
                               "lead_hold": 0.1, "tail_hold": 0.1, "curve": "cosine", "floor": 0.0,
                               "audio_vae": ["avae", 0], "layout_override": "auto", "audio_latent_rate": 0.0}}

    if fam == "ref2va":
        g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                     "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0], "prompt": prompt,
                                "width": width, "height": height, "length": length, "ref_image_size": "match"}}
    else:
        cond = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                "width": width, "height": height, "length": length}
        for key, name in (("first_frame", first_frame), ("last_frame", last_frame)):
            if name:
                g[f"load_{key}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
                cond[key] = [f"load_{key}", 0]
        g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond}
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["feather", 0]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": float(FPS), "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}}
    return g, {"steps": steps, "turbo": turbo, "seed": int(seed), "family": fam}
