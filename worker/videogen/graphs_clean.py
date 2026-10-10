"""API-format ComfyUI graph that cleans one frame for clip_edit's anchor removal: Krea 2
Turbo masked img2img in the same headless ComfyUI as H3.

  LoadImage(frame) + LoadImage(mask) -> VAEEncodeForInpaint (masked pixels greyed, so the
  object is gone from the input) -> KSampler (cfg 1, euler/simple, 8 steps, denoise 1)
  -> VAEDecode -> SaveImage

H3 alone redraws an object the scene implies (a parked car's shadow, the plaza layout);
an image model with the object greyed out does not, and H3 then copies the cleaned frame
along the shot. Same checkpoint, text encoder and VAE as the relay's Krea 2 fallback.
"""
UNET = "krea2_turbo_fp8_scaled.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
REQUIRED_NODES = ("UNETLoader", "CLIPLoader", "VAELoader", "VAEEncodeForInpaint", "ImageToMask",
                  "ConditioningZeroOut", "KSampler", "SaveImage")
MAX_PIXELS = 2.0e6   # Krea 2 Turbo's 8 steps hold up to ~2 MP


def size_for(width, height):
    """Krea's working size for a frame: at most MAX_PIXELS, both edges on a 16 px grid."""
    s = min(1.0, (MAX_PIXELS / (width * height)) ** 0.5)
    return max(16, int(width * s) // 16 * 16), max(16, int(height * s) // 16 * 16)


def build(image_name, mask_name, prompt, seed, prefix, steps=8, grow=8):
    """Return the API prompt dict. image and mask (white = redraw) must already be uploaded
    at the same size, on the 16 px grid."""
    if not (prompt or "").strip():
        raise ValueError("cleaning needs a prompt describing what the area shows")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "8": {"class_type": "ImageToMask", "inputs": {"image": ["7", 0], "channel": "red"}},
        "9": {"class_type": "VAEEncodeForInpaint",
              "inputs": {"pixels": ["6", 0], "vae": ["3", 0], "mask": ["8", 0], "grow_mask_by": grow}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "seed": int(seed), "steps": int(steps), "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0],
                          "negative": ["5", 0], "latent_image": ["9", 0], "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }
