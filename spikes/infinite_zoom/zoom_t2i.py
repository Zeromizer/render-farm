"""Batch Krea 2 text-to-image for infinite-zoom levels (optionally with a subject LoRA).

params.args = [<job json string>, <out dir>]. Job JSON:
  {"W": 1088, "H": 1920, "steps": 8,
   "items": [{"name": "l1", "prompt": "...", "seeds": [1, 2, 3],
              "lora": null, "lora_strength": 0.8}]}
Writes <out>/<name>_s<seed>.png.
"""
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoom_level import COMFY, CLIP, UNET, VAE, clear_out, run_png


def graph(prompt, W, H, seed, steps, lora, lora_strength):
    model = ["1", 0]
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
    }
    if lora:
        g["14"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": lora, "strength_model": float(lora_strength)}}
        model = ["14", 0]
    g["8"] = {"class_type": "KSampler",
              "inputs": {"model": model, "seed": int(seed), "steps": int(steps), "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0}}
    g["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}
    g["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "izoom/t2i"}}
    return g


def main():
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    clear_out(out)
    W, H, steps = int(job.get("W", 1088)), int(job.get("H", 1920)), int(job.get("steps", 8))
    runs = [(it, s) for it in job["items"] for s in it.get("seeds", [1, 2, 3])]
    try:
        for i, (it, seed) in enumerate(runs):
            g = graph(it["prompt"], W, H, seed, steps, it.get("lora"), it.get("lora_strength", 0.8))
            secs = run_png(g, os.path.join(out, f"{it['name']}_s{seed}.png"))
            print(f"{it['name']} seed {seed}: {secs:.0f}s", flush=True)
            print(f"PROGRESS {int((i + 1) * 100 / len(runs))}", flush=True)
    finally:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)


if __name__ == "__main__":
    main()
