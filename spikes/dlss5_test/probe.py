"""Read-only probe of the PC ComfyUI: custom nodes, H3/DMAD/DLSS5 node classes, LoRA + model files."""
import json, os, sys, httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
out = sys.argv[1] if len(sys.argv) > 1 else "farm_out"
os.makedirs(out, exist_ok=True)
info = httpx.get(COMFY + "/object_info", timeout=120).json()
rep = {"comfy": COMFY, "n_classes": len(info)}
keys = ("dlss", "h3", "mmh3", "minimax", "latent_up", "upscal", "seedvr", "rife", "refmod", "dmad")
rep["matching_classes"] = sorted(k for k in info if any(s in k.lower() for s in keys))
def opts(cls, field):
    try:
        req = info[cls]["input"]["required"]
        return req[field][0]
    except Exception:
        return None
for cls, field in (("LoraLoader", "lora_name"), ("LoraLoaderModelOnly", "lora_name"),
                   ("UNETLoader", "unet_name"), ("CheckpointLoaderSimple", "ckpt_name"), ("VAELoader", "vae_name")):
    v = opts(cls, field)
    if isinstance(v, list):
        rep[cls] = v
try:
    rep["system_stats"] = httpx.get(COMFY + "/system_stats", timeout=30).json()
except Exception as e:
    rep["system_stats"] = str(e)
with open(os.path.join(out, "probe.json"), "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps({k: rep[k] for k in ("n_classes", "matching_classes")}, indent=1))
