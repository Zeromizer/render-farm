"""Still upscale with Krea 2 on the render PC's ComfyUI: Lanczos the image up to ~8 MP on the
16 px grid, then a low-denoise Krea 2 Turbo img2img pass redraws fine detail at that size.

params.args = [<job json string>, <out dir>]. Job JSON:
  {"items": [{"image": "spikes/krea_upscale/in/front.png", "prompt": "..."}],
   "denoise": [0.25, 0.4], "seed": 1, "megapixels": 8}
Writes <out>/<name>_d<denoise>.png (plus <name>_lanczos.png, the pre-pass input).

Steps scale with pixels (8 steps smear backgrounds above 2 MP): 4 MP 20, 8 MP 28.
"""
import json
import os
import shutil
import sys
import time
import uuid

import httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)
UNET = "krea2_turbo_fp8_scaled.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"


def steps_for(mp):
    return 8 if mp <= 2.2 else 20 if mp <= 4.5 else 28


def size_for(w, h, mp):
    s = (mp * 1e6 / (w * h)) ** 0.5
    return int(w * s) // 16 * 16, int(h * s) // 16 * 16


def graph(image_name, prompt, seed, steps, denoise, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": int(seed), "steps": int(steps), "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["7", 0], "denoise": float(denoise)}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "krea_up", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_png(g, dest, timeout_s=1800):
    r = httpx.post(COMFY + "/prompt", json={"prompt": g, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:2000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            e = httpx.get(COMFY + f"/history/{pid}", timeout=60).json().get(pid)
        except httpx.TimeoutException:
            continue
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}")
            for node_out in (e.get("outputs") or {}).values():
                for f in node_out.get("images", []) or []:
                    if f.get("type", "output") == "output":
                        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                        with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                            resp.raise_for_status()
                            with open(dest, "wb") as out:
                                for chunk in resp.iter_bytes(1 << 20):
                                    out.write(chunk)
                        return time.monotonic() - t0
            raise RuntimeError("no image in outputs")
        time.sleep(2)
    raise RuntimeError("timed out")


def clear_out(out):
    """The output dir sits in a reused repo cache and is zipped whole: start it empty."""
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)


def main():
    import cv2
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    clear_out(out)
    mp = float(job.get("megapixels", 8))
    steps = int(job.get("steps") or steps_for(mp))
    denoises = job.get("denoise", [0.25, 0.4])
    seed = int(job.get("seed", 1))
    items = job["items"]
    total, done = len(items) * len(denoises), 0
    try:
        for it in items:
            name = os.path.splitext(os.path.basename(it["image"]))[0]
            src = cv2.imread(it["image"], cv2.IMREAD_COLOR)
            if src is None:
                raise RuntimeError(f"cannot read {it['image']}")
            h, w = src.shape[:2]
            W, H = size_for(w, h, mp)
            big = cv2.resize(src, (W, H), interpolation=cv2.INTER_LANCZOS4)
            lz = os.path.join(out, f"{name}_lanczos.png")
            cv2.imwrite(lz, big)
            ref = upload(lz)
            for d in denoises:
                secs = run_png(graph(ref, it["prompt"], seed, steps, d, f"krea_up/{name}"),
                               os.path.join(out, f"{name}_d{int(round(d * 100)):02d}.png"))
                done += 1
                print(f"{name} {W}x{H} denoise {d} steps {steps}: {secs:.0f}s", flush=True)
                print(f"PROGRESS {int(done * 100 / total)}", flush=True)
    finally:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)


if __name__ == "__main__":
    main()
