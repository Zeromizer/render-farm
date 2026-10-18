"""Spike: pablodawson/MiniMax-H3-360-Orbit-LoRA on cars (FL2VA, first frame = last frame).

The LoRA (rank 16, trained on 28 human Gaussian-splat orbits, 768x768, 73 frames, 28 steps) makes FL2VA
orbit the camera a full 360 around a frozen subject and land back on the start frame. This measures:

  human_ctrl          its own domain (frame 0 of the author's man_lora.mp4): does it load on our fp8 base?
  <car>_sq            single photo -> 360, the card's settings (768x768, 73 f, 28 steps)
  <car>_sq_turbo      the same stacked on our fl2v turbo-8 LoRA (8 steps)
  noah_wide           16:9 (1344x768)
  noah_sq_124         124 frames (5.2 s)
  noah_guided_*       front at 0 and N-1 + left / rear / right pinned with core MiniMaxH3AddGuide at 1/4, 1/2, 3/4
                      (the real side and rear views instead of a guess); *_nolora is the control
  noah_q1_*           one turntable quarter (front -> left, 1344x768, 124 f, turbo-8): LoRA vs today's prompt

Inputs (assets manifest -> spikes/h3_orbit/in/): noah_{front,left,rear,right}.png, emas7_{...}.png,
emas5_{front34,side,rear}.png (transparent cutouts, flattened on grey here).

params.args = [<job json>, <out dir>]; job json (optional): {"arms": [...], "seed": 4242}
Writes <out>/<arm>.mp4 (no audio), <arm>_sheet.jpg, <arm>_views.jpg, report.json.
"""
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import uuid

import cv2
import httpx
import numpy as np

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
T = httpx.Timeout(30.0, read=600.0)
FPS = 24
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "in")

HF = "https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/resolve/main/"
ORBIT_LORA = "minimax_h3_360_orbit_flf2v_v1.safetensors"   # local name of minimax_h3_flf2v_lora_v1.safetensors
UNET = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
TURBO8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TEXT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"

ORBIT_PROMPT = (
    "One frozen instant. Only the camera moves. In a continuous 360 orbit. Preserve every person and object in "
    "exactly the same world position, orientation, shape and pose throughout the shot. Airborne objects remain "
    "suspended at the captured height and angle: no wobbling, shaking, spinning, drifting, falling or continued "
    "action. Keep faces, hands, clothing, liquids and the background motionless while retaining their natural "
    "appearance. Camera parallax is the only source of apparent movement. No cuts, zoom, morphing or added objects.")
QUARTER_PROMPT = ORBIT_PROMPT.replace("In a continuous 360 orbit.", "In a continuous quarter orbit, a 90 degree arc.")
TT_PROMPT = (  # studio/turntable.py BASE_PROMPT for front_to_left with the Noah job's car line
    "Toyota Noah Hybrid 1.8X exact supplied white standard-body visual on a seamless plain light studio background, "
    "exactly as in the given frames. One single rigid car, rotating in place about its own vertical axis at a "
    "perfectly constant, slow angular speed, no easing in or out, no pauses, no speed changes, turning clockwise as "
    "seen from above, from the front view to the left side view. Its paint colour, rims, tyres, roofline, doors, "
    "windows, lamps, badges and spoiler stay exactly as in the frames throughout. The car stays the same size and "
    "stays centred; the whole car including all wheels stays fully inside the frame with empty space on both sides. "
    "Fixed camera at eye level, fixed framing, no camera movement, no zoom. No turntable, no floor, no ground "
    "shadow, no reflection, no horizon, just the plain background. Soft even studio lighting that does not change, "
    "no people, no text overlays. Quiet studio ambience.")

SQ = dict(w=768, h=768, n=73, steps=28, turbo=False, orbit=True, guides=False, prompt=ORBIT_PROMPT)
ARMS = {
    "human_ctrl": dict(SQ, car="human"),
    "noah_sq": dict(SQ, car="noah"),
    "emas7_sq": dict(SQ, car="emas7"),
    "emas5_sq": dict(SQ, car="emas5"),
    "noah_sq_turbo": dict(SQ, car="noah", steps=8, turbo=True),
    "emas7_sq_turbo": dict(SQ, car="emas7", steps=8, turbo=True),
    "noah_wide": dict(SQ, car="noah", w=1344),
    "noah_sq_124": dict(SQ, car="noah", n=124),
    "noah_guided_sq": dict(SQ, car="noah", guides=True),
    "noah_guided_wide_turbo": dict(SQ, car="noah", w=1344, n=243, steps=8, turbo=True, guides=True),
    "noah_guided_wide_turbo_nolora": dict(SQ, car="noah", w=1344, n=243, steps=8, turbo=True, guides=True, orbit=False),
    "noah_q1_lora": dict(SQ, car="noah", w=1344, n=124, steps=8, turbo=True, quarter=True, prompt=QUARTER_PROMPT),
    "noah_q1_base": dict(SQ, car="noah", w=1344, n=124, steps=8, turbo=True, quarter=True, orbit=False, prompt=TT_PROMPT),
    # round 2: 1344x768, pins = ready canvases in/pins/<set>_<deg>.png (prep_pins.py on the laptop): orig = the 4
    # photos, toy = Toyota's 12 configurator frames (30 deg apart), mix = photos at 0/90/180/270 + Toyota between
    "r2_mix12_124": dict(SQ, car="noah", w=1344, n=124, pins="mix", guides=True),
    "r2_orig4_124": dict(SQ, car="noah", w=1344, n=124, pins="orig", guides=True),
    "r2_toy12_124": dict(SQ, car="noah", w=1344, n=124, pins="toy", guides=True),
    "r2_mix12_243": dict(SQ, car="noah", w=1344, n=243, steps=20, pins="mix", guides=True),
    # round 3: one consistent pin source (Toyota only) at the slow speed that keeps lettering
    "r3_toy12_243_turbo": dict(SQ, car="noah", w=1344, n=243, steps=8, turbo=True, pins="toy", guides=True),
    "r3_toy12_243": dict(SQ, car="noah", w=1344, n=243, steps=20, pins="toy", guides=True),
}
VIEWS = ("front", "left", "rear", "right")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- tools

def _ffmpeg(name):
    found = shutil.which(name)
    if found:
        return found
    d = os.environ.get("FFMPEG_DIR")
    if d and os.path.exists(os.path.join(d, name + ".exe")):
        return os.path.join(d, name + ".exe")
    pat = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages", "*ffmpeg*", "**", f"{name}.exe")
    hits = glob.glob(pat, recursive=True)
    if hits:
        return hits[0]
    raise RuntimeError(f"{name} not found")


def ff(*args):
    r = subprocess.run([_ffmpeg("ffmpeg"), "-v", "error", "-y", *map(str, args)],
                       capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-1500:]}")


def download(url, dest):
    tmp = dest + ".part"
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)


# ---------------------------------------------------------------- ComfyUI

def comfy_up():
    try:
        return httpx.get(COMFY + "/system_stats", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def launch_comfy(wait_s=600):
    if not comfy_up():
        bat = os.path.join(COMFY_DIR, "run-headless.bat")
        log(f"launching {bat}")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | NO_WINDOW
        try:  # outlive this job's process tree if the job object allows it
            subprocess.Popen(["cmd", "/c", bat], cwd=COMFY_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=flags | 0x01000000)  # CREATE_BREAKAWAY_FROM_JOB
        except OSError:
            subprocess.Popen(["cmd", "/c", bat], cwd=COMFY_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=flags)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if comfy_up():
            return
        time.sleep(3)
    raise RuntimeError("ComfyUI did not come up")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "h3_orbit", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_outputs(graph, timeout_s=2400):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:3000]}")
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
            return e.get("outputs") or {}
        time.sleep(3)
    raise RuntimeError("timed out")


def fetch_mp4(outputs, dest):
    for node_out in outputs.values():
        for key in ("videos", "images", "gifs"):
            for f in node_out.get(key, []) or []:
                name = f.get("filename", "")
                if f.get("type", "output") != "output" or not name.lower().endswith(".mp4"):
                    continue
                params = {"filename": name, "subfolder": f.get("subfolder", ""), "type": "output"}
                with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as out:
                        for chunk in resp.iter_bytes(1 << 20):
                            out.write(chunk)
                return dest
    raise RuntimeError("no mp4 in the outputs")


def comfy_log_size():
    p = os.path.join(COMFY_DIR, "comfyui-headless.log")
    return os.path.getsize(p) if os.path.exists(p) else 0


def comfy_log_since(pos):
    p = os.path.join(COMFY_DIR, "comfyui-headless.log")
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        f.seek(min(pos, os.path.getsize(p)))
        return f.read().decode("utf-8", "replace")


def ensure_lora():
    d = os.path.join(COMFY_DIR, "models", "loras")
    dest = os.path.join(d, ORBIT_LORA)
    if not os.path.exists(dest):
        os.makedirs(d, exist_ok=True)
        log(f"downloading the orbit LoRA to {dest}")
        download(HF + "minimax_h3_flf2v_lora_v1.safetensors", dest)
    with open(dest, "rb") as f:   # safetensors header: u64 length + JSON
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    keys = [k for k in hdr if k != "__metadata__"]
    info = {"bytes": os.path.getsize(dest), "tensors": len(keys), "sample_keys": sorted(keys)[:4],
            "dtypes": sorted({v["dtype"] for k, v in hdr.items() if k != "__metadata__"})}
    log(f"orbit LoRA: {info}")
    return info


# ---------------------------------------------------------------- inputs

def flatten(path, bg=(200, 200, 200)):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"cannot read {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        im = (im[:, :, :3].astype(np.float32) * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)
    return im


def border_colour(im):
    b = np.concatenate([im[:8].reshape(-1, 3), im[-8:].reshape(-1, 3), im[:, :8].reshape(-1, 3), im[:, -8:].reshape(-1, 3)])
    return np.median(b, axis=0)


def subject_bbox(im, bg):
    d = np.abs(im.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    m = cv2.morphologyEx((d > 28).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m)
    if not len(xs):
        return 0, 0, im.shape[1], im.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def car_photos(car):
    if car == "emas5":
        paths = {"front": "emas5_front34.png", "left": "emas5_side.png", "rear": "emas5_rear.png"}
    else:
        paths = {v: f"{car}_{v}.png" for v in VIEWS}
    return {v: flatten(os.path.join(IN, p)) for v, p in paths.items() if os.path.exists(os.path.join(IN, p))}


def pad_set(photos, w, h, fill_w=0.86, fill_h=0.70):
    """Crop each photo to its subject and place all of them on a w x h canvas at ONE common scale
    (the one that fits the widest view), centred, on the front photo's background colour."""
    bg = border_colour(photos["front"])
    crops = {}
    for v, im in photos.items():
        x0, y0, x1, y1 = subject_bbox(im, border_colour(im))
        crops[v] = im[y0:y1, x0:x1]
    s = min(min(fill_w * w / c.shape[1], fill_h * h / c.shape[0]) for c in crops.values())
    out = {}
    for v, c in crops.items():
        cw, ch = max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))
        r = cv2.resize(c, (cw, ch), interpolation=cv2.INTER_AREA)
        can = np.empty((h, w, 3), np.uint8)
        can[:] = bg.astype(np.uint8)
        x, y = (w - cw) // 2, (h - ch) // 2
        can[y:y + ch, x:x + cw] = r
        out[v] = can
    return out, s


def human_frame(work):
    mp4 = os.path.join(work, "man_lora.mp4")
    if not os.path.exists(mp4):
        download(HF + "media/man_lora.mp4", mp4)
    cap = cv2.VideoCapture(mp4)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot read man_lora.mp4")
    return cv2.resize(fr, (768, 768), interpolation=cv2.INTER_AREA) if fr.shape[:2] != (768, 768) else fr


# ---------------------------------------------------------------- graph

def build(a, first, last, guides, seed, prefix):
    g = {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
         "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT, "type": "minimax", "device": "default"}},
         "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
         "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}}}
    model = ["unet", 0]
    if a["turbo"]:
        g["turbo"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model, "lora_name": TURBO8, "strength_model": 1.0}}
        model = ["turbo", 0]
    if a["orbit"]:
        g["orbit"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model, "lora_name": ORBIT_LORA, "strength_model": float(a.get("strength", 1.0))}}
        model = ["orbit", 0]
    g["first"] = {"class_type": "LoadImage", "inputs": {"image": first}}
    g["last"] = {"class_type": "LoadImage", "inputs": {"image": last}}
    g["cond"] = {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": a["prompt"], "width": a["w"],
                            "height": a["h"], "length": a["n"], "first_frame": ["first", 0], "last_frame": ["last", 0]}}
    positive, latent = ["cond", 0], ["cond", 1]
    for i, (name, idx) in enumerate(guides):
        g[f"gimg{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[f"guide{i}"] = {"class_type": "MiniMaxH3AddGuide",
                          "inputs": {"positive": positive, "latent": latent, "vae": ["vae", 0],
                                     "image": [f"gimg{i}", 0], "frame_idx": int(idx)}}
        positive = [f"guide{i}", 0]
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": positive}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": model, "scheduler": "simple", "steps": a["steps"], "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": latent}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": float(FPS)}}
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}}
    return g


# ---------------------------------------------------------------- measure

def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def measure(frames, refs, bg):
    n = len(frames)
    sw = 320
    sh = round(frames[0].shape[0] * sw / frames[0].shape[1])
    small = [cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA) for f in frames]
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]
    bgs = bg.astype(np.int16)

    def speed(a, b):
        fl = cv2.calcOpticalFlowFarneback(grey[a], grey[b], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        mag = np.linalg.norm(fl, axis=2)
        m = (np.abs(small[a].astype(np.int16) - bgs).max(axis=2) > 20) | (np.abs(small[b].astype(np.int16) - bgs).max(axis=2) > 20)
        return float(mag[m].mean()) if m.sum() > 50 else float(mag.mean())

    sp = [speed(i, i + 1) for i in range(n - 1)]
    core = np.array(sp[3:-3] if n > 12 else sp)
    med = max(float(np.median(core)), 0.05)
    # the clip ends on its first frame, so the loop drops the last frame: seam = frame n-2 -> frame 0
    seam = speed(n - 2, 0) / med
    lab = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0].mean() * 100 / 255 for f in small]
    dl = np.diff(lab)
    flashes = [int(i + 1) for i in range(1, len(dl)) if abs(dl[i - 1]) > 2.5 and abs(dl[i]) > 2.5 and np.sign(dl[i - 1]) != np.sign(dl[i])]
    views = {}
    for v, ref in refs.items():
        r = cv2.resize(ref, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.int16)
        maes = [float(np.abs(s.astype(np.int16) - r).mean()) for s in small]
        maes_flip = [float(np.abs(s.astype(np.int16) - r[:, ::-1]).mean()) for s in small]
        i = int(np.argmin(maes))
        views[v] = {"best_frame": i, "mae": round(maes[i], 2), "mae_flip": round(min(maes_flip), 2),
                    "best_frame_flip": int(np.argmin(maes_flip))}
    return {"frames": n, "speed_median_px": round(med, 2),
            "speed_min_x": round(float(core.min()) / med, 2), "speed_max_x": round(float(core.max()) / med, 2),
            "stall_frames_lt_0p4": int((core < 0.4 * med).sum()),
            "speed_x": [round(s / med, 2) for s in sp],
            "loop_seam_x": round(seam, 2),
            "closure_mae_first_last": round(float(np.abs(small[0].astype(np.int16) - small[-1].astype(np.int16)).mean()), 2),
            "luma_first_mid": [round(lab[0], 1), round(lab[n // 2], 1)], "luma_flash_frames": flashes,
            "views": views}


def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(frames, dest, cols=6, k=12, width=360):
    n = len(frames)
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    h = round(frames[0].shape[0] * width / frames[0].shape[1])
    tiles = [label(cv2.resize(frames[i], (width, h), interpolation=cv2.INTER_AREA), f"f{i}") for i in idx]
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, k, cols)]
    cv2.imwrite(dest, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def views_sheet(frames, refs, m, dest, width=360):
    h = round(frames[0].shape[0] * width / frames[0].shape[1])
    cols = []
    for v, ref in refs.items():
        i = m["views"][v]["best_frame"]
        a = label(cv2.resize(ref, (width, h), interpolation=cv2.INTER_AREA), f"photo {v}")
        b = label(cv2.resize(frames[i], (width, h), interpolation=cv2.INTER_AREA), f"f{i} mae {m['views'][v]['mae']}")
        cols.append(np.vstack([a, b]))
    if cols:
        cv2.imwrite(dest, np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- main

def main():
    job = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].strip() else {}
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    seed = int(job.get("seed", 4242))
    names = job.get("arms") or list(ARMS)
    report = {"seed": seed, "arms": {}}
    rpath = os.path.join(out, "report.json")

    def save():
        with open(rpath, "w") as f:
            json.dump(report, f, indent=1)

    launch_comfy()
    report["lora"] = ensure_lora()
    save()
    for k, name in enumerate(names):
        a = ARMS[name]
        log(f"PROGRESS {int(100 * k / len(names))}")
        log(f"== {name}: {a['w']}x{a['h']} {a['n']} f, {a['steps']} steps, turbo {a['turbo']}, orbit {a['orbit']}, "
            f"guides {a['guides']}")
        rec = {k2: a[k2] for k2 in ("car", "w", "h", "n", "steps", "turbo", "orbit", "guides")}
        rec["quarter"] = bool(a.get("quarter"))
        try:
            if a.get("pins"):
                canv = {}
                for p in sorted(glob.glob(os.path.join(IN, "pins", f"{a['pins']}_*.png"))):
                    deg = int(os.path.basename(p)[:-4].rsplit("_", 1)[1])
                    canv["front" if deg == 0 else f"d{deg:03d}"] = cv2.imread(p)
            elif a["car"] == "human":
                canv = {"front": human_frame(work)}
            else:
                canv, scale = pad_set(car_photos(a["car"]), a["w"], a["h"])
                rec["scale"] = round(scale, 4)
            names_up = {}
            for v, im in canv.items():
                p = os.path.join(work, f"{a['car']}_{a['w']}x{a['h']}_{v}.png")
                cv2.imwrite(p, im)
                names_up[v] = upload(p)
            first = names_up["front"]
            last = names_up["left"] if a.get("quarter") else names_up["front"]
            guides = []
            if a.get("pins"):
                for v in sorted(k2 for k2 in names_up if k2 != "front"):
                    guides.append((names_up[v], round(int(v[1:]) * (a["n"] - 1) / 360)))
            elif a["guides"]:
                for q, v in ((1, "left"), (2, "rear"), (3, "right")):
                    guides.append((names_up[v], round(q * (a["n"] - 1) / 4)))
            rec["guide_frames"] = [i for _, i in guides]
            g = build(a, first, last, guides, seed, f"h3_orbit/{name}")
            pos = comfy_log_size()
            t0 = time.monotonic()
            raw = fetch_mp4(run_outputs(g), os.path.join(work, f"{name}_raw.mp4"))
            rec["gen_s"] = round(time.monotonic() - t0, 1)
            txt = comfy_log_since(pos)
            rec["lora_keys_not_loaded"] = txt.count("lora key not loaded")
            rec["comfy_warnings"] = [l for l in txt.splitlines() if "lora" in l.lower() or "error" in l.lower()][:8]
            mp4 = os.path.join(out, f"{name}.mp4")
            # untagged-colour trap: copy the stream as-is, drop nothing but any audio
            ff("-i", raw, "-map", "0:v:0", "-c", "copy", mp4)
            frames = read_frames(mp4)
            refs = {v: canv[v] for v in canv}
            bg = border_colour(canv["front"])
            small_bg = bg
            m = measure(frames, refs, small_bg)
            rec.update(m)
            sheet(frames, os.path.join(out, f"{name}_sheet.jpg"))
            views_sheet(frames, refs, m, os.path.join(out, f"{name}_views.jpg"))
            log(f"   {name}: {rec['gen_s']} s, speed min {m['speed_min_x']}x max {m['speed_max_x']}x, "
                f"stalls {m['stall_frames_lt_0p4']}, seam {m['loop_seam_x']}, views "
                + ", ".join(f"{v} f{d['best_frame']} {d['mae']}" for v, d in m["views"].items())
                + f", lora misses {rec['lora_keys_not_loaded']}")
        except Exception as e:  # keep going: one failed arm should not cost the others
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"   {name} FAILED: {rec['error']}")
        report["arms"][name] = rec
        save()
    shutil.rmtree(work, ignore_errors=True)
    log("PROGRESS 100")


if __name__ == "__main__":
    main()
