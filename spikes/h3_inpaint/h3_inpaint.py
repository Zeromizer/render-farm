"""Spike: MiniMax H3 masked inpainting of an existing clip (vlo-style).

Runs on the render PC through the farm's python engine. For each test in the
job JSON it:

  1. cuts a 17k+5-frame window from the source clip,
  2. crops it (crop mode: mask bbox + padding, scaled up to the generation
     canvas; full mode: whole frame), writes source / video-mask / audio-mask
     clips at the generation size,
  3. queues an API-format graph on the farm's ComfyUI:
       VAEEncode(src) + VAEEncodeAudio -> LTXVConcatAVLatent
       -> vloMaskToLatentMask -> SetLatentNoiseMask
       -> vloSetAudioLatentBinaryMasks (all-zero = keep audio)
       -> vloLatentCompositeMasked(source = blank) -> vloFeatherAudioLatentMask
       -> SamplerCustomAdvanced (BasicGuider, MiniMaxH3ReferenceToVideo cond)
     which mirrors vlo's vlo_minimax_h3_inpaint workflow minus its UI plumbing,
     Spectrum and attention-backend nodes,
  4. composites the generated patch back over the ORIGINAL frames with a
     feathered mask, so every pixel outside the mask is untouched,
  5. writes the patched clip, a side-by-side compare, and report.json.

Usage: python h3_inpaint.py <job.json> <out_dir>
Needs ComfyUI-vlo (8a7092a) in C:\\ComfyUI\\custom_nodes.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid

import cv2
import httpx
import numpy as np

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
FPS = 24

UNET = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

T = httpx.Timeout(30.0, read=600.0)


def log(*a):
    print(*a, flush=True)


def progress(p):
    print(f"PROGRESS {int(p)}", flush=True)


# ---------------------------------------------------------------- ffmpeg

def ffmpeg_bin(name="ffmpeg"):
    d = os.environ.get("FFMPEG_DIR")
    if d and os.path.exists(os.path.join(d, name + ".exe")):
        return os.path.join(d, name + ".exe")
    w = shutil.which(name)
    if w:
        return w
    pat = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*FFmpeg*\**\bin\%s.exe" % name)
    hits = glob.glob(pat, recursive=True)
    if hits:
        return hits[0]
    raise RuntimeError(f"{name} not found (FFMPEG_DIR / PATH / winget)")


FFMPEG = ffmpeg_bin()


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def write_video(path, frames, fps=FPS, audio_wav=None, lossless=False):
    h, w = frames[0].shape[:2]
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", audio_wav]
    if lossless:
        cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if audio_wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def extract_audio(src, start_s, dur_s, wav):
    """Window of the source audio as wav; silence if the source has none."""
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{start_s:.4f}", "-t", f"{dur_s:.4f}",
                        "-i", src, "-vn", "-ac", "2", "-ar", "48000", wav], capture_output=True)
    if r.returncode != 0 or not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                        "anullsrc=r=48000:cl=stereo", "-t", f"{dur_s:.4f}", wav], check=True)


# ---------------------------------------------------------------- geometry

def snap_len(n):
    """Largest 17k+5 <= n (vlo: max(5, a - ((a - 5) % 17)))."""
    return max(5, n - ((n - 5) % 17))


def gen_dims(aspect, area):
    w = math.sqrt(area * aspect)
    h = w / aspect
    W = max(256, int(round(w / 32)) * 32)
    H = max(256, int(round(h / 32)) * 32)
    return W, H


def mask_boxes_for_frame(masks, i):
    """All boxes (x0,y0,x1,y1, source px) active at absolute frame i."""
    out = []
    for m in masks:
        if "per_frame" in m:
            b = m["per_frame"].get(str(i))
            if b:
                pad = m.get("pad", 0)
                out.append((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))
        else:
            a, z = m["frames"]
            if a <= i <= z:
                out.append(tuple(m["box"]))
    return out


def union_bbox(masks, frames_range):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i in frames_range:
        for b in mask_boxes_for_frame(masks, i):
            xs0.append(b[0]); ys0.append(b[1]); xs1.append(b[2]); ys1.append(b[3])
    if not xs0:
        raise ValueError("mask is empty inside the window")
    return min(xs0), min(ys0), max(xs1), max(ys1)


def fit_region(bbox, pad, aspect, fw, fh):
    """Expand bbox+pad to `aspect`, keep it inside the frame (shift, then shrink)."""
    x0, y0, x1, y1 = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > fw:
        w, h = fw, fw / aspect
    if h > fh:
        h, w = fh, fh * aspect
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    rx0 = min(max(0, cx - w / 2), fw - w)
    ry0 = min(max(0, cy - h / 2), fh - h)
    return int(round(rx0)), int(round(ry0)), int(round(w)), int(round(h))


def raster_mask(boxes, fw, fh):
    m = np.zeros((fh, fw), np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        m[max(0, y0):min(fh, y1), max(0, x0):min(fw, x1)] = 255
    return m


# ---------------------------------------------------------------- comfy

def comfy_up():
    try:
        return httpx.get(COMFY + "/system_stats", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_comfy():
    """Wait (up to 4 min) for ComfyUI; never launch it, the worker owns that."""
    for _ in range(120):
        if comfy_up():
            return
        time.sleep(2)
    raise RuntimeError("ComfyUI did not answer /system_stats within 4 min")


def require_nodes():
    info = httpx.get(COMFY + "/object_info", timeout=60).json()
    need = ["vloMaskToLatentMask", "vloLatentCompositeMasked", "vloSetAudioLatentBinaryMasks",
            "vloFeatherAudioLatentMask", "MiniMaxH3ReferenceToVideo", "LTXVConcatAVLatent",
            "LatentMultiply", "SetLatentNoiseMask", "VAEEncodeAudio", "ImageToMask",
            "LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo"]
    missing = [n for n in need if n not in info]
    if missing:
        raise RuntimeError(f"ComfyUI is missing nodes: {missing}")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "h3_inpaint_spike", "type": "input", "overwrite": "true"},
                       timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def submit(graph):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:3000]}")
    return r.json()["prompt_id"]


def wait(pid, timeout_s, on_tick=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        h = httpx.get(COMFY + f"/history/{pid}", timeout=15).json()
        e = h.get(pid)
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: "
                                   f"{err.get('exception_message') or json.dumps(msgs)[:1500]}")
            return e.get("outputs") or {}
        if on_tick:
            on_tick(time.monotonic() - t0)
        time.sleep(3)
    httpx.post(COMFY + "/interrupt", timeout=10)
    raise RuntimeError(f"timed out after {timeout_s}s")


def fetch(outputs, dest):
    for node_out in outputs.values():
        for key in ("videos", "images", "gifs"):
            for f in node_out.get(key, []) or []:
                if f.get("type", "output") != "output" or not f["filename"].lower().endswith(".mp4"):
                    continue
                params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as out:
                        for chunk in r.iter_bytes(1 << 20):
                            out.write(chunk)
                return dest
    raise RuntimeError(f"no mp4 in outputs: {json.dumps(outputs)[:600]}")


def free_vram():
    try:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


def build_graph(src_name, mask_name, amask_name, prompt, W, H, length, steps, turbo, seed, prefix):
    g = {}
    g["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}}
    model = ["unet", 0]
    if turbo:
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model, "lora_name": TURBO_LORA, "strength_model": 1.0}}
        model = ["lora", 0]
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}

    g["src"] = {"class_type": "LoadVideo", "inputs": {"file": src_name}}
    g["src_parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["src", 0]}}
    g["vmask_v"] = {"class_type": "LoadVideo", "inputs": {"file": mask_name}}
    g["vmask_parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["vmask_v", 0]}}
    g["vmask"] = {"class_type": "ImageToMask", "inputs": {"image": ["vmask_parts", 0], "channel": "red"}}
    g["amask_v"] = {"class_type": "LoadVideo", "inputs": {"file": amask_name}}
    g["amask_parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["amask_v", 0]}}
    g["amask"] = {"class_type": "ImageToMask", "inputs": {"image": ["amask_parts", 0], "channel": "red"}}

    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["src_parts", 0], "vae": ["vae", 0]}}
    g["aenc"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["src_parts", 1], "vae": ["avae", 0]}}
    g["av"] = {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["enc", 0], "audio_latent": ["aenc", 0]}}
    g["latmask"] = {"class_type": "vloMaskToLatentMask",
                    "inputs": {"latent": ["av", 0], "vae": ["vae", 0], "masks": ["vmask", 0],
                               "pooling_method": "max", "resize_mode": "bilinear"}}
    g["noisemask"] = {"class_type": "SetLatentNoiseMask",
                      "inputs": {"samples": ["av", 0], "mask": ["latmask", 0]}}
    g["amasked"] = {"class_type": "vloSetAudioLatentBinaryMasks",
                    "inputs": {"audio_latent": ["noisemask", 0], "masks": ["amask", 0],
                               "mask_fps": 0.0, "threshold": 0.5, "resize_mode": "nearest",
                               "existing_mask_mode": "overwrite", "audio_vae": ["avae", 0],
                               "layout_override": "auto", "audio_latent_rate": 0.0}}
    g["blank"] = {"class_type": "LatentMultiply", "inputs": {"samples": ["amasked", 0], "multiplier": 0.0}}
    g["cleared"] = {"class_type": "vloLatentCompositeMasked",
                    "inputs": {"destination": ["amasked", 0], "source": ["blank", 0],
                               "clear_mask": False, "force_binary_mask": True}}
    g["feather"] = {"class_type": "vloFeatherAudioLatentMask",
                    "inputs": {"audio_latent": ["cleared", 0], "mode": "outer", "lead_ramp": 0.15,
                               "tail_ramp": 0.15, "lead_hold": 0.1, "tail_hold": 0.1, "curve": "cosine",
                               "floor": 0.0, "audio_vae": ["avae", 0], "layout_override": "auto",
                               "audio_latent_rate": 0.0}}

    g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0],
                            "prompt": prompt, "width": W, "height": H, "length": length,
                            "ref_image_size": "match"}}
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["feather", 0]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo",
                  "inputs": {"images": ["dec", 0], "fps": float(FPS), "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}}
    return g


# ---------------------------------------------------------------- one test

def run_test(t, src_path, src_frames, out_dir, report):
    name = t["name"]
    fh, fw = src_frames[0].shape[:2]
    start = int(t.get("window_start", 0))
    length = snap_len(int(t.get("window_frames", 124)))
    if start + length > len(src_frames):
        raise ValueError(f"{name}: window {start}+{length} exceeds {len(src_frames)} frames")
    win = range(start, start + length)
    masks = t["masks"]
    area = float(t.get("area", 768 * 768))

    if t.get("mode", "crop") == "crop":
        bbox = union_bbox(masks, win)
        bw, bh = bbox[2] - bbox[0] + 2 * t.get("pad", 96), bbox[3] - bbox[1] + 2 * t.get("pad", 96)
        W, H = gen_dims(bw / bh, area)
        rx, ry, rw, rh = fit_region(bbox, t.get("pad", 96), W / H, fw, fh)
    else:
        W, H = gen_dims(fw / fh, area)
        rx, ry, rw, rh = 0, 0, fw, fh
    log(f"[{name}] window {start}..{start + length - 1} ({length} f), region {rx},{ry} {rw}x{rh} -> gen {W}x{H}")

    work = os.path.join(out_dir, name)
    os.makedirs(work, exist_ok=True)
    src_crop, vmask, amask, full_masks = [], [], [], []
    for i in win:
        m = raster_mask(mask_boxes_for_frame(masks, i), fw, fh)
        full_masks.append(m)
        src_crop.append(cv2.resize(src_frames[i][ry:ry + rh, rx:rx + rw], (W, H), interpolation=cv2.INTER_AREA))
        mc = cv2.resize(m[ry:ry + rh, rx:rx + rw], (W, H), interpolation=cv2.INTER_NEAREST)
        vmask.append(cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR))
        amask.append(np.zeros((H, W, 3), np.uint8))
    wav = os.path.join(work, "src.wav")
    extract_audio(src_path, start / FPS, length / FPS, wav)
    p_src = os.path.join(work, "gen_src.mp4")
    p_vm = os.path.join(work, "gen_mask.mp4")
    p_am = os.path.join(work, "gen_amask.mp4")
    write_video(p_src, src_crop, audio_wav=wav, lossless=True)
    write_video(p_vm, vmask, lossless=True)
    write_video(p_am, amask, lossless=True)
    n_src, n_vm, n_am = upload(p_src), upload(p_vm), upload(p_am)

    feather = int(t.get("feather", 12))
    for v in t["variants"]:
        vname = f"{name}_{v['label']}"
        steps = int(v.get("steps", 20))
        turbo = bool(v.get("turbo", False))
        seed = int(v.get("seed", 6332))
        g = build_graph(n_src, n_vm, n_am, t["prompt"], W, H, length, steps, turbo, seed,
                        f"h3_inpaint_spike/{vname}")
        t0 = time.time()
        rec = {"test": name, "variant": v["label"], "steps": steps, "turbo": turbo, "seed": seed,
               "gen_w": W, "gen_h": H, "frames": length, "region": [rx, ry, rw, rh]}
        try:
            for attempt in (1, 2):
                pid = submit(g)
                log(f"[{vname}] queued {pid} (steps={steps} turbo={turbo} seed={seed}) attempt {attempt}")
                try:
                    outs = wait(pid, int(t.get("timeout_s", 3600)))
                    break
                except RuntimeError as e:
                    # transient --fast-disk error: free, wait, resubmit once
                    if attempt == 1 and "read_file_slice" in str(e):
                        log(f"[{vname}] transient read_file_slice error; retrying once")
                        free_vram()
                        time.sleep(5)
                        continue
                    raise
        except Exception as e:  # keep going with the other variants
            rec["error"] = str(e)[:2000]
            report.append(rec)
            log(f"[{vname}] FAILED: {e}")
            continue
        rec["seconds"] = round(time.time() - t0, 1)
        raw = os.path.join(work, f"{vname}_raw.mp4")
        fetch(outs, raw)
        gen = read_frames(raw)
        rec["gen_frames_returned"] = len(gen)

        # Composite: feathered mask in source coords, patch scaled back to the region.
        patched = [f.copy() for f in src_frames]
        kmax_diff = []
        for k, i in enumerate(win):
            if k >= len(gen):
                break
            gk = gen[k]
            # how faithfully did the noise mask keep the UNMASKED area? (gen space)
            um = vmask[k][..., 0] < 128
            if um.any():
                kmax_diff.append(float(np.abs(gk.astype(np.int16) - src_crop[k].astype(np.int16))[um].mean()))
            m = full_masks[k]
            if not m.any():
                continue
            if feather > 0:
                mf = cv2.GaussianBlur(cv2.dilate(m, np.ones((feather, feather), np.uint8)),
                                      (0, 0), feather / 2.0).astype(np.float32) / 255.0
            else:
                mf = m.astype(np.float32) / 255.0
            back = cv2.resize(gk, (rw, rh), interpolation=cv2.INTER_CUBIC)
            canvas = patched[i].astype(np.float32)
            reg = canvas[ry:ry + rh, rx:rx + rw]
            a = mf[ry:ry + rh, rx:rx + rw, None]
            canvas[ry:ry + rh, rx:rx + rw] = reg * (1 - a) + back.astype(np.float32) * a
            patched[i] = np.clip(canvas, 0, 255).astype(np.uint8)
        rec["unmasked_mae_gen_vs_src"] = round(float(np.mean(kmax_diff)), 2) if kmax_diff else None

        full_wav = os.path.join(work, "full.wav")
        if not os.path.exists(full_wav):
            extract_audio(src_path, 0, len(src_frames) / FPS, full_wav)
        out_mp4 = os.path.join(out_dir, f"{vname}_patched.mp4")
        write_video(out_mp4, patched, audio_wav=full_wav)
        # side by side: original | patched (full frame), plus region zoom for crop mode
        sbs = []
        for i in range(len(src_frames)):
            row = np.hstack([src_frames[i], patched[i]])
            sbs.append(cv2.resize(row, (row.shape[1] // 2 * 2 // 2, row.shape[0] // 2 * 2 // 2)))
        write_video(os.path.join(out_dir, f"{vname}_compare.mp4"), sbs)
        if t.get("mode", "crop") == "crop":
            zoom = []
            for i in win:
                a = src_frames[i][ry:ry + rh, rx:rx + rw]
                b = patched[i][ry:ry + rh, rx:rx + rw]
                z = np.hstack([a, b])
                s = 1080 / z.shape[1]
                zoom.append(cv2.resize(z, (int(z.shape[1] * s) // 2 * 2, int(z.shape[0] * s) // 2 * 2)))
            write_video(os.path.join(out_dir, f"{vname}_zoom.mp4"), zoom)
        report.append(rec)
        log(f"[{vname}] done in {rec['seconds']}s, unmasked MAE {rec['unmasked_mae_gen_vs_src']}")


def main():
    job = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    ensure_comfy()
    require_nodes()
    src_path = job["source"]
    src_frames = read_frames(src_path)
    log(f"source {src_path}: {len(src_frames)} frames {src_frames[0].shape[1]}x{src_frames[0].shape[0]}")
    report = []
    tests = job["tests"]
    total = sum(len(t["variants"]) for t in tests)
    done = 0
    try:
        for t in tests:
            run_test(t, src_path, src_frames, out_dir, report)
            done += len(t["variants"])
            progress(100 * done / total)
            json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    finally:
        json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
        free_vram()
    log(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
