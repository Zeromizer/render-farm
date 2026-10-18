"""Spike: does MiniMax's official H3 prompt format beat the prose our agents write?

Replays real production video_gen jobs (cases.json: same first/last frames, seed, 768p, turbo-8, duration)
through the exact production graph (worker/videogen/graphs.py, audio included) with three prompts:

  A   the prompt the platform agent actually sent, verbatim
  AL  MiniMax's alignment line + A (the one-line change)
  B   A rewritten in the guide format (alignment line, integrated_multimodal_description,
      overall_soundscape, non_diegetic_music)

A and B run on two seeds (the job's own seed and seed+1000), AL on the job's seed only.
Inputs (assets manifest -> spikes/h3_prompt_ab/in/).

params.args = [<job json>, <out dir>]; job json: {"cases": ["turntable", ...], "seeds": 2}
Writes <out>/<case>__<arm>__s<seed>.mp4 (with audio), <case>_sheet.jpg, <case>_end.jpg, <clip>_spec.png,
report.json.
"""
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import wave

import cv2
import httpx
import numpy as np

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
T = httpx.Timeout(30.0, read=600.0)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IN = os.path.join(HERE, "in")

_spec = importlib.util.spec_from_file_location("graphs", os.path.join(ROOT, "worker", "videogen", "graphs.py"))
graphs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graphs)
FPS = graphs.FPS


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
                       data={"subfolder": "h3_prompt_ab", "type": "input", "overwrite": "true"}, timeout=T)
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


def fit(img, w, h, crop):
    """Mirror MiniMaxH3ImageToVideo: first frame is stretched, last frame is cover-cropped."""
    if not crop:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def measure(frames, first_img, last_img):
    n = len(frames)
    sw = 320
    sh = round(frames[0].shape[0] * sw / frames[0].shape[1])
    small = [cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA) for f in frames]
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]
    m = {"frames": n, "seconds": round(n / FPS, 2)}

    # camera motion: median flow vector in the border band (top 15 % + left/right 8 %), i.e. the background
    band = np.zeros((sh, sw), bool)
    band[: int(sh * 0.15)] = True
    band[:, : int(sw * 0.08)] = True
    band[:, -int(sw * 0.08):] = True
    vec, mags = np.zeros(2), []
    for i in range(n - 1):
        fl = cv2.calcOpticalFlowFarneback(grey[i], grey[i + 1], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        med = np.median(fl[band], axis=0)
        vec += med
        mags.append(float(np.hypot(*med)))
    mags = np.array(mags)
    m["cam_px_per_frame_median"] = round(float(np.median(mags)), 3)
    m["cam_moving_frac"] = round(float((mags > 0.25).mean()), 3)      # share of frame pairs with a coherent shift
    m["cam_net_drift_px"] = [round(float(v), 1) for v in vec]          # summed (dx, dy) at 320 px wide

    diffs = np.array([mae(small[i], small[i + 1]) for i in range(n - 1)])
    core = diffs[2:-2] if n > 10 else diffs
    medd = max(float(np.median(core)), 0.05)
    m["motion_median_mae"] = round(medd, 2)
    m["max_step_x"] = round(float(core.max()) / medd, 2)               # jerks / pops mid-shot
    if first_img is not None:
        m["first_mae"] = round(mae(small[0], cv2.resize(fit(first_img, frames[0].shape[1], frames[0].shape[0], False),
                                                         (sw, sh), interpolation=cv2.INTER_AREA)), 2)
    if last_img is not None:
        ref = cv2.resize(fit(last_img, frames[0].shape[1], frames[0].shape[0], True), (sw, sh), interpolation=cv2.INTER_AREA)
        to_ref = [mae(s, ref) for s in small]
        m["last_mae"] = round(to_ref[-1], 2)
        m["pre_last_mae"] = round(to_ref[-2], 2)
        m["end_snap_x"] = round(diffs[-1] / medd, 2)                   # a jump onto the keyframe at the very end
        m["end_tail_mae_6f"] = round(float(np.mean(to_ref[-7:-1])), 2)  # how close the last quarter-second already is
        thr = max(4.0, to_ref[-1] + 3.0)
        conv = n - 1
        for i in range(n - 1, -1, -1):
            if to_ref[i] > thr:
                break
            conv = i
        m["converged_at_s"] = round(conv / FPS, 2)                    # from here on the frame stays near the keyframe
    return m


def audio_stats(mp4, wav, png):
    try:
        ff("-i", mp4, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav)
    except RuntimeError:
        return {"audio": "none"}
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if not len(x):
        return {"audio": "empty"}
    rms = float(np.sqrt((x ** 2).mean()) + 1e-9)
    nfft, hop = 1024, 256
    win = np.hanning(nfft).astype(np.float32)
    frames = [x[i:i + nfft] * win for i in range(0, len(x) - nfft, hop)]
    spec = np.abs(np.fft.rfft(np.array(frames), axis=1)) + 1e-7
    flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    # sustained tonal lines (music / hum): bins >12 dB over the frame's median for >= 1 s straight
    db = 20 * np.log10(spec)
    peak = db > (np.median(db, axis=1, keepdims=True) + 12)
    run, best = np.zeros(peak.shape[1], int), np.zeros(peak.shape[1], int)
    for row in peak:
        run = np.where(row, run + 1, 0)
        best = np.maximum(best, run)
    sustained = int((best >= 16000 // hop).sum())
    img = db.T[::-1]
    img = np.clip((img - img.max() + 80) / 80, 0, 1)
    cv2.imwrite(png, cv2.applyColorMap((cv2.resize(img, (600, 256)) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA))
    return {"rms_dbfs": round(20 * np.log10(rms), 1), "flatness_median": round(float(np.median(flat)), 4),
            "sustained_tonal_bins": sustained}


def label(img, text, scale=0.5):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def row(frames, idx, width, name):
    h = round(frames[0].shape[0] * width / frames[0].shape[1])
    tiles = [label(cv2.resize(frames[i], (width, h), interpolation=cv2.INTER_AREA), f"{i / FPS:.2f}s") for i in idx]
    r = np.hstack(tiles)
    bar = np.full((22, r.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, r])


# ---------------------------------------------------------------- main

def main():
    job = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].strip() else {}
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        cases = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    names = job.get("cases") or list(cases)
    nseeds = int(job.get("seeds", 2))
    rpath = os.path.join(out, "report.json")
    report = {"cases": {}}

    def save():
        with open(rpath, "w") as f:
            json.dump(report, f, indent=1)

    plan = []
    for c in names:
        cs = cases[c]
        seeds = [cs["seed"] + 1000 * k for k in range(nseeds)]
        for s in seeds:
            for arm in ("A", "AL", "B"):
                if arm == "AL" and s != seeds[0]:
                    continue
                plan.append((c, arm, s))
    log(f"{len(plan)} clips: " + ", ".join(f"{c}/{a}/s{s}" for c, a, s in plan))

    launch_comfy()
    clips = {}
    for k, (c, arm, seed) in enumerate(plan):
        log(f"PROGRESS {int(100 * k / len(plan))}")
        cs = cases[c]
        name = f"{c}__{arm}__s{seed}"
        prompt = cs["A"] if arm == "A" else (cs["align"] + "\n\n" + cs["A"]) if arm == "AL" else cs["B"]
        rec = {"case": c, "arm": arm, "seed": seed, "prompt_chars": len(prompt)}
        log(f"== {name} ({cs['task']}, {cs['ratio']}, {cs['duration_s']} s)")
        try:
            inputs, first_img, last_img = {}, None, None
            if cs.get("first"):
                p = os.path.join(IN, cs["first"])
                inputs["first_frame"], first_img = upload(p), cv2.imread(p)
            if cs.get("last"):
                p = os.path.join(IN, cs["last"])
                inputs["last_frame"], last_img = upload(p), cv2.imread(p)
            p = {"prompt": prompt, "duration_s": cs["duration_s"], "resolution": "768p", "ratio": cs["ratio"],
                 "seed": seed, "turbo": True}
            g, meta = graphs.build("i2v", p, inputs, f"h3_prompt_ab/{name}")
            rec.update({k2: meta[k2] for k2 in ("width", "height", "length", "steps")})
            t0 = time.monotonic()
            mp4 = fetch_mp4(run_outputs(g), os.path.join(out, f"{name}.mp4"))
            rec["gen_s"] = round(time.monotonic() - t0, 1)
            frames = read_frames(mp4)
            rec.update(measure(frames, first_img, last_img))
            rec.update(audio_stats(mp4, os.path.join(work, f"{name}.wav"), os.path.join(out, f"{name}_spec.png")))
            clips[name] = frames
            log("   " + json.dumps({k2: v for k2, v in rec.items() if k2 not in ("case", "arm", "seed")}))
        except Exception as e:  # keep going: one failed clip should not cost the others
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"   {name} FAILED: {rec['error']}")
        report["cases"].setdefault(c, {})[name] = rec
        save()

        # sheets once a case is complete
        if k == len(plan) - 1 or plan[k + 1][0] != c:
            done = [(nm, fr) for nm, fr in clips.items() if nm.startswith(c + "__")]
            if done:
                width = 170 if cs["ratio"] == "9:16" else 300
                n = len(done[0][1])
                idx = [round(i * (n - 1) / 7) for i in range(8)]
                cv2.imwrite(os.path.join(out, f"{c}_sheet.jpg"),
                            np.vstack([row(fr, idx, width, nm) for nm, fr in done]), [cv2.IMWRITE_JPEG_QUALITY, 88])
                tail = list(range(n - 8, n))
                cv2.imwrite(os.path.join(out, f"{c}_end.jpg"),
                            np.vstack([row(fr, tail, width, nm) for nm, fr in done]), [cv2.IMWRITE_JPEG_QUALITY, 88])
            for nm in [nm for nm, _ in done]:
                del clips[nm]
    shutil.rmtree(work, ignore_errors=True)
    log("PROGRESS 100")


if __name__ == "__main__":
    main()
