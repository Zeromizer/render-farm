"""Spike: chain MiniMax H3 shots on their real latents (roycho87's Clown Anchoring idea).

Today the farm joins H3 shots with a still: the next shot's first_frame is the previous shot's
last frame, so the motion restarts at the join. This spike measures three ways to continue a
10 s shot, plus a loop, all at 832x480 / 243 frames / fl2va turbo-8 on the PC's ComfyUI:

  S1     t2v shot (latent saved)
  still  i2v, first_frame = S1's last frame (what the farm does now); join drops 1 frame
  guide  t2v + core MiniMaxH3AddGuide: S1's last 39 frames + audio pinned at frame 0 as guide
         tokens (the model redraws them); join drops the 39 redrawn frames
  hold   t2v + AimH3Hold head: S1's last 39 frames' LATENTS spliced in and masked (kept as-is);
         join drops those 39 frames
  loop   t2v + AimH3Hold head (S1's last 39) + AimH3Hold tail (S1's first 39); the loop is
         S1 + loop[39:204], and loop[204:243] == S1[0:39] closes it

The hold node lives in comfy_node/ and is copied to ComfyUI/custom_nodes/aimotion_h3_hold; ComfyUI
is restarted only when that file is new or changed (the farm runs one job at a time).

params.args = [<job json string>, <out dir>]. Job JSON (all optional):
  {"prompt", "width": 832, "height": 480, "frames": 243, "overlap": 39, "steps": 8,
   "seed_s1": 6332, "seed": 911, "arms": ["hold", "guide", "still", "loop"]}
Writes <out>/{S1,still,guide,hold,loop}.mp4, join_<arm>.mp4, loop_x2.mp4, compare.mp4,
seams_<arm>.jpg, report.json.
"""
import glob
import json
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
T = httpx.Timeout(30.0, read=600.0)
FPS = 24
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

UNET = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
TURBO = {8: "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
         4: "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"}
TEXT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT = ("Cinematic tracking shot filmed from a camera car driving alongside: a lime green compact "
          "electric hatchback cruises along a coastal highway at golden hour, the camera keeps pace "
          "beside it at a steady speed, the sea glitters on the right, palm trees and lamp posts slide "
          "past in the background, smooth continuous motion, realistic, natural road and wind sound.")


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


def stop_comfy():
    port = int(COMFY.rsplit(":", 1)[-1].split("/")[0])
    ps = (f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
          "if ($c) { $w = Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\"; "
          "'{0}|{1}' -f $c.OwningProcess, $w.ParentProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         timeout=60, creationflags=NO_WINDOW).stdout.strip()
    if not out:
        return
    q = httpx.get(COMFY + "/queue", timeout=10).json()
    if q.get("queue_running") or q.get("queue_pending"):
        raise RuntimeError("ComfyUI is busy; not restarting it")
    pid, parent = out.split("|")
    log(f"stopping ComfyUI pid {pid} (launcher {parent})")
    for p in (parent, pid):
        subprocess.run(["taskkill", "/PID", p, "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    t0 = time.monotonic()
    while comfy_up() and time.monotonic() - t0 < 60:
        time.sleep(1)


def install_node():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_node", "__init__.py")
    dst_dir = os.path.join(COMFY_DIR, "custom_nodes", "aimotion_h3_hold")
    dst = os.path.join(dst_dir, "__init__.py")
    new = open(src, "rb").read()
    old = open(dst, "rb").read() if os.path.exists(dst) else None
    if old != new:
        os.makedirs(dst_dir, exist_ok=True)
        with open(dst, "wb") as f:
            f.write(new)
        log(f"node {'updated' if old else 'installed'} at {dst_dir}")
    launch_comfy()
    have = "AimH3Hold" in httpx.get(COMFY + "/object_info/AimH3Hold", timeout=60).json()
    if old != new or not have:
        stop_comfy()
        launch_comfy()
        have = "AimH3Hold" in httpx.get(COMFY + "/object_info/AimH3Hold", timeout=60).json()
    if not have:
        tail = ""
        logf = os.path.join(COMFY_DIR, "comfyui-headless.log")
        if os.path.exists(logf):
            tail = open(logf, encoding="utf-8", errors="replace").read()[-3000:]
        raise RuntimeError(f"AimH3Hold did not load after a restart. ComfyUI log tail:\n{tail}")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "h3_chain", "type": "input", "overwrite": "true"}, timeout=T)
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


def free():
    try:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


# ---------------------------------------------------------------- graphs

def build(p, seed, prefix, first_frame=None, guide_clip=None, holds=(), save_as=None):
    """fl2va turbo graph. holds = [(source, "head"|"tail")]; guide_clip = uploaded mp4 to pin at frame 0."""
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": TURBO[p["steps"]], "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
    }
    cond = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": p["prompt"],
            "width": p["width"], "height": p["height"], "length": p["frames"]}
    if first_frame:
        g["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        cond["first_frame"] = ["first", 0]
    g["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond}
    positive, latent = ["cond", 0], ["cond", 1]
    if guide_clip:
        g["gload"] = {"class_type": "LoadVideo", "inputs": {"file": guide_clip}}
        g["gparts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["gload", 0]}}
        g["guide"] = {"class_type": "MiniMaxH3AddGuide",
                      "inputs": {"positive": positive, "latent": latent, "vae": ["vae", 0], "audio_vae": ["avae", 0],
                                 "image": ["gparts", 0], "audio": ["gparts", 1], "frame_idx": 0}}
        positive = ["guide", 0]
    for i, (source, at) in enumerate(holds):
        g[f"hold{i}"] = {"class_type": "AimH3Hold",
                         "inputs": {"latent": latent, "source": source, "frames": p["overlap"], "at": at,
                                    "hold_audio": True}}
        latent = [f"hold{i}", 0]
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": positive}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": ["lora", 0], "scheduler": "simple", "steps": p["steps"], "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": latent}}
    out = ["sample", 0]
    if save_as:
        g["savelat"] = {"class_type": "AimH3SaveLatent", "inputs": {"latent": out, "name": save_as}}
        out = ["savelat", 0]
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": out, "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": out, "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": float(FPS), "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}}
    return g


# ---------------------------------------------------------------- video / metrics

def read_frames(path, size=None, gray=False):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if size:
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        if gray:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(f)
    cap.release()
    return frames


def join(a, b, skip, dest, a_end=None, b_end=None):
    """a[0:a_end] then b[skip:b_end], audio cut at the same points with a 40 ms crossfade."""
    a_end = a_end or len(read_frames(a, (16, 16)))
    vb = f"trim=start_frame={skip}" + (f":end_frame={b_end}" if b_end else "")
    ab = f"atrim=start={skip / FPS:.6f}" + (f":end={b_end / FPS:.6f}" if b_end else "")
    fc = (f"[0:v]trim=end_frame={a_end},setpts=PTS-STARTPTS[v0];[1:v]{vb},setpts=PTS-STARTPTS[v1];"
          f"[v0][v1]concat=n=2:v=1:a=0[v];"
          f"[0:a]atrim=end={a_end / FPS:.6f},asetpts=PTS-STARTPTS[a0];[1:a]{ab},asetpts=PTS-STARTPTS[a1];"
          f"[a0][a1]acrossfade=d=0.04[a]")
    ff("-i", a, "-i", b, "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "14",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dest)
    return a_end


def seam_stats(frames, s):
    """frames: gray uint8 list; the join is between frames[s-1] and frames[s]."""
    f = [x.astype(np.float32) for x in frames]
    n = len(f)
    diffs = np.array([np.abs(f[i + 1] - f[i]).mean() for i in range(n - 1)])
    lo, hi = max(0, s - 13), min(n - 1, s + 13)
    around = [diffs[i] for i in range(lo, hi) if abs(i - (s - 1)) > 1]
    jump = float(diffs[s - 1])
    ratio = jump / max(1e-6, float(np.median(around)))

    def flow(i):
        fl = cv2.calcOpticalFlowFarneback(frames[i], frames[i + 1], None, 0.5, 3, 21, 3, 5, 1.1, 0)
        return fl.reshape(-1, 2).mean(axis=0)

    before = np.mean([flow(i) for i in range(s - 4, s - 1)], axis=0)
    after = np.mean([flow(i) for i in range(s, s + 3)], axis=0)
    at = flow(s - 1)
    ref = (before + after) / 2
    luma_before = float(np.mean([f[i].mean() for i in range(s - 6, s)]))
    luma_after = float(np.mean([f[i].mean() for i in range(s, s + 6)]))
    return {"seam_frame": s, "diff_at_seam": round(jump, 2), "diff_median_around": round(float(np.median(around)), 2),
            "seam_ratio": round(ratio, 2),
            "flow_before_px": [round(float(x), 2) for x in before], "flow_at_seam_px": [round(float(x), 2) for x in at],
            "flow_after_px": [round(float(x), 2) for x in after],
            "flow_seam_error_px": round(float(np.linalg.norm(at - ref)), 2),
            "speed_change_pct": round(100 * (np.linalg.norm(after) - np.linalg.norm(before)) / max(1e-6, np.linalg.norm(before)), 1),
            "luma_step": round(luma_after - luma_before, 2)}


def block_luma(frames, start=0):
    f = frames[start:]
    return [round(float(np.mean([x.mean() for x in f[i:i + 17]])), 1) for i in range(0, len(f) - 4, 17)]


def mae(a, b):
    return round(float(np.mean([np.abs(x.astype(np.float32) - y.astype(np.float32)).mean() for x, y in zip(a, b)])), 2)


def seam_strip(path, s, dest, label):
    fr = read_frames(path, (416, 240))
    tiles = []
    for i in range(s - 3, s + 3):
        t = fr[i].copy()
        cv2.putText(t, f"{i}{'  <join' if i == s else ''}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    img = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.putText(img, label, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(dest, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def compare_video(paths, labels, dest, start, end):
    caps = [cv2.VideoCapture(p) for p in paths]
    tmp = dest + ".avi"
    w, h = 832, 480
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h * len(paths)))
    for i in range(end):
        rows = []
        for c, lab in zip(caps, labels):
            ok, f = c.read()
            if not ok:
                f = np.zeros((h, w, 3), np.uint8)
            f = cv2.resize(f, (w, h))
            cv2.putText(f, lab, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            rows.append(f)
        if i >= start:
            vw.write(np.vstack(rows))
    vw.release()
    for c in caps:
        c.release()
    ff("-i", tmp, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dest)
    os.remove(tmp)


# ---------------------------------------------------------------- main

def main():
    p = {"prompt": PROMPT, "width": 832, "height": 480, "frames": 243, "overlap": 39, "steps": 8,
         "seed_s1": 6332, "seed": 911, "arms": ["hold", "guide", "still", "loop"]}
    p.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].strip() else {})
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "out")
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):  # the python engine's out dir is a shared cache: clear it
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)
    work = os.path.join(out, "work")
    os.makedirs(work)
    run_id = uuid.uuid4().hex[:8]
    s1_name = f"s1_{run_id}"
    N, K = p["frames"], p["overlap"]
    report = {"params": {k: v for k, v in p.items() if k != "prompt"}, "prompt": p["prompt"], "gen": {}, "seams": {}}

    install_node()
    log("PROGRESS 3")

    def gen(name, graph):
        t0 = time.monotonic()
        dest = fetch_mp4(run_outputs(graph), os.path.join(out, f"{name}.mp4"))
        secs = round(time.monotonic() - t0)
        n = len(read_frames(dest, (16, 16)))
        report["gen"][name] = {"seconds": secs, "frames": n}
        log(f"{name}: {n} frames in {secs}s")
        return dest

    s1 = gen("S1", build(p, p["seed_s1"], f"h3_chain/{run_id}_S1", save_as=s1_name))
    log("PROGRESS 20")

    # S1's tail as the guide clip and its last frame as the still
    s1_tail = os.path.join(work, "s1_tail.mp4")
    ff("-i", s1, "-vf", f"trim=start_frame={N - K},setpts=PTS-STARTPTS", "-af",
       f"atrim=start={(N - K) / FPS:.6f},asetpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "10",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", s1_tail)
    s1_last = os.path.join(work, "s1_last.png")
    ff("-i", s1, "-vf", f"select=eq(n\\,{N - 1})", "-frames:v", "1", s1_last)

    joins = {}
    done = 20
    step = 75 // max(1, len(p["arms"]))
    for arm in p["arms"]:
        try:
            if arm == "hold":
                path = gen("hold", build(p, p["seed"], f"h3_chain/{run_id}_hold", holds=[(s1_name, "head")]))
                joins["hold"] = (path, K)
            elif arm == "guide":
                path = gen("guide", build(p, p["seed"], f"h3_chain/{run_id}_guide", guide_clip=upload(s1_tail)))
                joins["guide"] = (path, K)
            elif arm == "still":
                path = gen("still", build(p, p["seed"], f"h3_chain/{run_id}_still", first_frame=upload(s1_last)))
                joins["still"] = (path, 1)
            elif arm == "loop":
                path = gen("loop", build(p, p["seed"], f"h3_chain/{run_id}_loop",
                                         holds=[(s1_name, "head"), (s1_name, "tail")]))
                joins["loop"] = (path, K)
        except RuntimeError as e:
            log(f"{arm}: FAILED {e}")
            report["gen"][arm] = {"error": str(e)[:2000]}
        done += step
        log(f"PROGRESS {done}")
    free()

    # ---- joins + seam metrics
    g1 = read_frames(s1, (416, 240), gray=True)
    report["S1_block_luma"] = block_luma(g1)
    for arm, (path, skip) in joins.items():
        g2 = read_frames(path, (416, 240), gray=True)
        rec = {"block_luma": block_luma(g2)}
        if arm in ("hold", "guide", "loop"):
            # how close the shot's first K frames are to the S1 frames they stand for
            rec["head_vs_S1_tail_mae"] = mae(g2[:K], g1[N - K:])
            rec["head_luma_minus_S1_tail"] = round(float(np.mean([x.mean() for x in g2[:K]]) - np.mean([x.mean() for x in g1[N - K:]])), 2)
        if arm == "loop":
            rec["tail_vs_S1_head_mae"] = mae(g2[N - K:], g1[:K])
            dest = os.path.join(out, "join_loop.mp4")
            join(s1, path, K, dest, a_end=N, b_end=N - K)
            loop_len = N + (N - 2 * K)
            ff("-stream_loop", "1", "-i", dest, "-c", "copy", os.path.join(out, "loop_x2.mp4"))
            gl = read_frames(os.path.join(out, "loop_x2.mp4"), (416, 240), gray=True)
            rec["seam_into_loop_shot"] = seam_stats(gl, N)
            rec["seam_wrap_to_S1"] = seam_stats(gl, loop_len)
            rec["loop_frames"] = loop_len
            seam_strip(os.path.join(out, "loop_x2.mp4"), loop_len, os.path.join(out, "seams_loop_wrap.jpg"), "loop: wrap back to S1 frame 0")
            seam_strip(os.path.join(out, "loop_x2.mp4"), N, os.path.join(out, "seams_loop_in.jpg"), "loop: S1 -> in-between shot")
        else:
            dest = os.path.join(out, f"join_{arm}.mp4")
            join(s1, path, skip, dest, a_end=N)
            gj = read_frames(dest, (416, 240), gray=True)
            rec["seam"] = seam_stats(gj, N)
            seam_strip(dest, N, os.path.join(out, f"seams_{arm}.jpg"), f"{arm}: S1 -> next shot")
        report["seams"][arm] = rec
        log(f"{arm}: {json.dumps(rec)[:600]}")

    arms = [a for a in ("still", "guide", "hold") if os.path.exists(os.path.join(out, f"join_{a}.mp4"))]
    if arms:
        compare_video([os.path.join(out, f"join_{a}.mp4") for a in arms],
                      [f"{a}  (join at 10.1 s)" for a in arms], os.path.join(out, "compare.mp4"), N - 72, N + 96)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    shutil.rmtree(work, ignore_errors=True)
    log("PROGRESS 100")


if __name__ == "__main__":
    main()
