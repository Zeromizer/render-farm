"""lipsync_check runner: does the mouth on screen follow the voice, and how far off is it?

SyncNet (joonson/syncnet_python 907c0b5, MIT) on each talking face: lipsync/lipsync_check.py does the
compute in its own cached venv (lipsync/requirements.txt); this side downloads the video and hands
back the result JSON as the job's output (renders bucket, output_ext json).

params (jsonb):
  check:
    bucket       "assets" (a clip, path sha256/<hex>) or "renders" (a render, path outputs/<id>.mp4)
    path         object path in that bucket
    windows      optional [{start_s, end_s, label}] - the spans where someone talks to camera.
                 Without them the whole video is one window and every face track in it is
                 reported (each scene/shot gives its own track).
    fps          optional output fps for offset_frames (default: the video's own)

Output (result JSON):
  {"ok": true, "fps": 30.0, "device": "cuda",
   "model": "SyncNet syncnet_v2 + S3FD (joonson/syncnet_python 907c0b5)",
   "thresholds": {"conf_min": 2.5, "conf_strong": 4.0, "early_ok_frames_25": 2, "late_ok_frames_25": 3,
                  "speech_min": 0.25},
   "windows": [{"label": "shot2", "start_s": 4.0, "end_s": 8.0, "face": true,
                "offset_frames_25": 0,       # + = the voice comes LATE, - = early (25 fps frames)
                "offset_frames": 0.0,        # same, in the video's fps
                "offset_ms": 0,
                "confidence": 6.1,           # SyncNet median-minus-min distance (LSE-C)
                "min_dist": 7.4,             # LSE-D
                "speech_fraction": 0.82,
                "verdict": "in_sync" | "off_by_N_frames" | "no_sync_signal" | "no_face",
                "reason": "plain sentence",
                "tracks": [{"start_s", "end_s", "face_px", ...same fields per face track}]}],
   "counts": {"in_sync": 1, "off": 0, "no_sync_signal": 3, "no_face": 1},
   "seconds": 41.2}

The window's verdict is its best-confidence face track (the person speaking).

Calibration 2026-11-19 (laptop CPU, auto mode, every face track; offsets + = voice late):
  their example.avi                 offset -3 (their demo of a lag)   conf 8.34  min dist 6.56
    voice delayed 160 ms            offset  0                         conf 7.70  -> sign confirmed
    voice advanced 160 ms           offset -8                         conf 7.39
  UNSYNCED MaczTech 431eebf5 v3 (Seedance mouths made with no lines, ElevenLabs VO on top),
    10 tracks 0.7-36.6 s            offsets -13..+15 (random)         conf 0.55-2.16  min dist 10.96-14.95
  IN SYNC real presenter footage (Downloads "video (42) lipsynced", "0715 lipsynced v3"),
    5 tracks, faces 157-207 px      offsets -2..+2                    conf 2.93-8.18  min dist 7.08-11.80
    video (42) voice +160 ms        both tracks moved exactly +4 frames (2 -> 6, -2 -> 2)
  -> CONF_MIN 2.5 splits the populations (all 10 unsynced = no_sync_signal, all 5 real = in_sync);
     tolerance early <= 2 / late <= 3 frames at 25 fps (inside ITU-R BT.1359 acceptability).
  Runtime on the laptop CPU (torch 2.5.1 CPU build): 277 s for the 36.5 s 1080x1920 v3 alone,
  ~510 s per 30 s clip with three running at once; S3FD detection is ~60 % of it. On the PC the
  venv is CUDA, untested there yet.
"""
import json
import os

import config
import proc
from venvs import venv_python
from runners import gate_common

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIPSYNC_DIR = os.path.join(_WORKER_DIR, "lipsync")


def models_dir():
    """SyncNet + S3FD weights, downloaded once and sha256-checked by lipsync_check.py."""
    return os.path.join(config.CACHE_DIR, "models", "syncnet")


def parse_params(params):
    """Validate params.check; returns (bucket, path, windows, fps)."""
    p = (params or {}).get("check") or {}
    bucket, path = p.get("bucket"), p.get("path")
    if not (bucket and path):
        raise RuntimeError("lipsync_check needs params.check.{bucket, path}")
    if bucket not in (getattr(config, "ASSETS_BUCKET", "assets"), "renders"):
        raise RuntimeError(f"lipsync_check: bucket must be assets or renders, got {bucket}")
    windows = p.get("windows") or []
    clean = []
    for i, w in enumerate(windows):
        try:
            s, e = float(w["start_s"]), float(w["end_s"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(f"lipsync_check: window {i + 1} needs numeric start_s and end_s")
        if e - s < 1.0:
            raise RuntimeError(f"lipsync_check: window {i + 1} is {e - s:.2f} s; SyncNet needs at least 1 s of face")
        label = str(w.get("label") or f"w{i + 1}")
        clean.append({"label": "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40] or f"w{i + 1}",
                      "start_s": s, "end_s": e})
    fps = p.get("fps")
    return bucket, path, clean, (float(fps) if fps else None)


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    bucket, path, windows, fps = parse_params(job.get("params"))
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}
    video = gate_common.download(bucket, path, work_dir, "video", log)
    if not os.path.splitext(video)[1]:
        os.replace(video, video + ".mp4")   # sha256/<hex> objects have no extension
        video += ".mp4"
    heartbeat.progress = 3
    py = venv_python(os.path.join(LIPSYNC_DIR, "requirements.txt"), log, run_kw)
    out = os.path.join(work_dir, "lipsync.json")
    cmd = [py, "-u", os.path.join(LIPSYNC_DIR, "lipsync_check.py"), video, out,
           "--models-dir", models_dir(), "--work-dir", os.path.join(work_dir, "sync")]
    if windows:
        wpath = os.path.join(work_dir, "windows.json")
        with open(wpath, "w", encoding="utf-8") as f:
            json.dump(windows, f)
        cmd += ["--windows", wpath]
    if fps:
        cmd += ["--fps-out", str(fps)]

    def on_line(line):
        if line.startswith("PROGRESS "):
            try:
                heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.94)
            except ValueError:
                pass
            return
        log(f"  {line}")

    proc.run_streaming(cmd, cwd=LIPSYNC_DIR, on_line=on_line,
                       cancel_check=cancel_check, timeout_seconds=timeout_seconds)
    with open(out, encoding="utf-8") as f:
        result = json.load(f)
    log(f"lipsync_check: {result.get('counts')} in {result.get('seconds')} s on {result.get('device')}")
    return out, "json", "application/json"
