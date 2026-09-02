"""Shared helpers for the asset-gate compute scripts (asset_check,
frame_extract, video_split). Pure: no Supabase, no worker imports — these run
inside the extract venv (worker/venvs.py), same as extract.py."""
import glob
import hashlib
import os
import shutil
import subprocess


def find_ffmpeg_tool(name):
    """Resolve ffmpeg/ffprobe without relying on the worker's PATH — the same
    three-step search extract.py uses (FFMPEG_DIR, PATH, the winget layout)."""
    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir:
        exe = os.path.join(env_dir, f"{name}.exe")
        if os.path.exists(exe):
            return exe
    found = shutil.which(name)
    if found:
        return found
    pattern = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", f"{name}.exe",
    )
    hits = sorted(glob.glob(pattern), reverse=True)
    if hits:
        return hits[0]
    raise RuntimeError(
        f"{name} not found: set FFMPEG_DIR in render-farm\\.env to its bin directory, "
        "or install with `winget install Gyan.FFmpeg`"
    )


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def probe_duration(ffprobe, video):
    out = run([ffprobe, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video])
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
