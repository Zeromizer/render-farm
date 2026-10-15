"""Env + path resolution for the render worker. Loads ../.env with an absolute
path so launching from the supervisor / Startup shortcut is CWD-independent."""
import os

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
load_dotenv(os.path.join(ROOT, ".env"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = "renders"
ASSETS_BUCKET = "assets"

WINDOWS = os.name == "nt"

BLENDER_EXE = os.environ.get(
    "BLENDER_EXE",
    r"C:\Users\shawn_fku5qux\AppData\Local\Programs\Blender\blender-4.5.9-windows-x64\blender.exe"
    if WINDOWS else "blender",
)
# Remotion's OpenGL backend: ANGLE over Direct3D on the PC; on Linux, ANGLE over EGL.
REMOTION_GL = os.environ.get("REMOTION_GL", "angle" if WINDOWS else "angle-egl")

CACHE_DIR = os.environ.get("RENDER_CACHE_DIR") or os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "render-farm"
)
REPOS_DIR = os.path.join(CACHE_DIR, "repos")
# Built venvs (venvs.py). Separate so the cloud pods can share them on the
# network volume while each keeps its job files on its own fast local disk.
VENVS_DIR = os.environ.get("VENVS_DIR") or os.path.join(CACHE_DIR, "venvs")
WORK_DIR = os.path.join(CACHE_DIR, "work")   # per-job temp render output dirs
ASSETS_DIR = os.path.join(CACHE_DIR, "assets")  # content-addressed asset cache (files named by sha256)

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "3"))
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "15"))
STALE_MINUTES = int(os.environ.get("STALE_MINUTES", "5"))
RECLAIM_EVERY_POLLS = 20  # ~60s at POLL_SECONDS=3

REPO_CACHE_MAX_AGE_DAYS = 14
WORK_MAX_AGE_DAYS = 2
ASSET_CACHE_MAX_AGE_DAYS = float(os.environ.get("ASSET_CACHE_MAX_AGE_DAYS", "7"))
# LRU cap on top of the age limit: least recently used (by mtime) go first.
ASSET_CACHE_MAX_GB = float(os.environ.get("ASSET_CACHE_MAX_GB", "40"))
# The cache sweeps ran only at worker start, and the worker runs for weeks.
CLEANUP_EVERY_SECONDS = int(os.environ.get("CLEANUP_EVERY_SECONDS", str(6 * 3600)))
SIGNED_URL_SECONDS = 7 * 24 * 3600

# --- video_gen (MiniMax H3 via headless ComfyUI) ---
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI" if WINDOWS else "/workspace/ComfyUI")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
# Pause the OmniVoice/Chatterbox TTS workers (6-8 GB VRAM) for the duration
# of a video job; "0" to leave them alone.
VIDEO_GEN_PAUSE_TTS = os.environ.get("VIDEO_GEN_PAUSE_TTS", "1") not in ("0", "false", "no")
TTS_STUDIO_DIR = os.environ.get("TTS_STUDIO_DIR", r"C:\Coding\Voice Output")
# Before an H3 job touches ComfyUI, wait until this much physical RAM is available
# (free + standby) so the --fast-disk weights don't thrash the pagefile
# (videogen/ram_gate.py). "0" disables. Gives up and starts anyway after the max wait.
VIDEO_GEN_MIN_AVAIL_RAM_GB = float(os.environ.get("VIDEO_GEN_MIN_AVAIL_RAM_GB", "12"))
VIDEO_GEN_RAM_WAIT_MAX_MINUTES = int(os.environ.get("VIDEO_GEN_RAM_WAIT_MAX_MINUTES", "20"))
# After a job that used ComfyUI, restart it (idle only) when its python holds more
# private memory than this: /free does not give back the host RAM an H3 run leaves
# behind (videogen/comfy_client.recycle_if_bloated). "0" disables.
COMFYUI_RECYCLE_ABOVE_GB = float(os.environ.get("COMFYUI_RECYCLE_ABOVE_GB", "8"))
# Unload ComfyUI's models after each job (the 16 GB PC must); "0" keeps them
# resident (the 96 GB cloud pod; videogen/comfy_client.free).
COMFYUI_FREE_AFTER_JOB = os.environ.get("COMFYUI_FREE_AFTER_JOB", "1") not in ("0", "false", "no")
VIDEO_GEN_DEFAULT_TIMEOUT_MINUTES = int(os.environ.get("VIDEO_GEN_DEFAULT_TIMEOUT_MINUTES", "60"))

# --- light lane (worker/light_lane.py): CPU engines beside the render loop ---
LIGHT_LANE = os.environ.get("LIGHT_LANE", "0") in ("1", "true", "yes")
# A light job starts only while all three hold, so it cannot push a running H3
# job into the commit-limit wedge (WinError 1455, 2026-09-14).
LIGHT_LANE_MIN_AVAIL_RAM_GB = float(os.environ.get("LIGHT_LANE_MIN_AVAIL_RAM_GB", "16"))
LIGHT_LANE_MIN_COMMIT_GB = float(os.environ.get("LIGHT_LANE_MIN_COMMIT_GB", "8"))
LIGHT_LANE_MIN_FREE_VRAM_MIB = int(os.environ.get("LIGHT_LANE_MIN_FREE_VRAM_MIB", "2048"))

# --- cloud GPU lane (RunPod; docs/cloud-gpu-brief.md) ---
# On the cloud pod: the engines this worker claims (comma list). Set = cloud mode:
# claim only these, no fetch lane, no patch library, exit (and terminate the pod)
# after IDLE_EXIT_MINUTES with nothing to do.
WORKER_ENGINES = {e.strip() for e in os.environ.get("WORKER_ENGINES", "").split(",") if e.strip()}
IDLE_EXIT_MINUTES = float(os.environ.get("IDLE_EXIT_MINUTES", "15"))
# On the PC: start a pod when engines gated to the cloud have jobs waiting (worker/cloud_lane.py).
CLOUD_LANE = os.environ.get("CLOUD_LANE", "0") in ("1", "true", "yes")
CLOUD_CAPABILITY = "gpu-cloud"
# On the PC: take a cloud-gated job anyway once it has waited this long (no pod
# came up, RunPod out of stock). "0" = never.
CLOUD_FALLBACK_MINUTES = float(os.environ.get("CLOUD_FALLBACK_MINUTES", "15"))
# ...and only while this much RAM is available: once the cloud lane is live the
# agent container may hold 8g and light jobs start at 8 GB free, so an H3
# fallback must not walk into the commit wedge (render-pc review, 2026-10-15).
CLOUD_FALLBACK_MIN_AVAIL_RAM_GB = float(os.environ.get("CLOUD_FALLBACK_MIN_AVAIL_RAM_GB", "24"))
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")
RUNPOD_DATACENTER = os.environ.get("RUNPOD_DATACENTER", "")
RUNPOD_GPU_TYPES = [g.strip() for g in os.environ.get(
    "RUNPOD_GPU_TYPES",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition,NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
).split(",") if g.strip()]
RUNPOD_IMAGE = os.environ.get("RUNPOD_IMAGE", "runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404")
# Branch the pod checks out at boot (cloud/boot.sh).
CLOUD_GIT_REF = os.environ.get("CLOUD_GIT_REF", "main")
# Pods at once: each is one GPU running one ComfyUI job at a time; more pods =
# more jobs in parallel (all share the network volume). The budget counts pod-hours.
CLOUD_MAX_PODS = int(os.environ.get("CLOUD_MAX_PODS", "2"))
CLOUD_MAX_HOURS_PER_DAY = float(os.environ.get("CLOUD_MAX_HOURS_PER_DAY", "8"))

for _d in (REPOS_DIR, WORK_DIR, ASSETS_DIR):
    os.makedirs(_d, exist_ok=True)

# Studio: local web UI for video_gen (worker/studio/server.py, start-studio.bat).
# Library, uploads and turntable work dirs live under STUDIO_DIR; binds 127.0.0.1 only.
STUDIO_DIR = os.environ.get("STUDIO_DIR") or os.path.join(os.path.expanduser("~"), "Videos", "H3-Studio")
STUDIO_PORT = int(os.environ.get("STUDIO_PORT", "8790"))
# Portable rife-ncnn-vulkan (frame interpolation) for the 60 fps option and the turntable flow.
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
