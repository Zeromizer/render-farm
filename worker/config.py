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

BLENDER_EXE = os.environ.get(
    "BLENDER_EXE",
    r"C:\Users\shawn_fku5qux\AppData\Local\Programs\Blender\blender-4.5.9-windows-x64\blender.exe",
)

CACHE_DIR = os.environ.get("RENDER_CACHE_DIR") or os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "render-farm"
)
REPOS_DIR = os.path.join(CACHE_DIR, "repos")
WORK_DIR = os.path.join(CACHE_DIR, "work")   # per-job temp render output dirs
ASSETS_DIR = os.path.join(CACHE_DIR, "assets")  # content-addressed asset cache (files named by sha256)

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "3"))
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "15"))
STALE_MINUTES = int(os.environ.get("STALE_MINUTES", "5"))
RECLAIM_EVERY_POLLS = 20  # ~60s at POLL_SECONDS=3

REPO_CACHE_MAX_AGE_DAYS = 14
WORK_MAX_AGE_DAYS = 2
ASSET_CACHE_MAX_AGE_DAYS = int(os.environ.get("ASSET_CACHE_MAX_AGE_DAYS", "30"))
SIGNED_URL_SECONDS = 7 * 24 * 3600

# --- video_gen (MiniMax H3 via headless ComfyUI) ---
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
# Pause the OmniVoice/Chatterbox TTS workers (6-8 GB VRAM) for the duration
# of a video job; "0" to leave them alone.
VIDEO_GEN_PAUSE_TTS = os.environ.get("VIDEO_GEN_PAUSE_TTS", "1") not in ("0", "false", "no")
TTS_STUDIO_DIR = os.environ.get("TTS_STUDIO_DIR", r"C:\Coding\Voice Output")
VIDEO_GEN_DEFAULT_TIMEOUT_MINUTES = int(os.environ.get("VIDEO_GEN_DEFAULT_TIMEOUT_MINUTES", "60"))

for _d in (REPOS_DIR, WORK_DIR, ASSETS_DIR):
    os.makedirs(_d, exist_ok=True)

# Studio: local web UI for video_gen (worker/studio/server.py, start-studio.bat).
# Library, uploads and turntable work dirs live under STUDIO_DIR; binds 127.0.0.1 only.
STUDIO_DIR = os.environ.get("STUDIO_DIR") or os.path.join(os.path.expanduser("~"), "Videos", "H3-Studio")
STUDIO_PORT = int(os.environ.get("STUDIO_PORT", "8790"))
# Portable rife-ncnn-vulkan (frame interpolation) for the 60 fps option and the turntable flow.
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
