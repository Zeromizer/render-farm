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

for _d in (REPOS_DIR, WORK_DIR, ASSETS_DIR):
    os.makedirs(_d, exist_ok=True)
