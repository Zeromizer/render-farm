"""Optional still-only worker; separate caches, CPU capture and resource admission.

Launch with .venv/Scripts/pythonw.exe worker/preview_worker.py after applying
docs/preview-worker.sql. Never launch another unrestricted render worker.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
base = Path(os.environ.get("RENDER_PREVIEW_CACHE_DIR") or Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "render-farm-preview")
os.environ["RENDER_CACHE_DIR"] = str(base)
os.environ["RENDER_WORKER_LANE"] = "preview"
os.environ["PRODUCER_BROWSER_GPU_MODE"] = "software"
os.environ["OMP_NUM_THREADS"] = "2"

if __name__ == "__main__":
    from render_worker import main
    main()
