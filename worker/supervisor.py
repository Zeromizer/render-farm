"""User-space supervisor for the render worker (no-admin service substitute).

Launched windowless via pythonw.exe from a Startup-folder shortcut; restarts
the worker 5s after any exit. Same pattern as the OmniVoice supervisor.
Run exactly ONE instance: a single worker process serializes the single GPU.
"""
import os
import subprocess
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "supervisor.log")
RESTART_DELAY = 5
PYTHONW = os.path.join(os.path.dirname(HERE), ".venv", "Scripts", "pythonw.exe")


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] [supervisor] {msg}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    import sys
    sys.path.insert(0, HERE)
    from singleton import ensure_single_instance
    ensure_single_instance("supervisor")
    log("render supervisor started")
    while True:
        try:
            rc = subprocess.run([PYTHONW, os.path.join(HERE, "render_worker.py")], cwd=HERE).returncode
            log(f"worker exited code={rc}; restarting in {RESTART_DELAY}s")
        except Exception as e:  # noqa: BLE001
            log(f"worker failed to launch: {e}; retrying in {RESTART_DELAY}s")
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    main()
