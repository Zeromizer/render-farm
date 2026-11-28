"""Music-from-reference spike: ACE-Step 1.5 arms (farm python job; venv = requirements.txt here).

Installs ACE-Step 1.5 the authors' way (git clone at a pinned commit + `uv sync`, their own
lockfile and torch build) under ~/acestep15 on first run, downloads their default model bundle
(`acestep-download`: turbo DiT + 1.7B LM), empties ComfyUI's card, then runs ace_run.py inside
their environment (`uv run --no-sync python ...`), relaying its output.

params.args = [<job json, repo-relative>, <out dir>]  (job format: ace_run.py)
"""
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "yue2_official"))
from run_official import comfy_free, log, stream  # noqa: E402

ACE_HOME = os.path.join(os.path.expanduser("~"), "acestep15")
ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5.git"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"


def beat():
    t0 = time.monotonic()
    while True:
        time.sleep(15)
        log(f"alive t={int(time.monotonic() - t0)}s")


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    job, out = sys.argv[1], sys.argv[2]
    job = job if os.path.isabs(job) else os.path.join(os.getcwd(), job)
    out = out if os.path.isabs(out) else os.path.join(os.getcwd(), out)
    os.makedirs(out, exist_ok=True)
    threading.Thread(target=beat, daemon=True).start()
    uv = [sys.executable, "-m", "uv"]
    if not os.path.isdir(os.path.join(ACE_HOME, ".git")):
        code, _ = stream(["git", "clone", "--progress", ACE_REPO, ACE_HOME], cwd=os.path.expanduser("~"))
        if code:
            raise SystemExit("clone failed")
    code, _ = stream(["git", "-c", "advice.detachedHead=false", "checkout", ACE_COMMIT], cwd=ACE_HOME)
    if code:
        stream(["git", "fetch", "origin"], cwd=ACE_HOME)
        code, _ = stream(["git", "-c", "advice.detachedHead=false", "checkout", ACE_COMMIT], cwd=ACE_HOME)
        if code:
            raise SystemExit("checkout failed")
    log("PROGRESS 5")
    code, secs = stream(uv + ["sync"], os.path.join(out, "uv_sync.log"), cwd=ACE_HOME)
    log(f"uv sync exit={code} in {secs}s")
    if code:
        raise SystemExit("uv sync failed")
    log("PROGRESS 20")
    code, secs = stream(uv + ["run", "--no-sync", "acestep-download"], os.path.join(out, "download.log"), cwd=ACE_HOME)
    log(f"acestep-download exit={code} in {secs}s")
    if code:
        raise SystemExit("model download failed")
    log("PROGRESS 35")
    log("comfy: " + comfy_free())
    code, secs = stream(uv + ["run", "--no-sync", "python", "-u", os.path.join(HERE, "ace_run.py"), job, out],
                        os.path.join(out, "ace_run.log"), cwd=ACE_HOME)
    log(f"ace_run exit={code} in {secs}s")
    log("PROGRESS 100")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
