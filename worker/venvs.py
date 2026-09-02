"""Cached venvs keyed by requirements-file content hash.

Shared by the python runner (repo-supplied requirements) and the
reference_extract runner (the worker's own extract/requirements.txt). A venv is
built once per unique requirements content and reused across jobs; bumping the
requirements file simply produces a new venv alongside the old one.
"""
import hashlib
import os
import subprocess
import sys

import config
import proc


def venv_python(requirements_path, log, run_kw):
    """Return the python.exe of a venv for these requirements, building it if needed."""
    if not requirements_path:
        return sys.executable
    with open(requirements_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    venv_dir = os.path.join(config.CACHE_DIR, "venvs", digest)
    py = os.path.join(venv_dir, "Scripts", "python.exe")
    marker = os.path.join(venv_dir, ".ready")
    if os.path.exists(py) and os.path.exists(marker):
        log(f"venv cache hit: {digest}")
        # Touch the marker so eviction is by last USE, not build date — a
        # venv in daily service must never age out under a job.
        try:
            os.utime(marker, None)
        except OSError:
            pass
        return py
    log(f"creating venv {digest} (first run for these requirements)")
    os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True,
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    proc.run_streaming(
        [py, "-m", "pip", "install", "--no-input", "-r", requirements_path],
        cwd=venv_dir, **run_kw,
    )
    with open(marker, "w") as f:
        f.write(digest)
    return py


def cleanup_old(log):
    """Drop venvs unused for VENV_MAX_AGE_DAYS; they rebuild on demand.

    Content-hash keying means every requirements bump strands the previous
    build forever — measured on the desktop at 14GB across seven venvs, 79%
    of it two torch builds, one of which no requirements hash will ever ask
    for again. Age is the .ready marker's mtime, which venv_python touches on
    every cache hit, so this is time-since-last-use. A half-built venv (no
    marker) is judged by the directory's own mtime and swept the same way.
    """
    root = os.path.join(config.CACHE_DIR, "venvs")
    if not os.path.isdir(root):
        return
    max_age = float(os.environ.get("VENV_MAX_AGE_DAYS", "30")) * 86400
    import shutil as _shutil
    import time as _time

    now = _time.time()
    for name in os.listdir(root):
        venv_dir = os.path.join(root, name)
        if not os.path.isdir(venv_dir):
            continue
        marker = os.path.join(venv_dir, ".ready")
        ref = marker if os.path.exists(marker) else venv_dir
        try:
            age = now - os.path.getmtime(ref)
        except OSError:
            continue
        if age > max_age:
            log(f"venv cleanup: dropping {name} (unused {age / 86400:.0f}d)")
            _shutil.rmtree(venv_dir, ignore_errors=True)
