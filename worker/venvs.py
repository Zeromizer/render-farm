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
