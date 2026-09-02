"""Python runner: run a repo script inside a cached venv (GPU jobs like
rembg/BiRefNet matting, Real-ESRGAN upscales, RIFE interpolation...).

params (jsonb):
  script        (required) repo-relative .py path
  args          (optional) list of string CLI args
  requirements  (optional) repo-relative requirements.txt; a venv is created
                per unique requirements content and cached across jobs
  output        (required) repo-relative file or directory the script writes;
                a directory is zipped for upload

The script runs with cwd=<repo> and gets RENDER_WORK_DIR in its env. Lines
matching "PROGRESS <0-100>" on stdout update the job's progress.
"""
import os
import re
import zipfile

import proc
from venvs import venv_python

_PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d{1,3})\b")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    """Run and return (local_output_path, ext, content_type)."""
    params = job.get("params") or {}
    script_rel = params.get("script")
    output_rel = params.get("output")
    if not script_rel:
        raise RuntimeError("python job needs params.script")
    if not output_rel:
        raise RuntimeError("python job needs params.output")
    script = os.path.join(repo, script_rel)
    if not os.path.exists(script):
        raise RuntimeError(f"script not found in repo: {script_rel}")

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}

    requirements = None
    if params.get("requirements"):
        requirements = os.path.join(repo, params["requirements"])
        if not os.path.exists(requirements):
            raise RuntimeError(f"requirements not found in repo: {params['requirements']}")
    py = venv_python(requirements, log, run_kw)
    heartbeat.progress = 5

    env = dict(os.environ)
    env["RENDER_WORK_DIR"] = work_dir
    env["PYTHONUNBUFFERED"] = "1"

    def on_line(line):
        m = _PROGRESS_RE.match(line)
        if m:
            heartbeat.progress = 5 + int(min(100, int(m.group(1))) * 0.93)
        log(f"  {line}")

    cmd = [py, "-u", script] + [str(a) for a in (params.get("args") or [])]
    log(f"running: {' '.join(cmd)}")
    proc.run_streaming(cmd, cwd=repo, on_line=on_line, env=env,
                       cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    out_path = os.path.join(repo, output_rel)
    if not os.path.exists(out_path):
        raise RuntimeError(f"script finished but output missing: {output_rel}")

    if os.path.isdir(out_path):
        files = sorted(
            os.path.join(dp, f)
            for dp, _, fs in os.walk(out_path) for f in fs
        )
        if not files:
            raise RuntimeError(f"output directory is empty: {output_rel}")
        zip_path = os.path.join(work_dir, "output.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, os.path.relpath(f, out_path))
        return zip_path, "zip", "application/zip"

    ext = os.path.splitext(out_path)[1].lstrip(".").lower() or "bin"
    content_type = {
        "png": "image/png", "jpg": "image/jpeg", "mp4": "video/mp4",
        "mov": "video/quicktime", "zip": "application/zip",
        "json": "application/json",
    }.get(ext, "application/octet-stream")
    return out_path, ext, content_type
