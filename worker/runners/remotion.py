"""Remotion runner: npx remotion render in the checked-out repo.

params (jsonb): composition (required), project_dir, entry, props (object),
codec, frame_range ("0-120").
"""
import json
import os
import re
import shutil

import proc

CODEC_EXT = {"h264": "mp4", "h265": "mp4", "vp8": "webm", "vp9": "webm",
             "gif": "gif", "prores": "mov"}
_PROGRESS_RE = re.compile(r"Rendered (\d+)/(\d+)")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    """Render and return (local_output_path, ext, content_type)."""
    params = job.get("params") or {}
    composition = params.get("composition")
    if not composition:
        raise RuntimeError("remotion job needs params.composition")

    project_dir = os.path.join(repo, params["project_dir"]) if params.get("project_dir") else repo

    import git_cache
    git_cache.npm_ci_if_needed(
        project_dir, log,
        {"on_line": lambda l: None, "cancel_check": cancel_check,
         "timeout_seconds": timeout_seconds},
    )
    heartbeat.progress = 5

    codec = params.get("codec", "h264")
    ext = CODEC_EXT.get(codec, "mp4")
    out_local = os.path.join(work_dir, f"out.{ext}")

    npx = shutil.which("npx.cmd") or shutil.which("npx")
    cmd = [npx, "remotion", "render", composition, out_local, "--gl=angle", f"--codec={codec}"]
    if params.get("entry"):
        cmd.insert(3, params["entry"])  # npx remotion render <entry> <comp> <out>
    if params.get("frame_range"):
        cmd.append(f"--frames={params['frame_range']}")
    if params.get("props"):
        props_file = os.path.join(work_dir, "props.json")
        with open(props_file, "w", encoding="utf-8") as f:
            json.dump(params["props"], f)
        cmd.append(f"--props={props_file}")

    def on_line(line):
        m = _PROGRESS_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if total:
                heartbeat.progress = 5 + int(93 * done / total)
        log(f"  {line}")

    log(f"rendering: {' '.join(cmd)}")
    proc.run_streaming(cmd, cwd=project_dir, on_line=on_line,
                       cancel_check=cancel_check, timeout_seconds=timeout_seconds)
    if not os.path.exists(out_local):
        raise RuntimeError("remotion reported success but output file missing")
    content_type = {"mp4": "video/mp4", "webm": "video/webm", "gif": "image/gif",
                    "mov": "video/quicktime"}.get(ext, "application/octet-stream")
    return out_local, ext, content_type
