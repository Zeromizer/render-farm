"""Remotion runner: npx remotion render (or still) in the checked-out repo.

params (jsonb): composition (required), project_dir, entry, props (object),
codec, frame_range ("0-120"), quality ("draft"|"final", default final),
assets (manifest — handled by render_worker before this runner).

quality="draft" renders at --scale=0.5 with --crf=32 (CRF-capable codecs) and
injects {"quality": "draft"} into props so the composition can halve fps via
calculateMetadata. Note: when the composition halves fps, frame_range refers
to the draft timeline (half the frame numbers).

STILLS
------
params.output_kind == "still" switches to `npx remotion still`, which produces
one image instead of a video. Extra params: image_format ("png"|"jpeg"|"webp",
default png) and frame (int, default 0 — which frame of the composition to
capture).

Named output_kind, not output: params is one shared jsonb across all three
engines and the python runner already owns params.output (the path its script
writes). Nothing would break today, since each runner only reads its own keys,
but two unrelated meanings under one name is a trap for whoever adds the next
engine.

`still` is a different command with a different flag set, not `render` with an
image codec: --codec, --crf and --frames do not exist on it and each is a hard
CLI error. That is why this branches on the command rather than adding "png" to
CODEC_EXT. What it does share with render is --scale, --props and --gl, so
draft mode and the props file work identically.

One image per job is deliberate, and is what lets a carousel be N jobs: every
runner here returns exactly one (path, ext, content_type) and db.upload_output
writes exactly one object. A multi-image render would have to zip, the way
blender.py does — a caller that wants the pieces separately should send
separate jobs.
"""
import json
import os
import re
import shutil

import proc

CODEC_EXT = {"h264": "mp4", "h265": "mp4", "vp8": "webm", "vp9": "webm",
             "gif": "gif", "prores": "mov"}
CRF_CODECS = {"h264", "h265", "vp8", "vp9"}  # --crf is invalid for prores/gif
_PROGRESS_RE = re.compile(r"Rendered (\d+)/(\d+)")

STILL_FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
VIDEO_CONTENT_TYPE = {"mp4": "video/mp4", "webm": "video/webm", "gif": "image/gif",
                      "mov": "video/quicktime"}


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

    quality = params.get("quality") or "final"
    still = params.get("output_kind") == "still"
    npx = shutil.which("npx.cmd") or shutil.which("npx")

    if still:
        ext = params.get("image_format") or "png"
        if ext not in STILL_FORMATS:
            raise RuntimeError(
                f"unsupported image_format {ext!r}; expected one of "
                f"{', '.join(sorted(STILL_FORMATS))}"
            )
        content_type = STILL_FORMATS[ext]
        out_local = os.path.join(work_dir, f"out.{ext}")
        cmd = [npx, "remotion", "still", composition, out_local,
               "--gl=angle", f"--image-format={ext}"]
        # --frame is singular and zero-based; --frames (plural) is a render-only
        # flag and errors out here.
        frame = params.get("frame")
        if frame is not None:
            cmd.append(f"--frame={int(frame)}")
        # No --crf: still has no such flag. --scale still means what it means,
        # so a draft is a half-size image rather than a cheaper-compressed one.
        if quality == "draft":
            cmd.append("--scale=0.5")
    else:
        codec = params.get("codec", "h264")
        ext = CODEC_EXT.get(codec, "mp4")
        content_type = VIDEO_CONTENT_TYPE.get(ext, "application/octet-stream")
        out_local = os.path.join(work_dir, f"out.{ext}")
        cmd = [npx, "remotion", "render", composition, out_local,
               "--gl=angle", f"--codec={codec}"]
        if params.get("frame_range"):
            cmd.append(f"--frames={params['frame_range']}")
        if quality == "draft":
            cmd.append("--scale=0.5")
            if codec in CRF_CODECS:
                cmd.append("--crf=32")

    if params.get("entry"):
        # npx remotion <render|still> <entry> <comp> <out>
        cmd.insert(3, params["entry"])

    props = dict(params.get("props") or {})
    if "quality" in params:
        props["quality"] = quality  # preset wins for this key: CLI flags and composition fps must agree
    if props:
        props_file = os.path.join(work_dir, "props.json")
        with open(props_file, "w", encoding="utf-8") as f:
            json.dump(props, f)
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
    return out_local, ext, content_type
