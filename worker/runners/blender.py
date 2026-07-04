"""Blender runner: headless render of a .blend inside the checked-out repo.

params (jsonb): blend_file (required, repo-relative), single_frame OR
frame_start+frame_end, output_format (FFMPEG default | PNG), engine_override
(e.g. CYCLES), use_optix (default true).
"""
import os
import re
import zipfile

import config
import proc

_FRAME_RE = re.compile(r"^Fra:(\d+)")

_OPTIX_EXPR = (
    "import bpy;"
    "p=bpy.context.preferences.addons['cycles'].preferences;"
    "p.compute_device_type='OPTIX';p.get_devices();"
    "[setattr(d,'use',d.type=='OPTIX') for d in p.devices];"
    "bpy.context.scene.cycles.device='GPU'"
)


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    """Render and return (local_output_path, ext, content_type)."""
    params = job.get("params") or {}
    blend_rel = params.get("blend_file")
    if not blend_rel:
        raise RuntimeError("blender job needs params.blend_file")
    blend = os.path.join(repo, blend_rel)
    if not os.path.exists(blend):
        raise RuntimeError(f"blend file not found in repo: {blend_rel}")

    single = params.get("single_frame")
    start = int(params.get("frame_start", 1))
    end = int(params.get("frame_end", start))
    if single is not None:
        start = end = int(single)
    total = max(1, end - start + 1)

    fmt = (params.get("output_format") or ("PNG" if single is not None else "FFMPEG")).upper()
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cmd = [config.BLENDER_EXE, "-b", blend]
    if params.get("engine_override"):
        cmd += ["-E", params["engine_override"]]
    if params.get("use_optix", True):
        cmd += ["--python-expr", _OPTIX_EXPR]
    cmd += ["-o", os.path.join(frames_dir, "frame_####"), "-F", fmt]
    if fmt == "FFMPEG":
        cmd += ["-x", "1"]
    if single is not None:
        cmd += ["-f", str(single)]
    else:
        cmd += ["-s", str(start), "-e", str(end), "-a"]

    def on_line(line):
        m = _FRAME_RE.match(line)
        if m:
            fra = int(m.group(1))
            heartbeat.progress = min(98, int(100 * (fra - start + 1) / total))
        if "Saved:" in line or line.startswith("Fra:") is False:
            log(f"  {line}")

    log(f"rendering: {os.path.basename(blend)} frames {start}-{end} fmt={fmt}")
    proc.run_streaming(cmd, cwd=repo, on_line=on_line,
                       cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    outputs = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if not f.endswith(".zip")
    )
    if not outputs:
        raise RuntimeError("blender finished but produced no output files")

    if len(outputs) == 1:
        out = outputs[0]
        ext = os.path.splitext(out)[1].lstrip(".").lower() or "bin"
        content_type = {"png": "image/png", "mp4": "video/mp4", "mkv": "video/x-matroska",
                        "avi": "video/x-msvideo", "exr": "image/x-exr"}.get(ext, "application/octet-stream")
        return out, ext, content_type

    # PNG sequence (or similar): zip it
    zip_path = os.path.join(work_dir, "frames.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in outputs:
            z.write(f, os.path.basename(f))
    return zip_path, "zip", "application/zip"
