"""matte runner: alpha-matte a clip on the GPU.

params (jsonb):
  matte:
    org_id / job_id       uuids, for filing and logging only
    source: {bucket, path}  the clip (platform resolves this from an asset_id,
                            and checks org ownership, so path is trusted here)
    model                 one of matte/matte.py MODELS
    output                webm_alpha | mask_mp4 | png_sequence
    start_s / end_s       optional window; absent means the whole clip
    fps                   optional; absent means every frame at source rate
    scale                 optional long-edge px; absent means source resolution
    feather_px / close_px / temporal_median   edge treatment

WHY THIS ENGINE EXISTS: the builder agent was running rembg inside its own
container, where onnxruntime is the CPU build and the container has no GPU
device request at all - 9.5 s/frame for birefnet-general-lite at 720x1280, so
~45 min for a 282-frame clip while the 4080 on the same box sat idle. This
moves the pixel work to the machine that has the GPU and leaves the agent a
thin orchestrator, the same shape as the asset-gate engines.

NOTE ON params: wait_for_matte scopes on params.matte.job_id, because
farm_render_jobs has no tenant column and without that check any uuid would
buy a signed url to another org's output. Nothing here rewrites params.
"""
import os

import db
import proc
from venvs import venv_python
from runners import gate_common

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATTE_DIR = os.path.join(_WORKER_DIR, "matte")

EXT = {"webm_alpha": ("webm", "video/webm"),
       "mask_mp4": ("mp4", "video/mp4"),
       "png_sequence": ("zip", "application/zip")}


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = (job.get("params") or {}).get("matte") or {}
    src = p.get("source") or {}
    if not (p.get("org_id") and p.get("job_id") and src.get("bucket") and src.get("path")):
        raise RuntimeError("matte needs params.matte.{org_id, job_id, source.bucket, source.path}")

    kind = p.get("output") or "webm_alpha"
    if kind not in EXT:
        # Fail rather than substitute: a caller that asked for alpha and
        # silently got a mask would composite a grey rectangle over its scene.
        raise RuntimeError(f"unsupported output {kind!r}; supported are {sorted(EXT)}")
    ext, content_type = EXT[kind]

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}

    db.set_phase(jid, "downloading", 1)
    video = gate_common.download(src["bucket"], src["path"], work_dir, "source", log)
    heartbeat.progress = 3

    db.set_phase(jid, "building venv", 3)
    py = venv_python(os.path.join(MATTE_DIR, "requirements.txt"), log, run_kw)

    out_local = os.path.join(work_dir, f"matte.{ext}")
    cmd = [py, "-u", os.path.join(MATTE_DIR, "matte.py"), video, out_local,
           "--model", str(p.get("model") or "birefnet-general-lite"),
           "--output", kind,
           "--feather-px", str(int(p.get("feather_px", 1))),
           "--close-px", str(int(p.get("close_px", 3))),
           "--temporal-median", str(int(p.get("temporal_median", 0)))]
    # Optional keys are OMITTED by the platform rather than sent as null, so
    # absence is meaningful: no fps means every frame, no scale means source
    # resolution. Only forward what was actually asked for.
    for key, flag in (("start_s", "--start-s"), ("end_s", "--end-s"),
                      ("fps", "--fps"), ("scale", "--scale")):
        if p.get(key) is not None:
            cmd += [flag, str(p[key])]

    measured = []

    def on_line(line):
        # PHASE/PROGRESS go to the row so the waiting agent sees real movement
        # rather than a spinner: "segmenting 140/282" is the whole point.
        if line.startswith("PHASE "):
            db.set_phase(jid, line[6:].strip()[:60])
            return
        if line.startswith("PROGRESS "):
            try:
                heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.9)
            except ValueError:
                pass
            return
        if "MEASURED" in line:
            measured.append(line)
        log(f"  {line}")

    proc.run_streaming(cmd, cwd=MATTE_DIR, on_line=on_line,
                       cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    if not os.path.exists(out_local):
        raise RuntimeError("matte.py finished but produced no output file")
    for line in measured:
        log(line.strip())
    log(f"matte done: {kind} -> {os.path.getsize(out_local)} bytes")

    # Returning the local path is all it takes: run_job uploads to
    # outputs/<farm_job_id>.<ext> in the renders bucket and writes output_path,
    # output_ext and signed_url onto the row itself. Deliberately NOT a custom
    # mattes/<job_id>/... path - special-casing the shared upload is the only
    # way to break the "exactly like a render" property the platform needs.
    return out_local, ext, content_type
