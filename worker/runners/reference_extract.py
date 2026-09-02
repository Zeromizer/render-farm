"""reference_extract runner: measure a reference video (or a render output)
into a motion_spec JSON — HANDOFF-reference-extraction.md.

params (jsonb):
  extract:
    kind          "reference" | "render_output"
    reference_id  rp_references.id  (kind=reference)
    render_id     rp_renders.id     (kind=render_output)
    bucket        source bucket ("assets" for references, "renders" for outputs)
    path          object path in that bucket
    stages        list of stage names (see extract/extract.py ALL_STAGES)
    threshold     optional ContentDetector threshold override
    ocr_fps       optional OCR sample rate (drop to 1 if the 90s budget busts)

Split of responsibilities: extract/extract.py is pure compute (video in,
JSON + artefacts out) and runs in a cached venv keyed by
extract/requirements.txt (worker/venvs.py) so torch/librosa never touch the
worker's own venv. THIS module owns all Supabase I/O: download the source,
upload artefacts to renders/refs/<id>/ (fixed paths, upsert — idempotent per
id, HANDOFF §13), and write the spec onto the row with the service key. The
job's single output stays motion_spec.json so the farm row is self-contained.

No repo: render_worker skips the git clone for this engine (repo is None,
repo_url is a "-" placeholder).
"""
import json
import os

import config
import db
import proc
from venvs import venv_python

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(_WORKER_DIR, "extract")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    """Extract and return (local motion_spec.json, "json", "application/json")."""
    params = (job.get("params") or {}).get("extract") or {}
    kind = params.get("kind") or "reference"
    target_id = params.get("reference_id") if kind == "reference" else params.get("render_id")
    bucket = params.get("bucket")
    path = params.get("path")
    if not (target_id and bucket and path):
        raise RuntimeError("reference_extract needs params.extract.{reference_id|render_id, bucket, path}")

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}

    local_video = _download(bucket, path, work_dir, log)
    heartbeat.progress = 3

    py = venv_python(os.path.join(EXTRACT_DIR, "requirements.txt"), log, run_kw)
    heartbeat.progress = 5

    out_dir = os.path.join(work_dir, "out")
    cmd = [py, "-u", os.path.join(EXTRACT_DIR, "extract.py"), local_video, out_dir,
           "--stages", ",".join(params.get("stages") or []),
           "--reference-id", str(target_id)]
    if params.get("threshold"):
        cmd += ["--threshold", str(params["threshold"])]
    if params.get("ocr_fps"):
        cmd += ["--ocr-fps", str(params["ocr_fps"])]

    def on_line(line):
        if line.startswith("PROGRESS "):
            try:
                heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.9)
            except ValueError:
                pass
        log(f"  {line}")

    log(f"extracting {kind} {target_id} ({path})")
    proc.run_streaming(cmd, cwd=EXTRACT_DIR, on_line=on_line,
                       cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    spec_path = os.path.join(out_dir, "motion_spec.json")
    if not os.path.exists(spec_path):
        raise RuntimeError("extract.py finished but motion_spec.json missing")
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    prefix = f"refs/{target_id}" if kind == "reference" else f"refs/outputs/{target_id}"
    _upload_artefacts(spec, out_dir, prefix, log)
    heartbeat.progress = 97

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)

    table = "rp_references" if kind == "reference" else "rp_renders"
    db.sb.table(table).update({"motion_spec": spec}).eq("id", target_id).execute()
    log(f"wrote {table}.motion_spec for {target_id} (status={spec.get('status')})")

    return spec_path, "json", "application/json"


def _download(bucket, path, work_dir, log):
    """Stream the source object to disk — a reference can be 200MB."""
    import httpx  # supabase dependency, present in the worker venv

    res = db.sb.storage.from_(bucket).create_signed_url(path, 3600)
    url = res.get("signedURL") or res.get("signedUrl")
    if not url:
        raise RuntimeError(f"could not sign {bucket}/{path}")
    ext = os.path.splitext(path)[1] or ".mp4"
    local = os.path.join(work_dir, f"input{ext}")
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"download failed: HTTP {r.status_code} for {bucket}/{path}")
        with open(local, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    log(f"downloaded {bucket}/{path} -> {os.path.getsize(local)} bytes")
    return local


def _upload_artefacts(spec, out_dir, prefix, log):
    """Contact sheet + keyframes -> renders bucket at fixed per-id paths
    (upsert, so a re-extract overwrites instead of accumulating), then patch
    the storage paths into the spec in place of local ones."""
    names = sorted(f for f in os.listdir(out_dir)
                   if f == "contact.jpg" or (f.startswith("kf_") and f.endswith(".jpg")))
    remote_for = {}
    for name in names[:80]:  # bound a pathological shot count
        remote = f"{prefix}/{name}"
        with open(os.path.join(out_dir, name), "rb") as f:
            db.sb.storage.from_(config.BUCKET).upload(
                remote, f, {"content-type": "image/jpeg", "upsert": "true"})
        remote_for[name] = remote
    log(f"uploaded {len(remote_for)} artefact(s) to {config.BUCKET}/{prefix}/")

    if "contact.jpg" in remote_for:
        spec["contact_sheet"] = remote_for["contact.jpg"]
    comp = spec.get("composition")
    if isinstance(comp, dict):
        if "contact.jpg" in remote_for:
            comp["contact_sheet"] = remote_for["contact.jpg"]
        for kf in comp.get("keyframes") or []:
            base = os.path.basename(kf.get("path") or "")
            if base in remote_for:
                kf["path"] = remote_for[base]
