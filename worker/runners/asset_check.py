"""asset_check runner: the §6 automated checks on one candidate element.

params (jsonb):
  check:
    element_id    rp_elements.id
    org_id
    scene_id
    slot
    bucket / path the candidate image
    palette_path  the scene's palette frame image (same bucket), or null
    threshold     org palette threshold (informational; the UI colours the badge)

Writes palette_distance / lighting_flags / text_detected + checks_status='done'
on the element. A failure marks checks_status='failed' — readiness treats
that as "ran", so a worker outage never makes a scene unfinishable.
"""
import json
import os

import config
import db
import proc
from venvs import venv_python
from runners import gate_common

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(_WORKER_DIR, "extract")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    p = (job.get("params") or {}).get("check") or {}
    element_id = p.get("element_id")
    if not (element_id and p.get("bucket") and p.get("path") and p.get("slot")):
        raise RuntimeError("asset_check needs params.check.{element_id, bucket, path, slot}")

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}
    try:
        image = gate_common.download(p["bucket"], p["path"], work_dir, "candidate", log)
        heartbeat.progress = 5
        palette = None
        if p.get("palette_path"):
            palette = gate_common.download(p.get("palette_bucket") or config.ASSETS_BUCKET,
                                           p["palette_path"], work_dir, "palette", log)
        heartbeat.progress = 8

        py = venv_python(os.path.join(EXTRACT_DIR, "requirements.txt"), log, run_kw)
        out = os.path.join(work_dir, "checks.json")
        cmd = [py, "-u", os.path.join(EXTRACT_DIR, "asset_check.py"), image, out,
               "--slot", str(p["slot"]), "--threshold", str(p.get("threshold") or 12)]
        if palette:
            cmd += ["--palette", palette]

        def on_line(line):
            if line.startswith("PROGRESS "):
                try:
                    heartbeat.progress = 8 + int(min(100, int(line.split()[1])) * 0.85)
                except ValueError:
                    pass
            log(f"  {line}")

        proc.run_streaming(cmd, cwd=EXTRACT_DIR, on_line=on_line,
                           cancel_check=cancel_check, timeout_seconds=timeout_seconds)
        with open(out, encoding="utf-8") as f:
            result = json.load(f)

        db.sb.table("rp_elements").update({
            "palette_distance": result.get("palette_distance"),
            "lighting_flags": result.get("lighting_flags"),
            "text_detected": result.get("text_detected"),
            "checks_status": "done",
            "updated_at": db.now_iso(),
        }).eq("id", element_id).execute()
        log(f"checks done for element {element_id}: Δ={result.get('palette_distance')}")
        return out, "json", "application/json"
    except Exception:
        try:
            db.sb.table("rp_elements").update(
                {"checks_status": "failed", "updated_at": db.now_iso()}).eq("id", element_id).execute()
        except Exception as e2:  # noqa: BLE001
            log(f"could not mark checks failed: {e2}")
        raise
