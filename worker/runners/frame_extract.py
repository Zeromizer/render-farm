"""frame_extract runner: the static-lock pan's frames become lock candidates
(render-asset-gate-spec.md §7.6).

params (jsonb):
  frames:
    generation_id  rp_scene_generations.id
    scene_id / org_id / job_id
    bucket / path  the clip
    frames         [{at_s, slot}]  (position_lock, detail_lock_dash, ...)

Each frame is stored content-addressed as an image asset, filed as a
candidate element (gen_model=seedance_frame, gen_batch_id=generation_id)
under its slot — the slot is created as optional if the scene lacks it — and
an asset_check is queued for it. Ends by marking the generation's
shots_status='done' (the column doubles as "post-processing finished").
"""
import json
import os

import db
import proc
from venvs import venv_python
from runners import gate_common

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(_WORKER_DIR, "extract")

LOCK_SORT = {"position_lock": 40, "detail_lock": 50}


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    p = (job.get("params") or {}).get("frames") or {}
    gen_id = p.get("generation_id")
    scene_id = p.get("scene_id")
    org_id = p.get("org_id")
    job_id = p.get("job_id")
    frames = p.get("frames") or []
    if not (gen_id and scene_id and org_id and job_id and p.get("bucket") and p.get("path") and frames):
        raise RuntimeError("frame_extract needs params.frames.{generation_id, scene_id, org_id, job_id, bucket, path, frames}")

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}
    try:
        video = gate_common.download(p["bucket"], p["path"], work_dir, "pan", log)
        heartbeat.progress = 5
        py = venv_python(os.path.join(EXTRACT_DIR, "requirements.txt"), log, run_kw)
        out_dir = os.path.join(work_dir, "frames")
        cmd = [py, "-u", os.path.join(EXTRACT_DIR, "frame_extract.py"), video, out_dir,
               "--frames", json.dumps(frames)]

        def on_line(line):
            if line.startswith("PROGRESS "):
                try:
                    heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.5)
                except ValueError:
                    pass
            log(f"  {line}")

        proc.run_streaming(cmd, cwd=EXTRACT_DIR, on_line=on_line,
                           cancel_check=cancel_check, timeout_seconds=timeout_seconds)
        with open(os.path.join(out_dir, "index.json"), encoding="utf-8") as f:
            index = json.load(f)

        gen = (db.sb.table("rp_scene_generations").select("prompt").eq("id", gen_id).execute().data or [{}])[0]
        prompt = gen.get("prompt")

        filed = []
        n = len(index["frames"])
        for i, fr in enumerate(index["frames"]):
            slot = fr["slot"]
            base = "detail_lock" if slot.startswith("detail_lock") else slot
            gate_common.ensure_slot(scene_id, slot, requires_lock=True,
                                    sort=LOCK_SORT.get(base, 55) + i)
            asset_id, sha = gate_common.put_asset(
                org_id, fr["path"], f"{slot}-{sha_short(fr['path'])}.png", "image/png", "image",
                f"seedance_frame:{gen_id}@{fr['at_s']}")
            name = gate_common.next_element_name(scene_id, slot)
            row = db.sb.table("rp_elements").insert({
                "org_id": org_id, "job_id": job_id, "scene_id": scene_id, "slot": slot,
                "asset_id": asset_id, "element_name": name, "state": "candidate",
                "gen_model": "seedance_frame", "gen_prompt": prompt, "gen_batch_id": gen_id,
            }).execute().data[0]
            gate_common.queue_checks(row["id"], org_id, scene_id, slot, f"sha256/{sha}", log)
            filed.append({"slot": slot, "element_id": row["id"], "element_name": name, "asset_id": asset_id})
            log(f"filed {name} ({slot}) from {fr['at_s']}s")
            heartbeat.progress = 55 + int(40 * (i + 1) / max(1, n))

        db.sb.table("rp_scene_generations").update(
            {"shots_status": "done"}).eq("id", gen_id).execute()
        db.sb.table("rp_scenes").update({"manifest_ready": False}).eq("id", scene_id).execute()

        report = os.path.join(work_dir, "filed.json")
        with open(report, "w", encoding="utf-8") as f:
            json.dump({"generation_id": gen_id, "filed": filed}, f, indent=1)
        return report, "json", "application/json"
    except Exception:
        try:
            db.sb.table("rp_scene_generations").update({"shots_status": "failed"}).eq("id", gen_id).execute()
        except Exception as e2:  # noqa: BLE001
            log(f"could not mark shots failed: {e2}")
        raise


def sha_short(path):
    return gate_common._sha256(path)[:8]
