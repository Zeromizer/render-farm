"""video_split runner: a Seedance take → selectable shots
(render-asset-gate-spec.md §10).

params (jsonb):
  split:
    generation_id  rp_scene_generations.id
    scene_id / org_id / job_id
    bucket / path  the take
    shot_count     what the prompt declared (informational)
    boundaries     optional [seconds] — declared cut points win over detection

Each shot is stored as its own video asset (with a poster, so the grid's
tiles are cheap) and one rp_scene_shots row per (generation, index).
"""
import json
import os

import db
import proc
from venvs import venv_python
from runners import gate_common

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(_WORKER_DIR, "extract")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    p = (job.get("params") or {}).get("split") or {}
    gen_id = p.get("generation_id")
    scene_id = p.get("scene_id")
    org_id = p.get("org_id")
    if not (gen_id and scene_id and org_id and p.get("bucket") and p.get("path")):
        raise RuntimeError("video_split needs params.split.{generation_id, scene_id, org_id, bucket, path}")

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}
    try:
        video = gate_common.download(p["bucket"], p["path"], work_dir, "take", log)
        heartbeat.progress = 5
        py = venv_python(os.path.join(EXTRACT_DIR, "requirements.txt"), log, run_kw)
        out_dir = os.path.join(work_dir, "shots")
        cmd = [py, "-u", os.path.join(EXTRACT_DIR, "video_split.py"), video, out_dir]
        if p.get("shot_count"):
            cmd += ["--shot-count", str(int(p["shot_count"]))]
        if p.get("boundaries"):
            cmd += ["--boundaries", ",".join(str(float(b)) for b in p["boundaries"])]

        def on_line(line):
            if line.startswith("PROGRESS "):
                try:
                    heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.6)
                except ValueError:
                    pass
            log(f"  {line}")

        proc.run_streaming(cmd, cwd=EXTRACT_DIR, on_line=on_line,
                           cancel_check=cancel_check, timeout_seconds=timeout_seconds)
        with open(os.path.join(out_dir, "index.json"), encoding="utf-8") as f:
            index = json.load(f)

        rows = []
        n = len(index["shots"])
        for i, sh in enumerate(index["shots"]):
            asset_id, _sha = gate_common.put_asset(
                org_id, sh["clip"], f"shot-{sh['index']}.mp4", "video/mp4", "video",
                f"seedance_shot:{gen_id}#{sh['index']}", poster_local=sh.get("poster"))
            poster_path = None
            a = db.sb.table("rp_assets").select("poster_path").eq("id", asset_id).execute().data
            if a:
                poster_path = a[0].get("poster_path")
            row = {
                "org_id": org_id, "scene_id": scene_id, "generation_id": gen_id,
                "index": int(sh["index"]), "start_s": sh["start_s"], "end_s": sh["end_s"],
                "asset_id": asset_id, "poster_path": poster_path,
            }
            db.sb.table("rp_scene_shots").upsert(row, on_conflict="generation_id,index").execute()
            rows.append(row)
            log(f"shot {sh['index']}: {sh['start_s']}-{sh['end_s']}s -> {asset_id}")
            heartbeat.progress = 65 + int(30 * (i + 1) / max(1, n))

        db.sb.table("rp_scene_generations").update(
            {"shots_status": "done"}).eq("id", gen_id).execute()

        report = os.path.join(work_dir, "shots.json")
        with open(report, "w", encoding="utf-8") as f:
            json.dump({"generation_id": gen_id, "source": index.get("source"), "shots": rows}, f, indent=1)
        return report, "json", "application/json"
    except Exception:
        try:
            db.sb.table("rp_scene_generations").update({"shots_status": "failed"}).eq("id", gen_id).execute()
        except Exception as e2:  # noqa: BLE001
            log(f"could not mark shots failed: {e2}")
        raise
