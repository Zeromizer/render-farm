"""Write a queue position and start ETA into the `phase` of jobs that are
still pending, so the platform can show "queued: 2 ahead, starts in ~15 min"
instead of nothing. Called from the worker loop every ~30 s.

claim_farm_job() takes rows in (priority, created_at) order, so a pending
job's position is simply its index in that order, plus one if a job is
running. ETAs use videogen.estimate for video_gen jobs and a flat guess for
the render engines (their runtime depends on the composition).
"""
import time

import db
from videogen import estimate

OTHER_ENGINE_SECONDS = 180.0
_last = {}


def _seconds(row):
    if row["engine"] == "video_gen":
        try:
            return estimate.job_seconds(row.get("params") or {})
        except Exception:  # noqa: BLE001 - estimates must never break the loop
            return 600.0
    return OTHER_ENGINE_SECONDS


def annotate(log):
    rows = (db.sb.table("farm_render_jobs")
            .select("id,engine,params,priority,created_at,status,progress,phase")
            .in_("status", ["pending", "processing"]).order("priority").order("created_at").execute().data)
    running = [r for r in rows if r["status"] == "processing"]
    pending = [r for r in rows if r["status"] == "pending"]
    wait = 0.0
    for r in running:
        wait += _seconds(r) * (1 - (r.get("progress") or 0) / 100.0)
    for i, r in enumerate(pending):
        ahead = i + len(running)
        phase = f"queued: {ahead} ahead, starts in ~{estimate.fmt_eta(wait)}" if ahead else "queued: next up"
        if _last.get(r["id"]) != phase and r.get("phase") != phase:
            try:
                db.update_job(r["id"], {"phase": phase})
                _last[r["id"]] = phase
            except Exception as e:  # noqa: BLE001
                log(f"queue annotate {r['id'][:8]}: {str(e)[:120]}")
        wait += _seconds(r)
    # forget finished ids
    live = {r["id"] for r in rows}
    for k in list(_last):
        if k not in live:
            _last.pop(k, None)
    return len(pending)
