"""Write a queue position and start ETA into the `phase` of jobs that are
still pending, so the platform can show
"queued: 2 ahead, starts in ~15 min; now running: matte 91%"
instead of nothing. Called from the worker loop every ~30 s and from the
heartbeat of a running job, so the ETA keeps moving while something renders.

claim_farm_job() takes rows in (priority, created_at) order, so a pending
job's position is simply its index in that order, plus one if a job is
running. Rows with cancel_requested are skipped: the claim never takes them.

ETAs: a running job's remaining time is projected from its elapsed time and
progress (a flat per-engine guess made a 30-min matte at 90% look 18 s from
done). Pending jobs use videogen.estimate for video_gen and, for the other
engines, the median runtime of their recent finished jobs.
"""
import statistics
import time
from datetime import datetime

import db
from videogen import estimate

OTHER_ENGINE_SECONDS = 180.0
HISTORY_TTL_S = 600.0
HISTORY_ROWS = 400
MIN_PROGRESS_FOR_PROJECTION = 5
_last = {}
_history = {"at": 0.0, "median": {}}


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def _engine_medians():
    """Median claimed->completed seconds per engine over recent done jobs, cached."""
    if time.time() - _history["at"] < HISTORY_TTL_S:
        return _history["median"]
    try:
        rows = (db.sb.table("farm_render_jobs").select("engine,claimed_at,completed_at")
                .eq("status", "done").order("completed_at", desc=True).limit(HISTORY_ROWS).execute().data)
        runs = {}
        for r in rows:
            a, b = _ts(r.get("claimed_at")), _ts(r.get("completed_at"))
            if a and b and b > a and len(runs.setdefault(r["engine"], [])) < 30:
                runs[r["engine"]].append(b - a)
        _history["median"] = {e: statistics.median(v) for e, v in runs.items() if v}
    except Exception:  # noqa: BLE001 - keep the previous medians
        pass
    _history["at"] = time.time()
    return _history["median"]


def _seconds(row):
    if row["engine"] == "video_gen":
        try:
            return estimate.job_seconds(row.get("params") or {})
        except Exception:  # noqa: BLE001 - estimates must never break the loop
            return 600.0
    return _engine_medians().get(row["engine"], OTHER_ENGINE_SECONDS)


def _remaining(row):
    """Seconds left on a running job: project from elapsed/progress once it has moved."""
    progress = row.get("progress") or 0
    claimed = _ts(row.get("claimed_at"))
    elapsed = time.time() - claimed if claimed else 0.0
    if claimed and progress >= MIN_PROGRESS_FOR_PROJECTION:
        left = elapsed * (100 - progress) / progress
    else:
        left = _seconds(row) - elapsed
    return max(left, 30.0 if progress < 100 else 0.0)


def annotate(log):
    rows = (db.sb.table("farm_render_jobs")
            .select("id,engine,params,priority,created_at,claimed_at,status,progress,phase,cancel_requested")
            .in_("status", ["pending", "processing"]).order("priority").order("created_at").execute().data)
    running = [r for r in rows if r["status"] == "processing"]
    pending = [r for r in rows if r["status"] == "pending" and not r.get("cancel_requested")]
    wait = sum(_remaining(r) for r in running)
    now_running = ", ".join(f"{r['engine']} {r.get('progress') or 0}%" for r in running)
    suffix = f"; now running: {now_running}" if now_running else ""
    for i, r in enumerate(pending):
        ahead = i + len(running)
        phase = (f"queued: {ahead} ahead, starts in ~{estimate.fmt_eta(wait)}{suffix}" if ahead
                 else "queued: next up")
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
