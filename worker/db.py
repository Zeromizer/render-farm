"""Supabase access for the render worker."""
from datetime import datetime, timezone

from supabase import create_client

import config

sb = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def claim_job():
    rows = sb.rpc("claim_farm_job").execute().data or []
    return rows[0] if rows else None


def claim_conditional(include=None, exclude=()):
    """Claim the next pending job (priority, then age) whose engine is in include
    / not in exclude, as claim_farm_job() would set it. A conditional update,
    atomic per row: used when the light lane splits engines between two claimers
    (worker/light_lane.py), since the RPC cannot filter by engine."""
    q = (sb.table("farm_render_jobs").select("id,attempts")
         .eq("status", "pending").eq("cancel_requested", False))
    if include:
        q = q.in_("engine", sorted(include))
    if exclude:
        q = q.not_.in_("engine", sorted(exclude))
    for r in q.order("priority").order("created_at").limit(5).execute().data or []:
        won = (sb.table("farm_render_jobs")
               .update({"status": "processing", "claimed_at": now_iso(), "heartbeat_at": now_iso(),
                        "attempts": (r.get("attempts") or 0) + 1, "progress": 0,
                        "phase": "cloning", "error": None})
               .eq("id", r["id"]).eq("status", "pending").execute().data)
        if won:
            return won[0]
    return None


def gated_engines():
    """Engines claim_farm_job() only gives to a worker with a capability
    (farm_engine_capabilities); the render loop advertises none."""
    rows = sb.table("farm_engine_capabilities").select("engine").execute().data or []
    return {r["engine"] for r in rows}


def reclaim_stale():
    return sb.rpc("reclaim_stale_farm_jobs", {"p_stale_minutes": config.STALE_MINUTES}).execute().data


def update_job(job_id, fields):
    sb.table("farm_render_jobs").update(fields).eq("id", job_id).execute()


def set_phase(job_id, phase, progress=None):
    fields = {"phase": phase, "heartbeat_at": now_iso()}
    if progress is not None:
        fields["progress"] = progress
    update_job(job_id, fields)


def cancel_requested(job_id):
    rows = (
        sb.table("farm_render_jobs").select("cancel_requested").eq("id", job_id).execute().data
    )
    return bool(rows and rows[0]["cancel_requested"])


def upload_output(job_id, local_path, ext, content_type):
    return upload_file(f"outputs/{job_id}.{ext}", local_path, content_type)


def upload_file(remote, local_path, content_type):
    """Upload to an explicit remote path, for the sidecars a job emits
    alongside its one real output (the matte proof sheet is the first)."""
    with open(local_path, "rb") as f:
        # Pass the file object (not f.read()) so httpx streams the body —
        # large finals no longer load fully into memory.
        sb.storage.from_(config.BUCKET).upload(
            remote, f, {"content-type": content_type, "upsert": "true"}
        )
    return remote


def create_signed_url(remote_path):
    res = sb.storage.from_(config.BUCKET).create_signed_url(
        remote_path, config.SIGNED_URL_SECONDS
    )
    return res.get("signedURL") or res.get("signedUrl")
