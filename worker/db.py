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
