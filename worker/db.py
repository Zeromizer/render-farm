"""Supabase access for the render worker."""
from datetime import datetime, timezone

import threading
import time

import httpcore
import httpx
from supabase import ClientOptions, create_client

import config

# Storage's default of 20 s failed a finished 22 MB render on the home uplink (2026-10-31).
STORAGE_TIMEOUT_SECONDS = 300
_LONG_LIVED = ("light-lane", "music-lane", "fetch-lane", "MainThread")


class _Clients:
    """A Supabase client per lane. A client is one HTTP/2 connection, and when that drops,
    every call on it fails at once: on one shared client that was up to 3 lanes in the
    same instant, 12 times in the load test's first 16 min (2026-10-31). Short-lived
    threads (heartbeats, stats) share one, so a job does not cost a new connection."""

    def __init__(self):
        self._guard = threading.Lock()
        self._clients = {}

    def _client(self):
        name = threading.current_thread().name
        key = name if name.startswith(_LONG_LIVED) else "shared"
        with self._guard:
            client = self._clients.get(key)
            if client is None:
                client = self._clients[key] = create_client(
                    config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY,
                    options=ClientOptions(storage_client_timeout=STORAGE_TIMEOUT_SECONDS))
            return client

    def __getattr__(self, name):
        return getattr(self._client(), name)


sb = _Clients()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def claim_job():
    rows = sb.rpc("claim_farm_job").execute().data or []
    return rows[0] if rows else None


def claim_conditional(include=None, exclude=(), created_before=None, claimed_by=None):
    """Claim the next pending job (priority, then age) whose engine is in include
    / not in exclude, as claim_farm_job() would set it. A conditional update,
    atomic per row: used when the light lane splits engines between two claimers
    (worker/light_lane.py), since the RPC cannot filter by engine, and by the
    cloud pod, which claims only its own engines. created_before (ISO time):
    only jobs queued before it (the PC's fallback for cloud-gated jobs).
    claimed_by: stamped on the row ("cloud:<pod id>"), so the PC's cloud lane
    can tell a pod that works from one that never claims."""
    q = (sb.table("farm_render_jobs").select("id,attempts")
         .eq("status", "pending").eq("cancel_requested", False))
    if include:
        q = q.in_("engine", sorted(include))
    if exclude:
        q = q.not_.in_("engine", sorted(exclude))
    if created_before:
        q = q.lt("created_at", created_before)
    for r in _retrying(lambda: q.order("priority").order("created_at").limit(5).execute().data) or []:
        stamp = now_iso()
        try:
            won = (sb.table("farm_render_jobs")
                   .update({"status": "processing", "claimed_at": stamp, "heartbeat_at": stamp,
                            "attempts": (r.get("attempts") or 0) + 1, "progress": 0,
                            "phase": "cloning", "error": None,
                            **({"claimed_by": claimed_by} if claimed_by else {})})
                   .eq("id", r["id"]).eq("status", "pending").execute().data)
        except (httpx.TransportError, httpcore.ProtocolError):
            # The update may have landed with only the reply lost: then the row is ours, and
            # raising would leave it "processing" with nobody running it (5af15c24, 2026-10-31).
            mine = _claimed_at(r["id"], stamp)
            if mine:
                return mine
            continue
        if won:
            return won[0]
    return None


def _moment(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None


def _claimed_at(job_id, stamp):
    """The job's row if it is processing under this exact claim time, else None. The time
    is microseconds, so it tells this attempt from any other worker's."""
    rows = _retrying(lambda: sb.table("farm_render_jobs").select("*").eq("id", job_id).execute().data)
    row = rows[0] if rows else None
    if row and row.get("status") == "processing" and _moment(row.get("claimed_at")) == _moment(stamp):
        return row
    return None


def gated_engines(capability=None):
    """Engines claim_farm_job() only gives to a worker with a capability
    (farm_engine_capabilities); the render loop advertises none. With
    capability: only the engines gated behind that one."""
    q = sb.table("farm_engine_capabilities").select("engine")
    if capability:
        q = q.eq("capability", capability)
    rows = q.execute().data or []
    return {r["engine"] for r in rows}


def reclaim_stale():
    return sb.rpc("reclaim_stale_farm_jobs", {"p_stale_minutes": config.STALE_MINUTES}).execute().data


def _retrying(fn, attempts=3):
    """Run a short Supabase call, retrying a dropped connection. The client keeps
    one HTTP/2 connection; after a quiet spell (a long venv build, an H3 load)
    the server closes it and the next call raises "Server disconnected"
    (2026-10-15: a cloud matte job failed in its cancel check that way)."""
    for i in range(attempts):
        try:
            return fn()
        except (httpx.TransportError, httpcore.ProtocolError):
            if i == attempts - 1:
                raise
            time.sleep(1 + i)


def update_job(job_id, fields):
    _retrying(lambda: sb.table("farm_render_jobs").update(fields).eq("id", job_id).execute())


def set_phase(job_id, phase, progress=None):
    fields = {"phase": phase, "heartbeat_at": now_iso()}
    if progress is not None:
        fields["progress"] = progress
    update_job(job_id, fields)


def cancel_requested(job_id):
    try:
        rows = _retrying(lambda: sb.table("farm_render_jobs").select("cancel_requested")
                         .eq("id", job_id).execute().data)
    except (httpx.TransportError, httpcore.ProtocolError):
        return False  # a cancel check must never fail the job; the next one asks again
    return bool(rows and rows[0]["cancel_requested"])


def upload_output(job_id, local_path, ext, content_type):
    return upload_file(f"outputs/{job_id}.{ext}", local_path, content_type)


def upload_file(remote, local_path, content_type):
    """Upload to an explicit remote path, for the sidecars a job emits
    alongside its one real output (the matte proof sheet is the first)."""
    def send():
        with open(local_path, "rb") as f:
            # Pass the file object (not f.read()) so httpx streams the body —
            # large finals no longer load fully into memory.
            sb.storage.from_(config.BUCKET).upload(
                remote, f, {"content-type": content_type, "upsert": "true"}
            )
    # upsert makes a retry safe, and the render it would otherwise throw away cost minutes.
    _retrying(send)
    return remote


def create_signed_url(remote_path):
    res = _retrying(lambda: sb.storage.from_(config.BUCKET).create_signed_url(
        remote_path, config.SIGNED_URL_SECONDS
    ))
    return res.get("signedURL") or res.get("signedUrl")
