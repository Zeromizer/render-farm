"""Supabase access for the render worker.

Projects (2026-11-11): one worker can serve more than one platform, each with its own
Supabase project (config.PROJECTS: "aimotion" always, "nbd" when its env is set). Every
call goes through `sb`, which resolves to the client of the thread's current project
(use_project / project). Claims walk the projects round-robin and tag the row with
"_project"; render_worker.process_job then runs the whole job under that project, so
runners need no change. With one project configured nothing differs from before.
"""
import contextlib
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

DEFAULT_PROJECT = config.DEFAULT_PROJECT
_local = threading.local()


def projects():
    """Configured project names, the default first."""
    return list(config.PROJECTS)


def current_project():
    """The project this thread's calls go to. A new thread starts on the default:
    thread-locals do not carry over, so a thread working for a job must be handed
    the job's project (heartbeat.Heartbeat is)."""
    return getattr(_local, "project", DEFAULT_PROJECT)


def use_project(name):
    """Point this thread's calls at a project. An unknown name fails loudly: a job run
    against the wrong database would write its result where nobody looks."""
    name = name or DEFAULT_PROJECT
    if name not in config.PROJECTS:
        raise RuntimeError(f"unknown project: {name}")
    _local.project = name
    return name


@contextlib.contextmanager
def project(name):
    """Run a block under a project, then put the thread back on the one it had."""
    before = current_project()
    use_project(name)
    try:
        yield name
    finally:
        _local.project = before


def job_project(job):
    """The project a claimed row came from (an untagged row: the default)."""
    return (job or {}).get("_project") or DEFAULT_PROJECT


_rr_lock = threading.Lock()
_rr_next = [0]


def _rotation():
    """Projects in claim order, starting after the one that last won, so a busy queue
    cannot starve the other (round-robin). One project: just that one."""
    names = projects()
    with _rr_lock:
        start = _rr_next[0] % len(names)
    return names[start:] + names[:start]


def _won(name):
    names = projects()
    with _rr_lock:
        _rr_next[0] = (names.index(name) + 1) % len(names)


def _tag(row, name):
    row["_project"] = name
    return row


_down = {}   # project -> the error last seen while it was failing (logged once per outage)


def _each(fn, claim=True):
    """fn(name) under each project in round-robin order; the first truthy answer wins
    and is tagged. With one project an error propagates exactly as before. With more,
    one project being down (its database unreachable) must not stop claims from the
    other: its error is logged once per outage, and raised only when every project
    failed in the same pass. claim=False (a look, not a claim) leaves the
    round-robin where it is."""
    names = _rotation()
    if len(names) == 1:
        with project(names[0]):
            row = fn(names[0])
        return _tag(row, names[0]) if row else None
    errors = []
    for name in names:
        try:
            with project(name):
                row = fn(name)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            if name not in _down:
                print(f"[{now_iso()}] project {name}: claim error, trying the others: {str(e)[:160]}",
                      flush=True)
            _down[name] = e
            continue
        if name in _down:
            _down.pop(name, None)
            print(f"[{now_iso()}] project {name}: claims working again", flush=True)
        if row:
            if claim:
                _won(name)
            return _tag(row, name)
    if errors and len(errors) == len(names):
        raise errors[0]
    return None


def _for(value, name):
    """include/exclude may be one set for every project or a {project: set} dict
    (gated engines differ per database). A dict without the project: None."""
    if isinstance(value, dict):
        return value.get(name)
    return value


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
        proj = current_project()
        key = (proj, name if name.startswith(_LONG_LIVED) else "shared")
        with self._guard:
            client = self._clients.get(key)
            if client is None:
                url, service_key = config.PROJECTS[proj]
                client = self._clients[key] = create_client(
                    url, service_key,
                    options=ClientOptions(storage_client_timeout=STORAGE_TIMEOUT_SECONDS))
            return client

    def __getattr__(self, name):
        return getattr(self._client(), name)


sb = _Clients()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def claim_job():
    """claim_farm_job() in each project, round-robin; the row comes back tagged."""
    def one(_name):
        rows = sb.rpc("claim_farm_job").execute().data or []
        return rows[0] if rows else None
    return _each(one)


def claim_conditional(include=None, exclude=(), created_before=None, claimed_by=None, phase="cloning"):
    """_claim_conditional in each project, round-robin; the row comes back tagged.
    include / exclude: a set for every project, or {project: set}. A project missing
    from either dict is skipped (its gating could not be read, so the render loop must
    not guess what to leave to the lanes there), as is one whose include set is empty."""
    def one(name):
        if isinstance(exclude, dict) and name not in exclude:
            return None
        inc = _for(include, name)
        if isinstance(include, dict) and not inc:
            return None
        return _claim_conditional(inc, _for(exclude, name) or (), created_before, claimed_by, phase)
    return _each(one)


def any_pending(engines):
    """Is a job of these engines waiting in any project? The lanes ask before they
    measure the box."""
    def one(_name):
        rows = (sb.table("farm_render_jobs").select("id")
                .in_("engine", sorted(engines)).eq("status", "pending")
                .eq("cancel_requested", False).limit(1).execute().data)
        return {"pending": True} if rows else None
    return bool(_each(one, claim=False))


def _claim_conditional(include=None, exclude=(), created_before=None, claimed_by=None, phase="cloning"):
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
                            "phase": phase, "error": None,
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


def gated_engines_by_project(capability=None):
    """{project: gated_engines(capability)}: gating lives in each database. With one
    project an error propagates as before; with more, a project that cannot be read is
    left out (claim_conditional then skips it), so one database being down at startup
    cannot cost the other its light lane. Raises only when none could be read."""
    names = projects()
    out, errors = {}, []
    for name in names:
        try:
            with project(name):
                out[name] = gated_engines(capability)
        except Exception as e:  # noqa: BLE001
            if len(names) == 1:
                raise
            errors.append(e)
    if errors and len(errors) == len(names):
        raise errors[0]
    return out


def gated_engines(capability=None):
    """Engines claim_farm_job() only gives to a worker with a capability
    (farm_engine_capabilities); the render loop advertises none. With
    capability: only the engines gated behind that one. Asks the thread's
    current project."""
    q = sb.table("farm_engine_capabilities").select("engine")
    if capability:
        q = q.eq("capability", capability)
    rows = q.execute().data or []
    return {r["engine"] for r in rows}


def reclaim_stale():
    """reclaim_stale_farm_jobs in every project. One project: its answer as is."""
    names = projects()
    if len(names) == 1:
        with project(names[0]):
            return _reclaim_stale()
    total, errors = 0, []
    for name in names:
        try:
            with project(name):
                n = _reclaim_stale()
            total += n if isinstance(n, int) else 0
        except Exception as e:  # noqa: BLE001 - one project down must not stop the other's reclaim
            errors.append(e)
    if len(errors) == len(names):
        raise errors[0]
    return total


def _reclaim_stale():
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
