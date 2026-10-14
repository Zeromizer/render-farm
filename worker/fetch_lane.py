"""Fetch lane: copy files from a provider URL into the platform's asset store,
beside the render loop rather than in it.

Why: Seedance 2.5 serves finished clips from a Beijing CDN that refuses the
platform's Vercel servers (measured 2026-09-29: three attempts, "fetch
failed"), while this PC's home connection reaches it in seconds. Until now the
agent pulled the file itself as a fallback. A url_fetch job is seconds of
network work; in the single render loop it would wait behind a 10-minute
video_gen or a 45-minute turntable, so it runs in its own thread instead.

Claiming: the render loop's claim_farm_job() never takes url_fetch, because
farm_engine_capabilities holds ('url_fetch', 'fetch-lane') and the loop
advertises no capabilities. This lane takes rows with a conditional update
(pending -> processing), which is atomic per row. reclaim_stale_farm_jobs()
puts a row whose lane died back to pending, and the lane takes it again.

params:
  files: [{url, role, content_type}]   role is a label the platform reads back
                                       ("main", "last_frame")
Result, merged into params.result:
  files: [{role, sha256, storage_path, size, content_type}]
storage_path is sha256/<hex> in the assets bucket: the same content-addressed
path every platform upload uses, so the platform files the asset from it
without downloading the bytes again.
"""
import hashlib
import ipaddress
import os
import socket
import tempfile
import threading
import time
from urllib.parse import urljoin, urlsplit

import httpx

import config
import db

ENGINE = "url_fetch"
# Engines this lane owns. queue_status leaves them out of the render queue's
# positions and ETAs: they never wait in it.
LANE_ENGINES = {ENGINE}
POLL_SECONDS = 2
MAX_BYTES = 1024 * 1024 * 1024
ATTEMPTS = 3
MAX_REDIRECTS = 5

# This PC sits beside ComfyUI (127.0.0.1:8188), the relay bridge, the router
# and the rest of the home network, and whatever a url_fetch job points at is
# uploaded to the assets bucket. So a url must be https, on a provider host,
# and resolve to a public address — checked again on every redirect hop, or an
# allowed host could bounce the request to 127.0.0.1 (render-pc review,
# 2026-10-02). Empty FETCH_ALLOWED_HOST_SUFFIXES lifts the host list only.
ALLOWED_HOST_SUFFIXES = tuple(
    s.strip().lower().lstrip(".")
    for s in os.environ.get("FETCH_ALLOWED_HOST_SUFFIXES", "volces.com,volccdn.com,supabase.co").split(",")
    if s.strip()
)
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


class Refused(RuntimeError):
    """A url this lane will not fetch. Never retried."""


def check_url(url):
    """Raise Refused unless url is https, on an allowed host, resolving only to public addresses."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise Refused(f"url refused: https only (got {parts.scheme or 'no'} scheme)")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise Refused("url refused: no host")
    if ALLOWED_HOST_SUFFIXES and not any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES):
        raise Refused(f"url refused: {host} is not a provider host ({', '.join(ALLOWED_HOST_SUFFIXES)})")
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise RuntimeError(f"cannot resolve {host}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified or (ip.version == 4 and ip in _CGNAT)):
            raise Refused(f"url refused: {host} resolves to a non-public address ({ip})")


def _fetch_once(url, dest, beat):
    h, size = hashlib.sha256(), 0
    timeout = httpx.Timeout(120.0, connect=20.0)
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        with httpx.stream("GET", current, timeout=timeout, follow_redirects=False) as r:
            if r.is_redirect:
                location = r.headers.get("location")
                if not location:
                    raise RuntimeError("redirect without a location")
                current = urljoin(current, location)
                continue
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise RuntimeError("file is larger than 1 GiB")
                    h.update(chunk)
                    f.write(chunk)
                    beat()
        if size == 0:
            raise RuntimeError("empty response")
        return h.hexdigest(), size
    raise Refused(f"url refused: more than {MAX_REDIRECTS} redirects")


def _download(url, dest, beat):
    """Stream url to dest, returning (sha256, size). Retries a flapping connect, never a refusal."""
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return _fetch_once(url, dest, beat)
        except Refused:
            raise
        except Exception as e:  # noqa: BLE001 - retried, then reported
            last = e
            if attempt < ATTEMPTS:
                time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after {ATTEMPTS} attempts: {str(last)[:300]}")


def _run(job, log):
    jid = job["id"]
    params = job.get("params") or {}
    files = params.get("files") or []
    if not files or not all(isinstance(f, dict) and f.get("url") for f in files):
        raise RuntimeError("url_fetch needs params.files: [{url, role, content_type}]")

    last_beat = [0.0]

    def beat():
        if time.time() - last_beat[0] >= config.HEARTBEAT_SECONDS:
            last_beat[0] = time.time()
            try:
                db.update_job(jid, {"heartbeat_at": db.now_iso()})
            except Exception:  # noqa: BLE001 - a missed beat must not fail the copy
                pass

    out, total = [], 0
    with tempfile.TemporaryDirectory(prefix=f"fetch-{jid[:8]}-") as tmp:
        for i, f in enumerate(files):
            if db.cancel_requested(jid):
                raise RuntimeError("canceled")
            role = str(f.get("role") or f"file{i}")
            ctype = str(f.get("content_type") or "application/octet-stream")
            db.set_phase(jid, f"downloading {role}", int(100 * i / len(files)))
            local = os.path.join(tmp, f"{i}.bin")
            sha, size = _download(f["url"], local, beat)
            path = f"sha256/{sha}"
            with open(local, "rb") as fh:
                db.sb.storage.from_(config.ASSETS_BUCKET).upload(
                    path, fh, {"content-type": ctype, "upsert": "true"})
            out.append({"role": role, "sha256": sha, "storage_path": path, "size": size,
                        "content_type": ctype})
            total += size
            log(f"  url_fetch {jid[:8]} {role}: {size / 1e6:.1f} MB -> assets/{path}")

    db.update_job(jid, {
        "status": "done",
        "params": {**params, "result": {"files": out}},
        "progress": 100,
        "phase": f"done: {len(out)} file(s), {total / 1e6:.1f} MB",
        "completed_at": db.now_iso(),
    })


def _claim():
    rows = (db.sb.table("farm_render_jobs")
            .select("id,attempts,max_attempts")
            .eq("engine", ENGINE).eq("status", "pending").eq("cancel_requested", False)
            .order("priority").order("created_at").limit(5).execute().data) or []
    for r in rows:
        won = (db.sb.table("farm_render_jobs")
               .update({"status": "processing", "claimed_at": db.now_iso(),
                        "heartbeat_at": db.now_iso(), "attempts": (r.get("attempts") or 0) + 1,
                        "progress": 0, "phase": "downloading", "error": None})
               .eq("id", r["id"]).eq("status", "pending").execute().data)
        if won:
            return won[0]
    return None


def _loop(log):
    log("fetch lane started")
    err_logged = False
    while True:
        try:
            job = _claim()
            err_logged = False
        except Exception as e:  # noqa: BLE001
            if not err_logged:
                log(f"fetch lane claim error (retrying quietly): {str(e)[:160]}")
                err_logged = True
            time.sleep(POLL_SECONDS * 5)
            continue
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        jid = job["id"]
        try:
            _run(job, log)
            log(f"fetch lane done {jid}")
        except Exception as e:  # noqa: BLE001 - one bad url must not stop the lane
            canceled = str(e) == "canceled"
            try:
                db.update_job(jid, {"status": "canceled" if canceled else "failed",
                                    "phase": "canceled" if canceled else "failed",
                                    "error": None if canceled else str(e)[:2000],
                                    "completed_at": db.now_iso()})
            except Exception:  # noqa: BLE001
                pass
            log(f"fetch lane {'canceled' if canceled else 'failed'} {jid}: {str(e)[:300]}")


def start(log):
    """Run the lane on a daemon thread; it dies with the worker and restarts with it."""
    t = threading.Thread(target=_loop, args=(log,), name="fetch-lane", daemon=True)
    t.start()
    return t
