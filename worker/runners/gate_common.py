"""Supabase I/O shared by the asset-gate runners (asset_check, frame_extract,
video_split). Same split as reference_extract: extract/*.py is pure compute in
the cached venv; THIS side owns downloads, uploads and rows.

Assets are content-addressed exactly the way lib/media/store.ts does it —
assets/sha256/<hex>, one rp_assets row per (org, sha256) — so a frame the
platform later re-derives lands on the same row instead of beside it.
"""
import os

import config
import db


def download(bucket, path, work_dir, name, log):
    """Stream one storage object to disk (a take can be 50MB)."""
    import httpx

    res = db.sb.storage.from_(bucket).create_signed_url(path, 3600)
    url = res.get("signedURL") or res.get("signedUrl")
    if not url:
        raise RuntimeError(f"could not sign {bucket}/{path}")
    ext = os.path.splitext(path)[1]
    local = os.path.join(work_dir, name + ext)
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"download failed: HTTP {r.status_code} for {bucket}/{path}")
        with open(local, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    log(f"downloaded {bucket}/{path} -> {os.path.getsize(local)} bytes")
    return local


def _sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put_asset(org_id, local, filename, mime, kind, source_url, poster_local=None):
    """Upload content-addressed + upsert the rp_assets row. Returns (asset_id, sha256)."""
    sha = _sha256(local)
    remote = f"sha256/{sha}"
    with open(local, "rb") as f:
        db.sb.storage.from_(config.ASSETS_BUCKET).upload(
            remote, f, {"content-type": mime, "upsert": "true"})
    row = {
        "org_id": org_id,
        "sha256": sha,
        "storage_path": remote,
        "filename": filename,
        "size": os.path.getsize(local),
        "mime": mime,
        "kind": kind,
        "source_url": source_url,
    }
    if poster_local and os.path.exists(poster_local):
        poster_remote = f"posters/{sha}.jpg"
        with open(poster_local, "rb") as f:
            db.sb.storage.from_(config.ASSETS_BUCKET).upload(
                poster_remote, f, {"content-type": "image/jpeg", "upsert": "true"})
        row["poster_path"] = poster_remote
    res = db.sb.table("rp_assets").upsert(row, on_conflict="org_id,sha256").execute()
    data = res.data or []
    if not data:
        found = db.sb.table("rp_assets").select("id").eq("org_id", org_id).eq("sha256", sha).execute().data
        data = found or []
    if not data:
        raise RuntimeError("rp_assets upsert returned no row")
    return data[0]["id"], sha


def palette_path_for(scene_id):
    """The scene's active palette frame's storage path, or None."""
    rows = (db.sb.table("rp_elements")
            .select("asset_id,state")
            .eq("scene_id", scene_id).eq("slot", "palette_frame")
            .in_("state", ["locked", "approved"]).limit(1).execute().data) or []
    if not rows or not rows[0].get("asset_id"):
        return None
    a = db.sb.table("rp_assets").select("storage_path").eq("id", rows[0]["asset_id"]).execute().data
    return a[0]["storage_path"] if a else None


def org_threshold(org_id):
    rows = db.sb.table("rp_orgs").select("palette_threshold").eq("id", org_id).execute().data or []
    try:
        return float(rows[0]["palette_threshold"]) if rows else 12.0
    except (KeyError, TypeError, ValueError):
        return 12.0


def queue_checks(element_id, org_id, scene_id, slot, storage_path, log):
    """Enqueue an asset_check for a freshly filed element — what the platform's
    lib/scenes/checks.ts does, done here because the worker filed the row."""
    params = {"check": {
        "element_id": element_id, "org_id": org_id, "scene_id": scene_id, "slot": slot,
        "bucket": config.ASSETS_BUCKET, "path": storage_path,
        "palette_path": palette_path_for(scene_id) if slot != "palette_frame" else None,
        "threshold": org_threshold(org_id),
    }}
    res = db.sb.table("farm_render_jobs").insert({
        "engine": "asset_check", "repo_url": "-", "git_ref": "main",
        "priority": 200, "timeout_minutes": 10, "params": params,
    }).execute()
    job_id = (res.data or [{}])[0].get("id")
    db.sb.table("rp_elements").update(
        {"checks_status": "queued", "checks_job_id": job_id}).eq("id", element_id).execute()
    log(f"queued asset_check {job_id} for element {element_id}")
    return job_id


def ensure_slot(scene_id, slot, requires_lock, sort):
    """A detail_lock_* the pan produced may not be on the scene yet: add it as
    optional so the candidate has a row to sit in."""
    existing = db.sb.table("rp_scene_slots").select("slot").eq("scene_id", scene_id).eq("slot", slot).execute().data
    if existing:
        return
    db.sb.table("rp_scene_slots").insert({
        "scene_id": scene_id, "slot": slot, "required": False,
        "requires_lock": requires_lock, "sort": sort,
    }).execute()


def next_element_name(scene_id, slot):
    res = db.sb.table("rp_elements").select("id", count="exact").eq("scene_id", scene_id).eq("slot", slot).execute()
    n = (res.count or 0) + 1
    return f"ast_{slot}_{n}"
