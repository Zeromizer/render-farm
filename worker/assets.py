"""Content-addressed asset cache: download job assets from the 'assets' bucket
into ASSETS_DIR/<sha256>, then materialize them into the checkout at the
manifest's repo-relative paths.

Manifest shape (params.assets): [{"path": "public/broll-1.mp4",
                                  "sha256": "<64 hex>", "size": 12345}, ...]

Materialization uses hardlinks (cache and checkout normally share the
%LOCALAPPDATA% volume) with a copy fallback. Hardlinks are safe because
renders only read assets; nothing writes them in place. git clean -fd in
git_cache.checkout() wipes materialized assets before each job, so ensure()
must run on every job that carries a manifest.
"""
import hashlib
import os
import re
import shutil
import time

import httpx

import config
import db

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _safe_dest(repo, rel):
    """Resolve a manifest path inside the checkout, rejecting escapes."""
    rel = str(rel).replace("\\", "/").strip("/")
    if not rel or ":" in rel or rel.startswith("/"):
        raise RuntimeError(f"bad asset path in manifest: {rel!r}")
    dest = os.path.realpath(os.path.join(repo, rel))
    root = os.path.realpath(repo)
    if not dest.startswith(root + os.sep):
        raise RuntimeError(f"asset path escapes the repo: {rel!r}")
    return dest


def _download(sha256, size, log):
    """Ensure ASSETS_DIR/<sha256> exists; return its path."""
    cache_path = os.path.join(config.ASSETS_DIR, sha256)
    if os.path.exists(cache_path):
        os.utime(cache_path)  # mtime = last use, drives eviction
        return cache_path

    try:
        res = db.sb.storage.from_(config.ASSETS_BUCKET).create_signed_url(
            f"sha256/{sha256}", 3600
        )
        url = res.get("signedURL") or res.get("signedUrl")
        if not url:
            raise KeyError("no signed url in response")
    except Exception as e:
        raise RuntimeError(
            f"asset {sha256[:12]}… not in '{config.ASSETS_BUCKET}' bucket "
            f"({str(e)[:120]}) — run sync_assets from the laptop first"
        )

    part = cache_path + ".part"
    digest = hashlib.sha256()
    got = 0
    with httpx.stream("GET", url, timeout=httpx.Timeout(60, read=300)) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_bytes(1024 * 1024):
                f.write(chunk)
                digest.update(chunk)
                got += len(chunk)
    if digest.hexdigest() != sha256 or (size and got != int(size)):
        os.remove(part)
        raise RuntimeError(
            f"asset hash/size mismatch for {sha256[:12]}… "
            f"(got {got} bytes, sha {digest.hexdigest()[:12]}…) — corrupt upload?"
        )
    os.replace(part, cache_path)  # atomic; crashed partials stay as .part and get evicted
    return cache_path


def ensure(manifest, repo, log):
    """Download all manifest assets into the cache and materialize into repo."""
    if not isinstance(manifest, list):
        raise RuntimeError("params.assets must be a list of {path, sha256, size}")
    for entry in manifest:
        rel = (entry or {}).get("path")
        sha = str((entry or {}).get("sha256") or "").lower()
        size = (entry or {}).get("size") or 0
        if not rel or not _SHA256_RE.match(sha):
            raise RuntimeError(f"bad manifest entry: {entry!r}")
        dest = _safe_dest(repo, rel)

        cached = os.path.exists(os.path.join(config.ASSETS_DIR, sha))
        cache_path = _download(sha, size, log)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            os.remove(dest)  # manifest wins over any git-tracked placeholder
        try:
            os.link(cache_path, dest)
        except OSError:
            shutil.copy2(cache_path, dest)
        log(f"  asset {rel} ({size or '?'} bytes) "
            f"[{'cached' if cached else 'downloaded'}]")


def cleanup_old(log):
    """Delete cached assets (and stray .part files) untouched for too long."""
    if not os.path.isdir(config.ASSETS_DIR):
        return
    now = time.time()
    for name in os.listdir(config.ASSETS_DIR):
        p = os.path.join(config.ASSETS_DIR, name)
        try:
            if now - os.path.getmtime(p) > config.ASSET_CACHE_MAX_AGE_DAYS * 86400:
                log(f"cleanup: removing asset {p}")
                os.remove(p)
        except OSError:
            pass
