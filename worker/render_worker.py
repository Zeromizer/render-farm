"""Render farm worker: claims render_jobs from Supabase, renders on the local
GPU (Remotion / Blender), uploads results to the 'renders' bucket.

pythonw-safe (tees output to worker.log). Crash-only: uncaught errors exit the
process and the supervisor restarts it; stale jobs are reclaimed via RPC.
"""
import os
import json
import shutil
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


class _Tee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


_logf = open(os.path.join(_HERE, "preview-worker.log" if os.environ.get("RENDER_WORKER_LANE") == "preview" else "worker.log"), "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)

import time

import assets
import config
import db
import git_cache
import proc
import venvs
from heartbeat import Heartbeat
import queue_status
from runners import (asset_check, blender, frame_extract, hyperframes, matte,
                     python_script, reference_extract, remotion, video_gen,
                     video_split)

RUNNERS = {"remotion": remotion.run, "blender": blender.run,
           "python": python_script.run,
           # HTML/GSAP compositions rendered by headless Chrome + system ffmpeg
           # (HeyGen HyperFrames). Clones like remotion; no npm ci needed.
           "hyperframes": hyperframes.run,
           "reference_extract": reference_extract.run,
           # The asset gate (render-platform lib/scenes): checks on a candidate
           # still, frames out of a static-lock pan, shots out of a take.
           "asset_check": asset_check.run,
           "frame_extract": frame_extract.run,
           "video_split": video_split.run,
           # GPU alpha matting. Lives here rather than in the agent container
           # because that container has no GPU device request at all, where
           # rembg costs 9.5 s/frame on CPU.
           "matte": matte.run,
           # MiniMax H3 text/image/reference-to-video with native audio, via
           # the headless ComfyUI at C:\ComfyUI. Replaces the remote Seedance
           # credits for b-roll. Minutes per clip on a 16 GB card.
           "video_gen": video_gen.run}

# Engines that work on a storage object, not a repo — the clone is skipped and
# repo_url is a "-" placeholder (the column is NOT NULL).
NO_CLONE = {"reference_extract", "asset_check", "frame_extract", "video_split",
            "matte", "video_gen"}


def log(msg):
    print(f"[{db.now_iso()}] {msg}", flush=True)


class PreviewRefused(Exception):
    """The preview lane claimed something it must not run; the row was re-queued."""


def preview_can_run(job):
    return job.get("engine") == "hyperframes" and (job.get("params") or {}).get("output_kind") == "still"


def upload_snapshot_batch(jid, manifest_local, times, cancel_check):
    """Upload every slide of a storyboard batch, then rewrite the local manifest with
    bucket paths (no local paths) for the caller to upload as outputs/<jid>.json.

    Raises before anything is marked done when the runner produced fewer or more
    frames than requested, when a slide upload fails, or when the job is canceled
    between slides. The row therefore ends failed / canceled, never done with a
    slide missing. Slides already uploaded are left in place; the retry rewrites
    them (upsert)."""
    with open(manifest_local, encoding="utf-8") as handle:
        batch = json.load(handle)
    snapshots = batch.get("snapshots") or []
    if len(snapshots) != len(times):
        raise RuntimeError(f"Incomplete snapshot batch: {len(snapshots)} images for {len(times)} timestamps")
    for index, (snapshot, at) in enumerate(zip(snapshots, times)):
        if float(snapshot.get("at", at)) != float(at):
            raise RuntimeError(f"snapshot batch order mismatch at slide {index}: {snapshot.get('at')} vs {at}")
        if cancel_check():
            raise proc.Canceled()
        local = snapshot.pop("file")
        if not os.path.exists(local) or os.path.getsize(local) == 0:
            raise RuntimeError(f"snapshot slide {index} is missing or empty: {local}")
        snapshot["index"] = index
        snapshot["at"] = at
        snapshot["bucket"] = config.BUCKET
        snapshot["path"] = db.upload_file(f"outputs/{jid}-slide-{index}.png", local, "image/png")
    batch.update({"version": 1, "job_id": jid, "count": len(times), "bucket": config.BUCKET, "snapshots": snapshots})
    with open(manifest_local, "w", encoding="utf-8") as handle:
        json.dump(batch, handle)
    return batch


def run_job(job):
    jid = job["id"]
    engine = job["engine"]
    if config.WORKER_LANE == "preview" and not preview_can_run(job):
        # Defence in depth behind the claim RPC's filter: hand the job back untouched
        # for the main worker instead of failing it.
        db.update_job(jid, {"status": "pending", "phase": "queued", "claimed_at": None,
                            "heartbeat_at": None, "error": None})
        raise PreviewRefused(f"preview lane refused {engine} job {jid}; returned to the queue")
    runner = RUNNERS.get(engine)
    if runner is None:
        raise RuntimeError(f"unknown engine: {engine}")
    timeout_seconds = int(job.get("timeout_minutes") or 120) * 60

    work_dir = os.path.join(config.WORK_DIR, jid)
    os.makedirs(work_dir, exist_ok=True)

    def cancel_check():
        return db.cancel_requested(jid)

    with Heartbeat(jid) as hb:
        if engine in NO_CLONE:
            db.set_phase(jid, "downloading", 1)
            repo = None
        else:
            db.set_phase(jid, "cloning", 1)
            repo = git_cache.checkout(job["repo_url"], job.get("git_ref") or "main", log)

            manifest = (job.get("params") or {}).get("assets")
            if manifest:
                db.set_phase(jid, "syncing_assets", 2)
                assets.ensure(manifest, repo, log)

        db.set_phase(jid, "rendering", 2)
        out_local, ext, content_type = runner(
            job, repo, work_dir, hb, log, cancel_check, timeout_seconds
        )

        db.set_phase(jid, "uploading", 99)
        if engine == "hyperframes" and (job.get("params") or {}).get("snapshot_times"):
            upload_snapshot_batch(jid, out_local, job["params"]["snapshot_times"], cancel_check)
        remote = db.upload_output(jid, out_local, ext, content_type)
        signed = db.create_signed_url(remote)

    db.update_job(jid, {
        "status": "done",
        "output_path": remote,
        "output_ext": ext,
        "signed_url": signed,
        "signed_url_expires_at": db.now_iso(),  # informational; MCP re-mints anyway
        "progress": 100,
        "phase": "done",
        "completed_at": db.now_iso(),
    })
    shutil.rmtree(work_dir, ignore_errors=True)
    return remote


def main():
    from singleton import ensure_single_instance
    ensure_single_instance("preview-worker" if config.WORKER_LANE == "preview" else "worker")
    log(f"render worker starting (cache={config.CACHE_DIR})")
    git_cache.cleanup_old(log)
    assets.cleanup_old(log)
    venvs.cleanup_old(log)
    try:
        n = db.reclaim_stale()
        if n:
            log(f"reclaimed {n} stale job(s)")
    except Exception as e:
        log(f"reclaim error (continuing): {str(e)[:160]}")

    claim_err_logged = False
    polls = 0
    while True:
        polls += 1
        if polls % config.RECLAIM_EVERY_POLLS == 0:
            try:
                db.reclaim_stale()
            except Exception:
                pass
        if polls % 10 == 1 and config.WORKER_LANE != "preview":
            try:
                queue_status.annotate(log)   # "queued: N ahead, starts in ~M min" on waiting rows
            except Exception as e:
                log(f"queue annotate error (ignored): {str(e)[:120]}")
        try:
            if config.WORKER_LANE == "preview":
                from preview_resources import can_start
                if not can_start():
                    time.sleep(max(5, config.POLL_SECONDS))
                    continue
            job = db.claim_job()
            claim_err_logged = False
        except Exception as e:
            if not claim_err_logged:
                log(f"claim error (retrying quietly): {str(e)[:160]}")
                claim_err_logged = True
            time.sleep(config.POLL_SECONDS)
            continue
        if not job:
            time.sleep(config.POLL_SECONDS)
            continue

        jid = job["id"]
        log(f"claimed {jid} engine={job['engine']} repo={job['repo_url']}@{job.get('git_ref')}")
        try:
            remote = run_job(job)
            log(f"done {jid} -> {remote}")
        except PreviewRefused as e:
            log(str(e))
        except proc.Canceled:
            db.update_job(jid, {"status": "canceled", "phase": "canceled",
                                "completed_at": db.now_iso()})
            log(f"canceled {jid}")
        except Exception as e:
            traceback.print_exc()
            db.update_job(jid, {"status": "failed", "error": str(e)[:2000],
                                "phase": "failed", "completed_at": db.now_iso()})
            log(f"failed {jid}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
