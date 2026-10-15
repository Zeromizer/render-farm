"""Render farm worker: claims render_jobs from Supabase, renders on the local
GPU (Remotion / Blender), uploads results to the 'renders' bucket.

pythonw-safe (tees output to worker.log). Crash-only: uncaught errors exit the
process and the supervisor restarts it; stale jobs are reclaimed via RPC.
"""
import contextlib
import os
import shutil
import sys
import threading
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# `type nul > worker\restart-requested` = finish the current job, then exit so the
# supervisor restarts the worker on the checked-out code (no mid-job kill).
RESTART_FLAG = os.path.join(_HERE, "restart-requested")


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


# WORKER_LOG: cloud pods share this directory on the network volume, so each
# writes its own file (cloud/boot.sh sets /workspace/logs/worker-<pod>.log).
_logf = open(os.environ.get("WORKER_LOG") or os.path.join(_HERE, "worker.log"), "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)

import time

import assets
import cloud_lane
import config
import db
from still import make_still
import fetch_lane
import git_cache
import light_lane
import locks
import proc
import venvs
from heartbeat import Heartbeat
from job_stats import JobStats
import queue_status
from runners import (asset_check, audio_gen, blender, clip_edit, footage, frame_extract, hyperframes,
                     lettering_fix, matte, planar_patch, python_script, ramp_shot, reference_extract, remotion,
                     video_gen, video_split, voiceover)
from videogen import comfy_client, ram_gate

# Engines whose runners drive the headless ComfyUI; after one of these the
# worker checks ComfyUI's host RAM (comfy_client.recycle_if_bloated).
COMFY_ENGINES = {"video_gen", "audio_gen", "footage", "voiceover", "lettering_fix", "clip_edit"}
# Plus python: spike scripts drive ComfyUI over HTTP and leave it bloated too
# (2026-10-10: 16.3 GB after an H3 spike). Under the threshold it is a no-op.
RECYCLE_AFTER = COMFY_ENGINES | {"python"}

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
           "video_gen": video_gen.run,
           # Tracked corner-pin of clean artwork (plate, badge) onto a finished
           # clip: fixes the text H3 gets wrong without regenerating. CPU only.
           "planar_patch": planar_patch.run,
           # Correct plates/badges by letting H3 propagate the REAL lettering from a
           # reference photo through the clip (masked inpaint, ComfyUI-vlo nodes).
           # Reads as the model's own pixels where a corner pin reads as pasted on.
           "lettering_fix": lettering_fix.run,
           # Surgical H3 edits of a generated clip: remove/replace what is in boxes,
           # extend/prepend seconds, bridge two shots, retake the sound. Same masked
           # inpaint graph as lettering_fix; untouched pixels are put back exactly.
           "clip_edit": clip_edit.run,
           # An instrumental music bed from a description (YuE2 + instrumental
           # LoRA) in the same headless ComfyUI video_gen uses. Feeds the
           # platform's music library; a minute or two per bed.
           "audio_gen": audio_gen.run,
           # Non-linear footage timeline: probe/assemble are CPU-only and
           # answer with a manifest, never a media file, so they keep the
           # separate collector video_gen's MP4 one would misfile.
           "footage": footage.run,
           # Narration with per-word timings: the TTS studio's OmniVoice /
           # Chatterbox (loaded per job) or edge-tts, timed by faster-whisper
           # and mapped back onto the written script. Explainers key off it.
           "voiceover": voiceover.run,
           # Speed-ramp copy (fx-library tools/ramp_copy.py): analyze a
           # reference, or fit H3 takes / a hold still to one of its shots and
           # render it (TIME-01 / CAM-13 in the same checkout) with its cut blur.
           # Clones fx-library; rembg on the CPU, Remotion like the remotion engine.
           "ramp_shot": ramp_shot.run}

# Engines that work on a storage object, not a repo — the clone is skipped and
# repo_url is a "-" placeholder (the column is NOT NULL).
NO_CLONE = {"reference_extract", "asset_check", "frame_extract", "video_split",
            "matte", "video_gen", "planar_patch", "audio_gen", "footage", "voiceover", "lettering_fix",
            "clip_edit"}


def log(msg):
    print(f"[{db.now_iso()}] {msg}", flush=True)


def run_job(job, log=log):
    jid = job["id"]
    engine = job["engine"]
    runner = RUNNERS.get(engine)
    if runner is None:
        raise RuntimeError(f"unknown engine: {engine}")
    timeout_seconds = int(job.get("timeout_minutes") or 120) * 60

    work_dir = os.path.join(config.WORK_DIR, jid)
    os.makedirs(work_dir, exist_ok=True)

    def cancel_check():
        return db.cancel_requested(jid)

    # A checkout is shared by every job on that repo, in either lane
    # (worker/light_lane.py): hold it from checkout until the render is done.
    repo_lock = (contextlib.nullcontext() if engine in NO_CLONE
                 else locks.named(f"repo:{git_cache.repo_dir(job['repo_url'])}"))
    with Heartbeat(jid) as hb:
        with repo_lock:
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
        remote = db.upload_output(jid, out_local, ext, content_type)
        signed = db.create_signed_url(remote)

        # The video's still, from the file already on this disk (worker/still.py):
        # the platform shows it instead of opening the master. Best effort.
        poster = None
        still_local = make_still(out_local, ext, os.path.join(work_dir, "still.jpg"), log)
        if still_local:
            try:
                poster = db.upload_file(f"outputs/{jid}.jpg", still_local, "image/jpeg")
            except Exception as e:
                log(f"still: upload failed ({str(e)[:160]})")

    db.update_job(jid, {
        "status": "done",
        "output_path": remote,
        "output_ext": ext,
        **({"poster_path": poster} if poster else {}),
        "signed_url": signed,
        "signed_url_expires_at": db.now_iso(),  # informational; MCP re-mints anyway
        "progress": 100,
        # a runner may leave a one-line summary here (matte: "done: tensorrt
        # 0.235 s/frame"); status, not phase, is what says the job finished
        "phase": job.get("_done_phase") or "done",
        "completed_at": db.now_iso(),
    })
    shutil.rmtree(work_dir, ignore_errors=True)
    return remote


def sweep_caches():
    for name, fn in (("repos/work", git_cache.cleanup_old), ("assets", assets.cleanup_old),
                     ("venvs", venvs.cleanup_old)):
        try:
            fn(log)
        except Exception as e:
            log(f"cleanup {name} error (continuing): {str(e)[:160]}")


def main():
    from singleton import ensure_single_instance
    ensure_single_instance("worker")
    cloud = bool(config.WORKER_ENGINES)  # this is the RunPod pod (cloud_lane.py)
    log(f"render worker starting (cache={config.CACHE_DIR}"
        + (f", cloud pod for {','.join(sorted(config.WORKER_ENGINES))})" if cloud else ")"))
    sweep_caches()
    last_sweep = time.time()
    try:
        # A ComfyUI left bloated by jobs the previous worker ran (no job of ours
        # is running yet, so this is the same between-jobs moment as the loop's).
        comfy_client.recycle_if_bloated(log)
    except Exception as e:  # noqa: BLE001
        log(f"comfyui recycle error (continuing): {str(e)[:160]}")
    try:
        n = db.reclaim_stale()
        if n:
            log(f"reclaimed {n} stale job(s)")
    except Exception as e:
        log(f"reclaim error (continuing): {str(e)[:160]}")
    if cloud:
        return cloud_loop()
    try:
        # Provider downloads (Seedance clips from a Beijing CDN the platform's
        # servers cannot reach) run beside this loop, never queued behind it.
        fetch_lane.start(log)
    except Exception as e:  # noqa: BLE001
        log(f"fetch lane not started (continuing): {str(e)[:160]}")
    try:
        # Tells the platform which subjects have planar_patch artwork on this box.
        planar_patch.publish_library_index(log)
    except Exception as e:
        log(f"patch library index not published (continuing): {str(e)[:160]}")
    if config.CLOUD_LANE:
        try:
            cloud_lane.start(log)
        except Exception as e:  # noqa: BLE001 - without it the PC fallback still runs the jobs
            log(f"cloud lane not started (continuing): {str(e)[:160]}")
    cloud_gated = set()

    light = None
    heavy_exclude = set()
    if config.LIGHT_LANE:
        try:
            heavy_exclude = light_lane.LIGHT_ENGINES | db.gated_engines()
            light = light_lane.start(process_job, RESTART_FLAG,
                                     lambda m: log(f"[light] {m}"))
        except Exception as e:  # noqa: BLE001 - without the lane the loop takes every engine
            heavy_exclude = set()
            log(f"light lane not started (continuing single-lane): {str(e)[:160]}")

    claim_err_logged = False
    polls = 0
    while True:
        polls += 1
        if polls % config.RECLAIM_EVERY_POLLS == 1:
            try:
                db.reclaim_stale()
            except Exception:
                pass
        if polls % 5 == 1:
            # Gating changes at runtime (an engine moved to the cloud or back). Every
            # ~15 s: at go-live a once-a-minute read let the PC claim a job queued
            # 20 s after its engine was gated.
            try:
                cloud_gated = db.gated_engines(config.CLOUD_CAPABILITY)
                if light is not None:
                    heavy_exclude = light_lane.LIGHT_ENGINES | db.gated_engines()
            except Exception:
                pass
        if polls % 10 == 1:
            try:
                queue_status.annotate(log)   # "queued: N ahead, starts in ~M min" on waiting rows
            except Exception as e:
                log(f"queue annotate error (ignored): {str(e)[:120]}")
        if os.path.exists(RESTART_FLAG):
            # Clean deploy: exit between jobs, the supervisor respawns us on the new code.
            if light is not None:
                light_lane.drain(log)
            os.remove(RESTART_FLAG)
            log("restart requested: exiting between jobs")
            sys.exit(0)
        try:
            # With the light lane on, this loop leaves its engines to it; the RPC
            # cannot filter by engine, so it claims with a conditional update.
            job = db.claim_conditional(exclude=heavy_exclude) if light is not None else db.claim_job()
            if (not job and cloud_gated and config.CLOUD_FALLBACK_MINUTES > 0
                    and ram_gate.available_gb() >= config.CLOUD_FALLBACK_MIN_AVAIL_RAM_GB):
                # No pod took it in time (out of stock, budget used, broken boot).
                job = db.claim_conditional(include=cloud_gated, created_before=cloud_lane.fallback_cutoff())
                if job:
                    log(f"cloud fallback: {job['engine']} job waited over "
                        f"{config.CLOUD_FALLBACK_MINUTES:g} min; running it here")
            claim_err_logged = False
        except Exception as e:
            if not claim_err_logged:
                log(f"claim error (retrying quietly): {str(e)[:160]}")
                claim_err_logged = True
            time.sleep(config.POLL_SECONDS)
            continue
        if not job:
            # Between jobs only, and with the light lane idle too, so nothing is
            # mid-render (the fetch lane only writes new assets, never reads old ones).
            if time.time() - last_sweep > config.CLEANUP_EVERY_SECONDS and light_lane.is_idle():
                sweep_caches()
                last_sweep = time.time()
            time.sleep(config.POLL_SECONDS)
            continue
        process_job(job, log)


def cloud_loop():
    """The RunPod pod's loop: only WORKER_ENGINES, one job at a time, and when
    nothing has come for IDLE_EXIT_MINUTES, terminate the pod and exit."""
    claimed_by = f"cloud:{os.environ.get('RUNPOD_POD_ID', 'local')}"
    engines = set(config.WORKER_ENGINES)
    last_work = time.time()
    claim_err_logged = False
    polls = 0
    while True:
        polls += 1
        if polls % config.RECLAIM_EVERY_POLLS == 1:
            try:
                db.reclaim_stale()
                # An engine routed back to the PC (its gpu-cloud row deleted) leaves
                # this pod's list within a minute, not when the pod ends.
                gated = db.gated_engines(config.CLOUD_CAPABILITY)
                if gated:
                    engines = set(config.WORKER_ENGINES) & gated
            except Exception:
                pass
        if os.path.exists(RESTART_FLAG):
            os.remove(RESTART_FLAG)
            log("restart requested: exiting between jobs")
            sys.exit(0)
        try:
            job = db.claim_conditional(include=engines, claimed_by=claimed_by) if engines else None
            claim_err_logged = False
        except Exception as e:
            if not claim_err_logged:
                log(f"claim error (retrying quietly): {str(e)[:160]}")
                claim_err_logged = True
            time.sleep(config.POLL_SECONDS)
            continue
        if job:
            process_job(job, log)
            last_work = time.time()
            continue
        if time.time() - last_work > config.IDLE_EXIT_MINUTES * 60:
            cloud_lane.terminate_self(log)
            sys.exit(0)
        time.sleep(config.POLL_SECONDS)


def process_job(job, log):
    """Run one claimed job to its final status. Called by this loop and by the
    light lane's thread (worker/light_lane.py), each with its own log."""
    jid = job["id"]
    lane = "light" if threading.current_thread().name == "light-lane" else "main"
    log(f"claimed {jid} engine={job['engine']} repo={job['repo_url']}@{job.get('git_ref')}")
    if lane == "main":
        # Tells the light lane what is running here; it stays off the box
        # during engines that need all of it (light_lane.PAUSE_DURING).
        light_lane.main_engine[0] = job["engine"]
        if config.LIGHT_LANE and job["engine"] in light_lane.PAUSE_DURING:
            light_lane.wait_for_light(log)
    try:
        # RAM/VRAM/GPU/CPU this job cost, to the log and job_stats.jsonl
        with JobStats(jid, job["engine"], log, lane=lane):
            remote = run_job(job, log)
        log(f"done {jid} -> {remote}")
    except proc.Canceled:
        db.update_job(jid, {"status": "canceled", "phase": "canceled",
                            "completed_at": db.now_iso()})
        log(f"canceled {jid}")
    except Exception as e:
        traceback.print_exc()
        db.update_job(jid, {"status": "failed", "error": str(e)[:2000],
                            "phase": "failed", "completed_at": db.now_iso()})
        log(f"failed {jid}: {str(e)[:300]}")
    # Here, between jobs, nothing of ours is using ComfyUI, so restarting it
    # cannot fail a job (2026-09-27: a restart from outside the worker did).
    # These engines are never light-lane engines, so this is still the loop.
    if job["engine"] in RECYCLE_AFTER:
        try:
            comfy_client.recycle_if_bloated(log)
        except Exception as e:  # noqa: BLE001 - housekeeping must not stop the loop
            log(f"comfyui recycle error (ignored): {str(e)[:200]}")
    if lane == "main":
        light_lane.main_engine[0] = None

if __name__ == "__main__":
    main()
