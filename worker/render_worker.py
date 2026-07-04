"""Render farm worker: claims render_jobs from Supabase, renders on the local
GPU (Remotion / Blender), uploads results to the 'renders' bucket.

pythonw-safe (tees output to worker.log). Crash-only: uncaught errors exit the
process and the supervisor restarts it; stale jobs are reclaimed via RPC.
"""
import os
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


_logf = open(os.path.join(_HERE, "worker.log"), "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)

import time

import config
import db
import git_cache
import proc
from heartbeat import Heartbeat
from runners import blender, remotion

RUNNERS = {"remotion": remotion.run, "blender": blender.run}


def log(msg):
    print(f"[{db.now_iso()}] {msg}", flush=True)


def run_job(job):
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

    with Heartbeat(jid) as hb:
        db.set_phase(jid, "cloning", 1)
        repo = git_cache.checkout(job["repo_url"], job.get("git_ref") or "main", log)

        db.set_phase(jid, "rendering", 2)
        out_local, ext, content_type = runner(
            job, repo, work_dir, hb, log, cancel_check, timeout_seconds
        )

        db.set_phase(jid, "uploading", 99)
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
    log(f"render worker starting (cache={config.CACHE_DIR})")
    git_cache.cleanup_old(log)
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
        try:
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
