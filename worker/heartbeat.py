"""Heartbeat thread: while a job runs (including clone/npm phases), update
heartbeat_at + progress every HEARTBEAT_SECONDS so reclaim_stale_jobs knows
the worker is alive."""
import threading

import config
import db


class Heartbeat:
    def __init__(self, job_id):
        self.job_id = job_id
        self.progress = 0          # shared int, written by stdout parsers
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(config.HEARTBEAT_SECONDS):
            try:
                db.update_job(self.job_id, {
                    "heartbeat_at": db.now_iso(),
                    "progress": int(self.progress),
                })
            except Exception:
                pass  # transient network errors must not kill the render

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
