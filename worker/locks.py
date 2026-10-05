"""Named in-process locks for the caches both job lanes share (repo checkouts,
venvs, the asset cache). One job at a time never needed them; with the light
lane (worker/light_lane.py) a remotion render and a python job could otherwise
checkout two refs into the same repo dir, or build the same venv twice."""
import threading

_guard = threading.Lock()
_locks = {}


def named(key):
    """The lock for this key (re-entrant, so a holder may call helpers that take it again)."""
    with _guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.RLock()
        return lock
