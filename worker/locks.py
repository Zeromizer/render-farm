"""Named in-process locks for the caches both job lanes share (repo checkouts,
venvs, the asset cache). One job at a time never needed them; with the light
lane (worker/light_lane.py) a remotion render and a python job could otherwise
checkout two refs into the same repo dir, or build the same venv twice."""
import contextlib
import threading

_guard = threading.Lock()
_locks = {}
_limits = {}


def named(key):
    """The lock for this key (re-entrant, so a holder may call helpers that take it again)."""
    with _guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.RLock()
        return lock


def limited(key, count):
    """A semaphore for this key: at most count holders at once."""
    with _guard:
        slot = _limits.get(key)
        if slot is None:
            slot = _limits[key] = threading.BoundedSemaphore(max(1, count))
        return slot


@contextlib.contextmanager
def exclusive(key, count):
    """Every place of limited(key, count): alone among its holders. named(key) keeps two of
    these from each holding some places and waiting for the rest."""
    slot = limited(key, count)
    with named(key):
        for _ in range(max(1, count)):
            slot.acquire()
        try:
            yield
        finally:
            for _ in range(max(1, count)):
                slot.release()
