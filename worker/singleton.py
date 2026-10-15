"""Single-instance guard: a second instance exits immediately.

The farm requires exactly one supervisor and one worker per box (a duplicate
worker races on the repo cache and double-books the GPU). Windows uses a named
mutex, Linux (the cloud pod) an flock on a file in /tmp; both are held for the
process lifetime and released by the OS on any exit, so crashes never leave a
stale lock.
"""
import os
import sys

ERROR_ALREADY_EXISTS = 183
_held = []  # keeps the Linux lock file open for the process lifetime


def ensure_single_instance(name):
    """Exit the process if another instance already holds the lock."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.CreateMutexW(None, False, f"Global\\render-farm-{name}")
        if ctypes.windll.kernel32.GetLastError() != ERROR_ALREADY_EXISTS:
            return
    else:
        import fcntl
        f = open(f"/tmp/render-farm-{name}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _held.append(f)
            return
        except OSError:
            pass
    print(f"another render-farm {name} instance is already running — exiting")
    sys.exit(0)
