"""Windows named-mutex singleton guard: a second instance exits immediately.

The farm requires exactly one supervisor and one worker per PC (a duplicate
worker races on the repo cache and double-books the GPU). The mutex is held
for the process lifetime and released by the OS on any exit, so crashes never
leave a stale lock.
"""
import ctypes
import sys

ERROR_ALREADY_EXISTS = 183


def ensure_single_instance(name):
    """Exit the process if another instance already holds the named mutex."""
    ctypes.windll.kernel32.CreateMutexW(None, False, f"Global\\render-farm-{name}")
    if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        print(f"another render-farm {name} instance is already running — exiting")
        sys.exit(0)
