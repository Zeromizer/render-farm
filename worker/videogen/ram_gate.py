"""Hold an H3 job until the box has enough free RAM to run it without thrashing.

WHY: this PC has 32 GB and H3 runs with --fast-disk, so the model weights live
in the Windows page cache. When something else takes that RAM (2026-09-25: the
rp-agent container running four parallel plate renders, 7.3 GB), the weights
page in from disk on every step: the text encoder took 41 min to load and the
sampler ran at ~300 s/step instead of a few seconds. The job still "runs", so
nothing fails and the queue simply stops moving. Better to wait for the RAM
up front and say so in the job's phase than to start and crawl.

"Available" = Windows' available physical memory (free + standby), which is
what the page cache can grow into. After max_wait the job starts anyway (with
a log line), so a machine that is simply short of RAM can't stall the queue for good.
"""
import ctypes
import time

import proc


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                          "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                          "ullAvailExtendedVirtual")]


def available_gb():
    m = _MemoryStatusEx()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return m.ullAvailPhys / (1 << 30)


def wait_for_ram(need_gb, max_wait_s, cancel_check, on_wait, log,
                 read=available_gb, sleep=time.sleep, clock=time.monotonic, poll_s=10, settle=2):
    """Block until `settle` consecutive readings are >= need_gb, or max_wait_s passes.

    Returns the seconds spent waiting (0.0 when RAM was already there).
    on_wait(avail_gb) is called on every short reading (for the job phase).
    Raises proc.Canceled if cancel_check() turns true while waiting.
    """
    if need_gb <= 0:
        return 0.0
    avail = read()
    if avail >= need_gb:
        return 0.0  # the common case: no wait, no settle
    start = clock()
    ok = 0
    while True:
        ok = ok + 1 if avail >= need_gb else 0
        if ok >= settle:
            break
        waited = clock() - start
        if waited >= max_wait_s:
            log(f"ram gate: still only {avail:.1f} GB available after {waited / 60:.0f} min "
                f"(want {need_gb:g} GB) - starting anyway, expect it to be slow")
            return waited
        if not ok:
            on_wait(avail)
        if cancel_check():
            raise proc.Canceled()
        sleep(poll_s)
        avail = read()
    waited = clock() - start
    log(f"ram gate: {avail:.1f} GB available after waiting {waited / 60:.1f} min")
    return waited
