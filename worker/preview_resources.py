"""Fail-closed resource admission for an optional CPU snapshot worker."""
import ctypes
import os

_previous = None


def can_start():
    global _previous
    if os.name != "nt":
        return False  # Only the measured Windows host is supported initially.
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        return False
    idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return False
    current = (idle.value, kernel.value + user.value)
    previous, _previous = _previous, current
    if previous is None or current[1] <= previous[1]:
        return False
    busy = 100 * (1 - (current[0] - previous[0]) / (current[1] - previous[1]))
    return memory.available >= max(8, float(os.environ.get("PREVIEW_MIN_FREE_GB", "8"))) * 1024**3 and busy < min(70, float(os.environ.get("PREVIEW_MAX_CPU_PERCENT", "70")))
