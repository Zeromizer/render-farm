"""Pause the two TTS workers (OmniVoice + Chatterbox) while an H3 job runs.

WHY: they hold 6-8 GB of the 4080's 16 GB permanently, and H3 at 16 GB is
already streaming weights off disk. This is the same process-matching logic
as voicelab\pause-tts-workers.bat / resume-tts-workers.bat, minus the
interactive `pause`, so it can run inside the render worker. The TTS
supervisor is stopped FIRST (otherwise it relaunches the workers under us)
and relaunched afterwards, which brings both workers back by itself.

Gated by VIDEO_GEN_PAUSE_TTS (default on). resume() runs even when nothing
was paused: relaunching an already-running supervisor is a no-op thanks to
its own singleton guard.
"""
import os
import subprocess

import config

_MARKS = ("supervisor.py", "tts_worker.py", "chatterbox_worker.py")
_PS_FLAGS = subprocess.CREATE_NO_WINDOW


def _ps(script):
    return subprocess.run(["powershell", "-NoProfile", "-Command", script],
                          capture_output=True, text=True, creationflags=_PS_FLAGS, timeout=60)


def _tts_processes():
    """[(pid, commandline)] of python processes belonging to the TTS studio."""
    r = _ps("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
            "ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }")
    out = []
    root = config.TTS_STUDIO_DIR.lower().replace("/", "\\")
    for line in r.stdout.splitlines():
        pid, _, cmd = line.partition("|")
        c = (cmd or "").lower().replace("/", "\\")
        # tts_worker.py / chatterbox_worker.py are unique to the TTS studio and
        # are launched with a bare cwd-relative name (the venv launcher stub AND
        # its real interpreter child both show it), so match them anywhere.
        # supervisor.py exists in the render farm too, so that one is scoped to
        # the TTS studio directory: the worker must never shoot itself.
        if "tts_worker.py" in c or "chatterbox_worker.py" in c or (
                "supervisor.py" in c and root in c):
            out.append((int(pid), cmd))
    return out


def pause(log):
    if not config.VIDEO_GEN_PAUSE_TTS:
        return False
    procs = _tts_processes()
    if not procs:
        log("tts workers: none running")
        return False
    # supervisor first, then workers, so nothing respawns mid-kill
    procs.sort(key=lambda p: 0 if "supervisor.py" in p[1].lower() else 1)
    for pid, cmd in procs:
        tag = next((m for m in _MARKS if m in cmd.lower()), "?")
        log(f"tts workers: stopping {tag} (pid {pid})")
        _ps(f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue")
    return True


def resume(log):
    if not config.VIDEO_GEN_PAUSE_TTS:
        return
    pyw = os.path.join(config.TTS_STUDIO_DIR, ".venv", "Scripts", "pythonw.exe")
    sup = os.path.join(config.TTS_STUDIO_DIR, "supervisor.py")
    if not (os.path.exists(pyw) and os.path.exists(sup)):
        log(f"tts workers: cannot resume, missing {pyw} or {sup}")
        return
    subprocess.Popen([pyw, sup], cwd=config.TTS_STUDIO_DIR,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                     | getattr(subprocess, "DETACHED_PROCESS", 0))
    log("tts workers: supervisor relaunched")


class paused:
    """with tts_guard.paused(log): ...  — resumes on any exit, including errors."""

    def __init__(self, log):
        self.log = log
        self.did_pause = False

    def __enter__(self):
        self.did_pause = pause(self.log)
        return self

    def __exit__(self, *exc):
        if self.did_pause:
            try:
                resume(self.log)
            except Exception as e:  # noqa: BLE001 - never turn a good video into a failure
                self.log(f"tts workers: resume failed: {e}")
        return False
