"""Streamed subprocess runner with cancellation and timeout.

Kills the whole process tree (taskkill /T /F on Windows, the process group
on Linux) so node/ffmpeg/chromium children spawned by npx or blender die with
the parent.
"""
import os
import signal
import subprocess
import time

WINDOWS = os.name == "nt"


class Canceled(Exception):
    pass


class TimedOut(Exception):
    pass


def kill_pid_tree(pid):
    if WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        return
    import psutil
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _kill_tree(proc):
    if not WINDOWS:
        # run_streaming starts each child in its own session: one signal takes
        # the group, including grandchildren that already lost their parent.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    kill_pid_tree(proc.pid)


def run_streaming(cmd, cwd, on_line, cancel_check, timeout_seconds, env=None):
    """Run cmd, calling on_line(line) per stdout line and cancel_check() every ~5s.

    Raises Canceled / TimedOut / RuntimeError(nonzero exit with output tail).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {"start_new_session": True}),
    )
    tail = []
    started = time.monotonic()
    last_cancel_check = started
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
            on_line(line)
            now = time.monotonic()
            if now - started > timeout_seconds:
                _kill_tree(proc)
                raise TimedOut(f"exceeded {timeout_seconds}s")
            if now - last_cancel_check >= 5:
                last_cancel_check = now
                if cancel_check():
                    _kill_tree(proc)
                    raise Canceled()
        rc = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            _kill_tree(proc)
    if rc != 0:
        raise RuntimeError(f"exit code {rc}: ...{chr(10).join(tail[-15:])}")
