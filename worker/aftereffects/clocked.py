"""A child-process runner whose cancel/timeout clock does not depend on the
child talking. proc.run_streaming only looks at the clock when a stdout line
arrives; aerender is silent for stretches and ffmpeg at -loglevel error says
nothing at all, so a cancel or timeout could go unnoticed until the next
phase (measured 2026-09-11). Here a reader thread feeds a queue and the loop
polls every 0.5 s regardless; cancel_check runs every 5 s.
"""
import os
import queue
import subprocess
import sys
import threading
import time

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

import proc  # noqa: E402


def run_clocked(cmd, cwd, on_line, cancel_check, timeout_seconds, pid_sink=None, env=None):
    """Run cmd; returns (returncode, tail_lines). Raises proc.Canceled / proc.TimedOut
    (after killing the process tree). Nonzero exit is the caller's to judge."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
    if pid_sink:
        pid_sink(p.pid, os.path.basename(cmd[0]))
    lines = queue.Queue()

    def reader():
        try:
            for line in p.stdout:
                lines.put(line.rstrip("\r\n"))
        finally:
            lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    tail = []
    started = time.monotonic()
    last_cancel = started
    eof = False
    try:
        while True:
            try:
                line = lines.get(timeout=0.5)
                if line is None:
                    eof = True
                else:
                    tail.append(line)
                    if len(tail) > 60:
                        tail.pop(0)
                    on_line(line)
                    continue  # drain what is queued before looking at the clock
            except queue.Empty:
                pass
            now = time.monotonic()
            if now - started > timeout_seconds:
                proc._kill_tree(p)
                raise proc.TimedOut(f"exceeded {timeout_seconds:.0f}s")
            if now - last_cancel >= 5:
                last_cancel = now
                if cancel_check():
                    proc._kill_tree(p)
                    raise proc.Canceled()
            if eof and p.poll() is not None:
                return p.returncode, tail
    finally:
        if p.poll() is None:
            proc._kill_tree(p)
