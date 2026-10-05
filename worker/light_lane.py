"""Light lane: CPU-side engines run on their own thread, beside the render loop,
so a 30 s hyperframes job no longer waits behind a 20 min H3 video_gen.

Off unless LIGHT_LANE=1 in the .env. When it is on, the two lanes split the
engines between them with no overlap:
  light lane   LIGHT_ENGINES, one job at a time
  render loop  everything else (GPU / ComfyUI / TTS engines, python, blender,
               and any engine added later until it is listed here)
Both claim with a conditional update (pending -> processing), atomic per row,
like worker/fetch_lane.py; the loop stops using claim_farm_job() so it never
takes a light job. Stale reclaim is unchanged: a job whose lane died goes back
to pending and is taken again.

Admission: a light job starts only while the box has room, so it cannot push an
H3 run into the 2026-09-14 commit-limit wedge (WinError 1455): available RAM,
commit headroom and free VRAM all above the LIGHT_LANE_MIN_* settings. When
there is no room the lane waits and the job stays pending.

Pause during video_gen: measured 2026-10-05 (768p 5 s t2v, job 3b475951), H3
pins available RAM at ~2 GB and free VRAM at ~0.5 GB from ~30 s in to the end,
so no light job can share the box with it. The RAM reading alone is not enough:
in H3's first ~30 s it still looks free. So the lane also never starts a job
while the render loop runs a PAUSE_DURING engine, and the loop waits for a
running light job to finish before it starts one (wait_for_light()).

Shared caches (repo checkouts, venvs, assets) are serialised in worker/locks.py.
Restart: the render loop drains this lane (stop claiming, finish the current
job) before it exits on worker/restart-requested.
"""
import ctypes
import os
import subprocess
import threading
import time

import psutil

import config
import db

# Engines whose work is CPU (Chrome capture, ffmpeg, OpenCV) with at most a
# little GPU, and that never touch ComfyUI or pause the TTS workers.
LIGHT_ENGINES = {"hyperframes", "remotion", "reference_extract", "planar_patch",
                 "frame_extract", "video_split", "asset_check"}
POLL_SECONDS = 3
# Render-loop engines that need the whole box; see "Pause during video_gen".
PAUSE_DURING = {e.strip() for e in os.environ.get("LIGHT_LANE_PAUSE_DURING", "video_gen").split(",")
                if e.strip()}
# The engine the render loop is running now (set by render_worker.process_job).
main_engine = [None]

_stop = threading.Event()
_idle = threading.Event()
_idle.set()
_waiting_logged = [False]


def _commit_free_gb():
    """System commit headroom (what WinError 1455 runs out of), in GB."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MS()
    ms.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPageFile / (1 << 30)


def _free_vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001 - no reading: don't block on it
        return None


def room():
    """(ok, reason): whether the box has room to start a light job now."""
    if main_engine[0] in PAUSE_DURING:
        return False, f"render loop is running {main_engine[0]}"
    avail = psutil.virtual_memory().available / (1 << 30)
    if avail < config.LIGHT_LANE_MIN_AVAIL_RAM_GB:
        return False, f"available RAM {avail:.1f} GB < {config.LIGHT_LANE_MIN_AVAIL_RAM_GB:g}"
    commit = _commit_free_gb()
    if commit < config.LIGHT_LANE_MIN_COMMIT_GB:
        return False, f"commit headroom {commit:.1f} GB < {config.LIGHT_LANE_MIN_COMMIT_GB:g}"
    vram = _free_vram_mib()
    if vram is not None and vram < config.LIGHT_LANE_MIN_FREE_VRAM_MIB:
        return False, f"free VRAM {vram} MiB < {config.LIGHT_LANE_MIN_FREE_VRAM_MIB}"
    return True, ""


def _pending_light():
    rows = (db.sb.table("farm_render_jobs").select("id")
            .in_("engine", sorted(LIGHT_ENGINES)).eq("status", "pending")
            .eq("cancel_requested", False).limit(1).execute().data)
    return bool(rows)


def _loop(process_job, restart_flag, log):
    log(f"light lane started ({', '.join(sorted(LIGHT_ENGINES))})")
    err_logged = False
    while not _stop.is_set():
        if os.path.exists(restart_flag):
            _stop.wait(POLL_SECONDS)
            continue
        try:
            # Only measure the box when there is something to start.
            if not _pending_light():
                err_logged = False
                _stop.wait(POLL_SECONDS)
                continue
            ok, why = room()
            if not ok:
                if not _waiting_logged[0]:
                    log(f"waiting for room: {why}")
                    _waiting_logged[0] = True
                _stop.wait(POLL_SECONDS * 5)
                continue
            if _waiting_logged[0]:
                log("room again, claiming")
                _waiting_logged[0] = False
            _idle.clear()
            if _stop.is_set():  # drain() started after the top-of-loop check
                _idle.set()
                break
            if main_engine[0] in PAUSE_DURING:  # the loop took one after room() looked
                _idle.set()
                continue
            job = db.claim_conditional(include=LIGHT_ENGINES)
            err_logged = False
        except Exception as e:  # noqa: BLE001
            _idle.set()
            if not err_logged:
                log(f"claim error (retrying quietly): {str(e)[:160]}")
                err_logged = True
            _stop.wait(POLL_SECONDS * 5)
            continue
        if not job:
            _idle.set()
            _stop.wait(POLL_SECONDS)
            continue
        try:
            process_job(job, log)
        except Exception as e:  # noqa: BLE001 - process_job records failures itself
            log(f"unexpected error after {job['id']}: {str(e)[:200]}")
        finally:
            _idle.set()


def start(process_job, restart_flag, log):
    """Run the lane on a daemon thread. log should mark its lines as the lane's."""
    t = threading.Thread(target=_loop, args=(process_job, restart_flag, log),
                         name="light-lane", daemon=True)
    t.start()
    return t


def wait_for_light(log, max_wait_s=240):
    """Called by the render loop before a PAUSE_DURING job, after it set
    main_engine (so no new light job starts): let the running one finish.
    Bounded below the 5 min stale reclaim, since the job's heartbeat has not
    started yet; past it the heavy job starts anyway."""
    if not _idle.is_set():
        log(f"waiting up to {max_wait_s}s for the running light job to finish")
        if not _idle.wait(max_wait_s):
            log("light job still running; starting anyway")


def drain(log, max_wait_s=None):
    """Stop claiming and wait for the current light job to finish."""
    _stop.set()
    if not _idle.is_set():
        log("light lane: waiting for its current job before exiting")
    _idle.wait(max_wait_s)
