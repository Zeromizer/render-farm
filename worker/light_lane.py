"""Light lane: CPU-side engines run on their own thread, beside the render loop,
so a 30 s hyperframes job no longer waits behind a 20 min H3 video_gen.

Off unless LIGHT_LANE=1 in the .env. When it is on, the two lanes split the
engines between them with no overlap:
  light lane   LIGHT_ENGINES, up to LIGHT_LANE_SLOTS jobs, engine_limit() of each
  render loop  everything else (GPU / ComfyUI / TTS engines, python, blender,
               and any engine added later until it is listed here)
Both claim with a conditional update (pending -> processing), atomic per row,
like worker/fetch_lane.py; the loop stops using claim_farm_job() so it never
takes a light job. Stale reclaim is unchanged: a job whose lane died goes back
to pending and is taken again.

Admission: while the render loop is running a job, a light job starts only if
the box has room beside it, so it cannot push that job into the 2026-09-14
commit-limit wedge (WinError 1455): available RAM, commit headroom and free
VRAM all above the LIGHT_LANE_MIN_* settings. While the loop is idle a light
job always starts, exactly as it would have in the single loop: the box idles
at ~14.5 GB available (WSL, browser, an idle ComfyUI), so gating the idle case
on 16 GB starved every light job (2026-10-05, three planar_patch/hyperframes
jobs sat pending for 18 min).

Pause during video_gen: measured 2026-10-05 (768p 5 s t2v, job 3b475951), H3
pins available RAM at ~2 GB and free VRAM at ~0.5 GB from ~30 s in to the end,
so no light job can share the box with it. The RAM reading alone is not enough:
in H3's first ~30 s it still looks free. So the lane also never starts a job
while the render loop runs a PAUSE_DURING engine, and the loop waits for a
running light job to finish before it starts one (wait_for_light()).

Pause while ComfyUI works: PAUSE_DURING names engines, so a job of another
engine that drives H3 through ComfyUI itself slipped past it. On 2026-10-18 a
`python` spike (job 62e36c25, spikes/h3_orbit) ran H3 for 70 min while this lane
claimed a dealer remotion render beside it (86d4f1dc). The box hit 30.8 GB
with ComfyUI at 16.8 GB, and the render failed ("No frame found at position
8192"); the same plate rendered fine 4 min later. So while the render loop has
a job, the lane also waits whenever ComfyUI has a prompt running or queued,
whoever queued it (_comfy_busy()). Only while the loop has a job: with the loop
idle, ComfyUI's queue cannot be ours, and a prompt stuck there must not starve
the lane. No answer from ComfyUI counts as not busy.

Slots (2026-10-31, the PC went to 64 GB): each slot is a thread claiming as the
single lane did. With another job already on the box the bars apply even when the
render loop is idle; the first job on an idle box still always starts. HyperFrames
and Remotion stay one at a time inside their runners, because HyperFrames' frames
changed under a parallel run (Kinetograph's did not), so a slot may wait there.

Shared caches (repo checkouts, venvs, assets) are serialised in worker/locks.py.
Restart: the render loop drains this lane (stop claiming, finish the current
job) before it exits on worker/restart-requested.
"""
import ctypes
import json
import os
import subprocess
import threading
import time
import urllib.request

import psutil

import config
import db

# Engines whose work is CPU (Chrome capture, ffmpeg, OpenCV) with at most a
# little GPU, and that never touch ComfyUI or pause the TTS workers.
LIGHT_ENGINES = {"hyperframes", "remotion", "reference_extract", "planar_patch",
                 "frame_extract", "video_split", "asset_check"}
POLL_SECONDS = 3
# Render-loop engines that need the whole box; see "Pause during video_gen".
PAUSE_DURING = {e.strip() for e in os.environ.get("LIGHT_LANE_PAUSE_DURING", "video_gen,lettering_fix,clip_edit,face_swap").split(",")
                if e.strip()}
# Also wait while ComfyUI has work, whatever the engine (see "Pause while ComfyUI works").
PAUSE_ON_COMFY = os.environ.get("LIGHT_LANE_PAUSE_ON_COMFY", "1") not in ("0", "false", "no")
# The engine the render loop is running now (set by render_worker.process_job).
main_engine = [None]
# What the music lane is running (set by music_lane).
music_engine = [None]

_stop = threading.Event()
_idle = threading.Event()   # no light job running, in any slot
_idle.set()
_admit = threading.Lock()
_running = {}               # engine -> jobs running now, under _admit
_pending_lock = threading.Lock()
_pending_seen = [-POLL_SECONDS, False]   # when last asked, and the answer
_waiting_logged = [None]   # the reason last logged, so each new one is logged once


def may_recycle(lane):
    """Whether a job that just ended in this lane may restart an idle ComfyUI. A python
    spike in the render loop drives ComfyUI with no lock, and between its prompts the
    queue looks empty, so only the loop itself may decide while it has a job."""
    return lane == "main" or main_engine[0] is None


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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001 - no reading: don't block on it
        return None


def _comfy_busy():
    """True while ComfyUI has a prompt running or queued. False when it does not
    answer (not started, restarting): then nothing of it is on the box."""
    try:
        with urllib.request.urlopen(config.COMFYUI_URL.rstrip("/") + "/queue", timeout=2) as r:
            q = json.loads(r.read())
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:  # noqa: BLE001
        return False


def room(others_running=False):
    """(ok, reason): whether the box has room to start a light job now.
    others_running: another light job or a bed is on the box, so the bars apply
    even with the render loop idle."""
    if main_engine[0] is None and not others_running:
        return True, ""
    if main_engine[0] in PAUSE_DURING:
        return False, f"render loop is running {main_engine[0]}"
    # A bed of the music lane is what the lane may run beside, so its prompt does not count.
    if main_engine[0] is not None and PAUSE_ON_COMFY and music_engine[0] is None and _comfy_busy():
        return False, f"ComfyUI busy (render loop is running {main_engine[0]})"
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
    """Is a light job waiting? Asked once per poll however many slots ask."""
    with _pending_lock:
        if time.monotonic() - _pending_seen[0] >= POLL_SECONDS:
            rows = (db.sb.table("farm_render_jobs").select("id")
                    .in_("engine", sorted(LIGHT_ENGINES)).eq("status", "pending")
                    .eq("cancel_requested", False).limit(1).execute().data)
            _pending_seen[:] = [time.monotonic(), bool(rows)]
        return _pending_seen[1]


def engine_limit(engine):
    """Jobs of one engine at once. 1 unless it has been shown safe: two of an engine
    was never exercised before slots, and one busy engine must not fill them all
    (reference_extract is the lane's biggest user, 13 hours in 14 days)."""
    if engine == "hyperframes" and config.KINETOGRAPH_CLI and config.KINETOGRAPH_ROUTE == "on":
        return config.KINETOGRAPH_PARALLEL
    return 1


def _open_engines():
    return {e for e in LIGHT_ENGINES if _running.get(e, 0) < engine_limit(e)}


def _busy():
    return any(_running.values())


def _leave(engine):
    with _admit:
        _running[engine] -= 1
        if not _busy():
            _idle.set()


def _claim_next():
    """(job or None, wait reason or ""). Holding _admit makes the room reading, the
    engine limits and the claim one step, so two slots cannot both act on one reading."""
    with _admit:
        open_engines = _open_engines()
        if not open_engines:  # no reading of the box, no query: nothing could start
            return None, ""
        ok, why = room(_busy() or music_engine[0] is not None)
        if not ok:
            return None, why
        _idle.clear()
        job = None
        try:
            # main_engine again: the loop may have taken a pause engine after room() looked
            if not _stop.is_set() and main_engine[0] not in PAUSE_DURING:
                job = db.claim_conditional(include=open_engines)
            if job:
                _running[job["engine"]] = _running.get(job["engine"], 0) + 1
        finally:
            if not _busy():
                _idle.set()
        return job, ""


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
            job, why = _claim_next()
            if why:
                if _waiting_logged[0] != why.split(" ")[0:2]:
                    log(f"waiting for room: {why}")
                    _waiting_logged[0] = why.split(" ")[0:2]
                _stop.wait(POLL_SECONDS * 5)
                continue
            if _waiting_logged[0]:
                log("room again, claiming")
                _waiting_logged[0] = None
            err_logged = False
        except Exception as e:  # noqa: BLE001
            if not err_logged:
                log(f"claim error (retrying quietly): {str(e)[:160]}")
                err_logged = True
            _stop.wait(POLL_SECONDS * 5)
            continue
        if not job:
            _stop.wait(POLL_SECONDS)
            continue
        try:
            process_job(job, log)
        except Exception as e:  # noqa: BLE001 - process_job records failures itself
            log(f"unexpected error after {job['id']}: {str(e)[:200]}")
        finally:
            _leave(job["engine"])


def start(process_job, restart_flag, log):
    """Run the lane's slots on daemon threads; returns the first. log should mark
    its lines as the lane's."""
    threads = [threading.Thread(target=_loop, args=(process_job, restart_flag, log),
                                name=f"light-lane-{i}", daemon=True)
               for i in range(max(1, config.LIGHT_LANE_SLOTS))]
    for t in threads:
        t.start()
    return threads[0]


def is_idle():
    """True when no light job is running (always true with the lane off)."""
    return _idle.is_set()


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
