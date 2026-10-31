"""Music lane: audio_gen runs on its own thread, so a bed no longer waits behind a
python job or a blender render it has nothing to do with (14 days to 2026-10-31:
median wait 149 s, one job in ten over 90 min).

Off unless MUSIC_LANE=1 in the .env, which also needs LIGHT_LANE=1: the render
loop only claims by engine list when the lanes are on. One job at a time, because
ComfyUI's queue is serial; a second bed in parallel would need a second ComfyUI.

ComfyUI is shared, so nothing else may use it meanwhile. render_worker.run_job
holds locks.named("comfyui") around every GPU_EXCLUSIVE_ENGINES runner, which is
what makes this safe: audio_gen swaps comfy_client.sampling_progress for the whole
process, /free drops whatever is loaded, and tts_guard is not re-entrant. That list
also has matte, for its VRAM. This lane stays off while the render loop runs one of
them, and while anything else has ComfyUI busy (a python spike drives it over HTTP,
unseen by the lock). Music still yields to video the way it did: a PC H3 job waits
at most for the bed in progress.

python is the reason for the lane, so it is not on that list: every long wait for a
bed in those 14 days was behind one.

Admission reads the box as light_lane does. 265 beds peaked at 6.05 GB over the
baseline, and the bar is set so one fits. TTS auto-start has been off since
2026-09-23, so nothing frees VRAM when the job pauses it.

Restart: the render loop drains this lane too before it exits.
"""
import os
import threading

import psutil

import config
import db
import light_lane

ENGINES = {"audio_gen"}
POLL_SECONDS = 3

_stop = threading.Event()
_idle = threading.Event()
_idle.set()
_waiting_logged = [None]


def room():
    """(ok, reason): whether a bed may start now."""
    engine = light_lane.main_engine[0]
    if engine in config.GPU_EXCLUSIVE_ENGINES or engine in light_lane.PAUSE_DURING:
        return False, f"render loop is running {engine}"
    if light_lane._comfy_busy():
        return False, "ComfyUI busy"
    avail = psutil.virtual_memory().available / (1 << 30)
    if avail < config.MUSIC_LANE_MIN_AVAIL_RAM_GB:
        return False, f"available RAM {avail:.1f} GB < {config.MUSIC_LANE_MIN_AVAIL_RAM_GB:g}"
    commit = light_lane._commit_free_gb()
    if commit < config.LIGHT_LANE_MIN_COMMIT_GB:
        return False, f"commit headroom {commit:.1f} GB < {config.LIGHT_LANE_MIN_COMMIT_GB:g}"
    vram = light_lane._free_vram_mib()
    if vram is not None and vram < config.MUSIC_LANE_MIN_FREE_VRAM_MIB:
        return False, f"free VRAM {vram} MiB < {config.MUSIC_LANE_MIN_FREE_VRAM_MIB}"
    return True, ""


def _pending():
    rows = (db.sb.table("farm_render_jobs").select("id").in_("engine", sorted(ENGINES))
            .eq("status", "pending").eq("cancel_requested", False).limit(1).execute().data)
    return bool(rows)


def _loop(process_job, restart_flag, log):
    log(f"music lane started ({', '.join(sorted(ENGINES))})")
    err_logged = False
    while not _stop.is_set():
        if os.path.exists(restart_flag):
            _stop.wait(POLL_SECONDS)
            continue
        try:
            if not _pending():
                err_logged = False
                _stop.wait(POLL_SECONDS)
                continue
            ok, why = room()
            if not ok:
                if _waiting_logged[0] != why.split(" ")[0:2]:
                    log(f"waiting for room: {why}")
                    _waiting_logged[0] = why.split(" ")[0:2]
                _stop.wait(POLL_SECONDS * 5)
                continue
            if _waiting_logged[0]:
                log("room again, claiming")
                _waiting_logged[0] = None
            _idle.clear()
            if _stop.is_set():
                _idle.set()
                break
            # Set before the claim, so the light lane already counts this job.
            light_lane.music_engine[0] = "audio_gen"
            include = ENGINES - db.gated_engines()
            job = db.claim_conditional(include=include) if include else None
            err_logged = False
        except Exception as e:  # noqa: BLE001
            light_lane.music_engine[0] = None
            _idle.set()
            if not err_logged:
                log(f"claim error (retrying quietly): {str(e)[:160]}")
                err_logged = True
            _stop.wait(POLL_SECONDS * 5)
            continue
        if not job:
            light_lane.music_engine[0] = None
            _idle.set()
            _stop.wait(POLL_SECONDS)
            continue
        try:
            process_job(job, log)
        except Exception as e:  # noqa: BLE001 - process_job records failures itself
            log(f"unexpected error after {job['id']}: {str(e)[:200]}")
        finally:
            light_lane.music_engine[0] = None
            _idle.set()


def start(process_job, restart_flag, log):
    """Run the lane on a daemon thread. log should mark its lines as the lane's."""
    t = threading.Thread(target=_loop, args=(process_job, restart_flag, log), name="music-lane", daemon=True)
    t.start()
    return t


def is_idle():
    return _idle.is_set()


def drain(log, max_wait_s=None):
    """Stop claiming and wait for the current bed to finish."""
    _stop.set()
    if not _idle.is_set():
        log("music lane: waiting for its current job before exiting")
    _idle.wait(max_wait_s)
