"""Runtime estimates for video_gen jobs, so the platform can show an ETA
instead of a spinner. Calibrated on the RTX 4080 SUPER 2026-09-06:

  H3 sampling      ~2.5e-7 s per pixel-frame per step
                   (1344x768 x 243 f -> 62 s/step; 832x480 x 73 f -> 7 s/step)
  model load       ~25 s when the checkpoint is cold (--fast-disk streaming),
                   ~5 s when the same family ran last
  decode + save    ~1.2e-7 s per pixel-frame (768p 10 s -> ~30 s)
  SeedVR2          ~55 s per second of video at 1080p (3 s segment ~165 s)
  lanczos          ~10 s
  turntable        one generation per segment (2 halves or 4 quarters, minus reused
                   ones) + ~90 s of post; retries/repairs add a segment each

Everything here is an estimate for progress text; the runner reports real
step counts from ComfyUI's log once sampling starts.
"""
from videogen import graphs

STEP_S_PER_PXF = 2.5e-7
DECODE_S_PER_PXF = 1.2e-7
LOAD_S = 25.0
SEEDVR2_S_PER_VIDEO_S = 55.0
LANCZOS_S = 10.0
TURNTABLE_POST_S = 90.0


def _dims(vg):
    try:
        return graphs.dims(vg.get("resolution") or "480p", vg.get("ratio") or "16:9")
    except ValueError:
        return graphs.dims("480p", "16:9")


def _steps(vg, mode):
    if vg.get("steps"):
        return int(vg["steps"])
    fam = "ref2va" if mode == "r2v" else "fl2va"
    if vg.get("turbo", True) is False:
        return graphs.DEFAULT_FULL_STEPS
    return graphs.DEFAULT_TURBO_STEPS[fam]


def sampling_hint(vg, mode="i2v"):
    """{steps, step_seconds, load_seconds, tail_seconds} for one H3 generation."""
    w, h = _dims(vg)
    frames = graphs.frames_for(vg.get("duration_s", 5))
    pxf = w * h * frames
    return {"steps": _steps(vg, mode), "step_seconds": STEP_S_PER_PXF * pxf, "load_seconds": LOAD_S,
            "tail_seconds": DECODE_S_PER_PXF * pxf}


def generation_seconds(vg, mode="i2v"):
    h = sampling_hint(vg, mode)
    return h["load_seconds"] + h["steps"] * h["step_seconds"] + h["tail_seconds"]


def upscale_seconds(u, seconds_of_video):
    if not u:
        return 0.0
    if (u.get("method") or "lanczos") == "lanczos":
        return LANCZOS_S
    return SEEDVR2_S_PER_VIDEO_S * float(seconds_of_video) + 30.0


def job_seconds(params):
    """Whole-job estimate from params.video_gen (any mode)."""
    vg = (params or {}).get("video_gen") or {}
    mode = vg.get("mode")
    if not mode:
        mode = ("turntable" if vg.get("turntable") else "upscale" if vg.get("source")
                else "i2v" if vg.get("first_frame") or vg.get("last_frame")
                else "r2v" if vg.get("ref_images") or vg.get("ref_videos") or vg.get("ref_audios") else "t2v")
    if mode == "turntable":
        tt = vg.get("turntable") or {}
        try:
            from studio import turntable as flow
            pl = flow.plan(tt)
            n, secs = len(pl["generate"]), pl["seconds"]
        except Exception:  # noqa: BLE001 - an invalid job fails fast in the runner; estimate something
            n, secs = 2, tt.get("seconds_per_half", 10)
        seg = {"duration_s": secs, "resolution": tt.get("resolution", "768p"), "ratio": tt.get("ratio", "16:9")}
        return n * generation_seconds(seg, "i2v") + TURNTABLE_POST_S
    if mode == "upscale":
        return upscale_seconds(vg.get("upscale") or {}, 10)
    secs = generation_seconds(vg, mode)
    if vg.get("upscale"):
        secs += upscale_seconds(vg["upscale"], vg.get("duration_s", 5))
    return secs


def fmt_eta(seconds):
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} s"
    m = int(round(s / 60))
    return f"{m} min"
