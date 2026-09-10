"""Runtime estimates for video_gen jobs, so the platform can show an ETA
instead of a spinner. Calibrated on the RTX 4080 SUPER 2026-09-06:

  H3 sampling      ~2.5e-7 s per pixel-frame per step
                   (1344x768 x 243 f -> 62 s/step; 832x480 x 73 f -> 7 s/step)
  model load       ~25 s when the checkpoint is cold (--fast-disk streaming),
                   ~5 s when the same family ran last
  decode + save    ~1.2e-7 s per pixel-frame (768p 10 s -> ~30 s)
  SeedVR2          ~55 s per second of video at 1080p (3 s segment ~165 s)
  lanczos          ~10 s
  h3 latent upscale  PROVISIONAL (not yet measured on the PC): model load ~60 s,
                   3D upscaler ~0.02 s per megapixel-frame, then the refine tail
                   (~3 steps of the source profile) at the generation cost per
                   pixel-frame per step, per tile (x1.35 for overlap/context),
                   two decodes, ~40 s fidelity check
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
# h3_latent_upscale (provisional; calibrate on the 4080 SUPER, see docs/h3-latent-upscale-pc-handoff.md)
H3UP_LOAD_S = 60.0
H3UP_UPSCALER_S_PER_MPXF = 0.02
H3UP_STEP_S_PER_PXF = STEP_S_PER_PXF
H3UP_TILE_OVERHEAD = 1.35
H3UP_DECODE_S_PER_PXF = DECODE_S_PER_PXF * 2
FIDELITY_S = 40.0
H3UP_REFINE_STEPS = 3          # denoise 0.375 of an 8-step turbo profile
H3UP_DEFAULT_SOURCE = (480, 832, 124)   # 9:16 480p 5 s when the source is unknown


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


def latent_upscale_plan(u, src_w, src_h, frames):
    """{refine, tiles, steps, step_seconds, load_seconds, upscaler_seconds, decode_seconds, total}
    for one h3_latent_upscale pass. Pure; never raises (falls back to defaults)."""
    from videogen import graphs_h3
    u = dict(u or {})
    try:
        v = graphs_h3.validate_upscale_params(dict(u, method=graphs_h3.METHOD), "upscale")
        dims = graphs_h3.target_dims(src_w, src_h, v)
    except ValueError:
        v = dict(u, tile_width=640, tile_height=384, tile_overlap=64, variant=u.get("variant") or "tile",
                 steps_override=int(u.get("steps_override") or 0))
        dims = {"refine": (int(src_w) * 2 // 32 * 32, int(src_h) * 2 // 32 * 32)}
    w, h = dims["refine"]
    frames = int(frames)
    steps = int(v.get("steps_override") or 0) or H3UP_REFINE_STEPS
    if v.get("steps_override"):
        steps = max(1, int(round(int(v["steps_override"]) * (float(v.get("denoise") or 0.375) or 0.375))))
    variant = v.get("variant") or "tile"
    if variant == "full":
        tiles = 1
        pxf = w * h * frames
        step_s = H3UP_STEP_S_PER_PXF * pxf
    else:
        tiles = graphs_h3.tile_count((w, h), v)
        pxf = int(v["tile_width"]) * int(v["tile_height"]) * frames
        step_s = H3UP_STEP_S_PER_PXF * pxf * H3UP_TILE_OVERHEAD
    upscaler_s = H3UP_UPSCALER_S_PER_MPXF * (w * h * frames / 1e6)
    decode_s = H3UP_DECODE_S_PER_PXF * w * h * frames
    total = H3UP_LOAD_S + upscaler_s + tiles * steps * step_s + decode_s + FIDELITY_S
    return {"refine": [w, h], "tiles": tiles, "steps": steps, "step_seconds": step_s, "load_seconds": H3UP_LOAD_S,
            "upscaler_seconds": upscaler_s, "decode_seconds": decode_s, "total": total}


def latent_upscale_seconds(u, src_w, src_h, frames):
    return latent_upscale_plan(u, src_w, src_h, frames)["total"]


def latent_upscale_hint(u, src_w, src_h, frames):
    """The {steps, step_seconds, load_seconds, tail_seconds} dict comfy_client.wait
    consumes. With tiles the tqdm bar restarts per tile, so steps is the total
    over tiles and step_seconds the per-step cost of one tile."""
    p = latent_upscale_plan(u, src_w, src_h, frames)
    return {"steps": p["steps"] * p["tiles"], "step_seconds": p["step_seconds"],
            "load_seconds": p["load_seconds"] + p["upscaler_seconds"], "tail_seconds": p["decode_seconds"]}


def upscale_seconds(u, seconds_of_video, src_dims=None):
    if not u:
        return 0.0
    method = (u.get("method") or "lanczos")
    if method == "lanczos":
        return LANCZOS_S
    if method == "h3_latent_upscale":
        w, h = src_dims or H3UP_DEFAULT_SOURCE[:2]
        frames = graphs.frames_for(max(1, min(15, float(seconds_of_video) or 5)))
        return latent_upscale_seconds(u, w, h, frames)
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
        return upscale_seconds(vg.get("upscale") or {}, 5 if (vg.get("upscale") or {}).get("method") == "h3_latent_upscale" else 10)
    secs = generation_seconds(vg, mode)
    if vg.get("upscale"):
        secs += upscale_seconds(vg["upscale"], vg.get("duration_s", 5), src_dims=_dims(vg))
    return secs


def fmt_eta(seconds):
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} s"
    m = int(round(s / 60))
    return f"{m} min"
