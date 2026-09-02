"""Reference video extraction: measure a clip into a motion_spec JSON.

Pure compute, no Supabase: video in, motion_spec.json + keyframes + contact
sheet out in <out_dir>. The reference_extract runner owns download/upload/DB.
Runs inside the cached venv built from extract/requirements.txt (worker/venvs.py).

CLI:
  python extract.py <video> <out_dir> --stages probe,frames,shots,audio
                    [--reference-id <uuid>] [--ocr-fps 1] [--threshold 27]

Stage contract (HANDOFF §4): stages are independent — a failure marks that
section {"status": "failed", "error": ...} and the rest still run. Only probe
is fatal (nothing downstream can run without duration/fps). Stage timings land
in spec["timings"] and as "STAGE <name> <s>" stdout lines; "PROGRESS <n>" lines
drive the farm job's progress bar.

Times in seconds, positions normalised 0-1 (§5). Wrong labels are worse than
missing ones — camera/reveal classifiers emit "unknown" freely.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


def _find_ffmpeg_tool(name):
    """Resolve ffmpeg/ffprobe without relying on the worker's PATH.

    The worker is launched by a Startup-shortcut supervisor, whose environment
    can predate any PATH edit — the first real job on the desktop failed with
    WinError 2 exactly this way. Order: FFMPEG_DIR env (set it in
    render-farm\\.env; the runner passes it through), then PATH, then the
    winget install layout both machines use.
    """
    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir:
        exe = os.path.join(env_dir, f"{name}.exe")
        if os.path.exists(exe):
            return exe
    found = shutil.which(name)
    if found:
        return found
    pattern = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", f"{name}.exe",
    )
    hits = sorted(glob.glob(pattern), reverse=True)
    if hits:
        return hits[0]
    raise RuntimeError(
        f"{name} not found: set FFMPEG_DIR in render-farm\\.env to its bin directory, "
        "or install with `winget install Gyan.FFmpeg`"
    )


FFMPEG = None
FFPROBE = None  # resolved once in main()

ALL_STAGES = ["probe", "frames", "shots", "audio", "motion", "grade", "text", "composition"]
PROGRESS_AT = {"probe": 5, "frames": 15, "shots": 35, "audio": 50, "motion": 65,
               "grade": 75, "text": 90, "composition": 95}
SAMPLE_FPS = 2.0
MAX_FRAMES = 300
BEAT_TOLERANCE_S = 0.08
SAFE_TOP = 0.12     # vertical-ad safe zone: avoid top 12% and bottom 20%
SAFE_BOTTOM = 0.80
CONTACT_COLS = 5
CONTACT_TILE_W = 320
CONTACT_MAX_TILES = 24


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def _jsonable(v):
    """numpy scalars/arrays -> plain python, recursively; floats rounded sanely."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ---------------------------------------------------------------- stages

def stage_probe(ctx):
    out = json.loads(_run([
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", ctx["video"],
    ]))
    vstream = next((s for s in out.get("streams", []) if s.get("codec_type") == "video"), None)
    astream = next((s for s in out.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not vstream:
        raise RuntimeError("no video stream")
    num, _, den = (vstream.get("r_frame_rate") or "0/1").partition("/")
    fps = (float(num) / float(den)) if float(den or 1) else 0.0
    duration = float(out.get("format", {}).get("duration") or vstream.get("duration") or 0)
    if duration <= 0 or fps <= 0:
        raise RuntimeError(f"unusable probe: duration={duration} fps={fps}")
    w, h = int(vstream["width"]), int(vstream["height"])
    ctx["duration"], ctx["fps"], ctx["w"], ctx["h"] = duration, fps, w, h
    ctx["has_audio"] = astream is not None
    return {
        "duration": duration, "fps": round(fps, 3), "width": w, "height": h,
        "aspect": _aspect_label(w, h), "has_audio": astream is not None,
        "audio_sample_rate": int(astream["sample_rate"]) if astream else None,
    }


def _aspect_label(w, h):
    ratio = w / h
    for label, r in (("9:16", 9 / 16), ("16:9", 16 / 9), ("1:1", 1.0),
                     ("4:5", 4 / 5), ("4:3", 4 / 3), ("3:4", 3 / 4)):
        if abs(ratio - r) / r < 0.03:
            return label
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def stage_frames(ctx):
    rate = min(SAMPLE_FPS, MAX_FRAMES / ctx["duration"])
    frames_dir = os.path.join(ctx["out_dir"], "frames")
    os.makedirs(frames_dir, exist_ok=True)
    _run([FFMPEG, "-y", "-v", "error", "-i", ctx["video"],
          "-vf", f"fps={rate:.6f}", "-q:v", "2",
          os.path.join(frames_dir, "f_%04d.jpg")])
    files = sorted(f for f in os.listdir(frames_dir) if f.startswith("f_"))
    if not files:
        raise RuntimeError("ffmpeg produced no frames")
    ctx["frames"] = [os.path.join(frames_dir, f) for f in files]
    # fps filter emits frame n at t = n/rate (first at 0)
    ctx["frame_times"] = [i / rate for i in range(len(files))]
    ctx["sample_rate_fps"] = rate
    return {"count": len(files), "sample_fps": round(rate, 3)}


def stage_shots(ctx):
    from scenedetect import ContentDetector, detect

    # min_scene_len must sit well under the default 15 frames: the stutter runs
    # these ads live on (four cuts in 1.6s, 0.43s spacings) are 12-13 frames
    # apart at 30fps and the default silently merges them — measured on the
    # "Hook" reference: 12 detected of 21 real cuts before, 21 after.
    scenes = detect(ctx["video"],
                    ContentDetector(threshold=ctx["threshold"], min_scene_len=4))
    if scenes:
        bounds = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    else:  # no cuts found: the whole clip is one shot
        bounds = [(0.0, ctx["duration"])]
    cuts = [round(b[0], 3) for b in bounds]           # first entry is 0.0
    durations = [round(e - s, 3) for s, e in bounds]
    ctx["shot_bounds"] = bounds

    density = [0] * max(1, math.ceil(ctx["duration"]))
    for t in cuts[1:]:
        density[min(int(t), len(density) - 1)] += 1

    transitions = _classify_transitions(ctx, cuts[1:])
    _extract_keyframes(ctx, bounds)
    _contact_sheet(ctx)

    return {
        "count": len(bounds),
        "cuts": cuts,
        "durations": durations,
        "mean_len": round(float(np.mean(durations)), 3),
        "median_len": round(float(np.median(durations)), 3),
        "longest_hold": round(float(np.max(durations)), 3),
        "cut_density": density,
        "transitions": transitions,
    }


def _classify_transitions(ctx, cut_times):
    """hard vs dissolve vs whip per cut, via native-fps frame diffs around it.

    hard: the inter-frame difference is one dominant spike. dissolve: elevated
    diffs spread over 3-6 frames. whip/flash: very large motion or a luminance
    spike in the neighbourhood. Anything ambiguous is "unknown" (§4.3).
    """
    cap = cv2.VideoCapture(ctx["video"])
    labels = []
    try:
        for t in cut_times:
            try:
                labels.append(_transition_at(cap, t, ctx["fps"]))
            except Exception:
                labels.append("unknown")
    finally:
        cap.release()
    return labels


def _transition_at(cap, t, fps):
    n = 6  # frames each side
    start = max(0.0, t - n / fps)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    grays, lumas = [], []
    for _ in range(2 * n):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        grays.append(g.astype(np.float32))
        lumas.append(float(g.mean()))
    if len(grays) < 4:
        return "unknown"
    diffs = np.array([float(np.abs(grays[i + 1] - grays[i]).mean())
                      for i in range(len(grays) - 1)])
    peak = diffs.max()
    if peak < 8:  # nothing much happened where the detector said a cut is
        return "unknown"
    big = diffs > peak * 0.5
    span = int(big.sum())
    luma_jump = max(lumas) - min(lumas)
    if span >= 3:
        return "whip" if (peak > 60 or luma_jump > 120) else "dissolve"
    return "hard"


def _extract_keyframes(ctx, bounds):
    cap = cv2.VideoCapture(ctx["video"])
    keyframes = []
    try:
        for i, (s, e) in enumerate(bounds):
            mid = (s + e) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            path = os.path.join(ctx["out_dir"], f"kf_{i:03d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            keyframes.append({"index": i, "t": round(mid, 3), "path": path})
    finally:
        cap.release()
    ctx["keyframes"] = keyframes


def _contact_sheet(ctx):
    from PIL import Image, ImageDraw

    kfs = ctx.get("keyframes") or []
    if not kfs:
        return
    if len(kfs) > CONTACT_MAX_TILES:  # subsample evenly
        idx = np.linspace(0, len(kfs) - 1, CONTACT_MAX_TILES).round().astype(int)
        kfs = [kfs[i] for i in idx]
    tiles = []
    for kf in kfs:
        im = Image.open(kf["path"])
        th = round(im.height * CONTACT_TILE_W / im.width)
        tiles.append((im.resize((CONTACT_TILE_W, th)), kf["t"]))
    th = tiles[0][0].height
    cols = min(CONTACT_COLS, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * CONTACT_TILE_W, rows * th), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (tile, t) in enumerate(tiles):
        x, y = (i % cols) * CONTACT_TILE_W, (i // cols) * th
        sheet.paste(tile, (x, y))
        label = f"{t:.1f}s"
        draw.rectangle([x + 4, y + 4, x + 10 + 7 * len(label), y + 18], fill=(0, 0, 0))
        draw.text((x + 7, y + 6), label, fill=(255, 255, 255))
    path = os.path.join(ctx["out_dir"], "contact.jpg")
    sheet.save(path, quality=80)
    ctx["contact_sheet"] = path


def stage_audio(ctx):
    if not ctx.get("has_audio"):
        return {"status_note": "no audio stream", "bpm": None, "beats": [], "onsets": [],
                "drops": [], "segments": [{"t0": 0, "t1": ctx["duration"], "kind": "silence"}],
                "cuts_on_beat": None, "audio_style": "silent"}
    import librosa

    wav = os.path.join(ctx["out_dir"], "audio.wav")
    _run([FFMPEG, "-y", "-v", "error", "-i", ctx["video"],
          "-ac", "1", "-ar", "22050", "-vn", wav])
    y, sr = librosa.load(wav, sr=22050, mono=True)

    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    while bpm and bpm < 60:
        bpm *= 2
    while bpm and bpm >= 180:
        bpm /= 2

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    env_times = librosa.times_like(onset_env, sr=sr)

    # drops: strongest onset peaks, >=1.5s apart, top 5
    order = np.argsort(onset_env)[::-1]
    drops = []
    for i in order:
        t = float(env_times[i])
        if all(abs(t - d) >= 1.5 for d in drops):
            drops.append(t)
        if len(drops) >= 5:
            break
    drops.sort()

    segments, music_cover, speech_cover = _audio_segments(y, sr, ctx["duration"])

    cuts = [b[0] for b in (ctx.get("shot_bounds") or [])[1:]]
    cuts_on_beat = None
    if cuts and len(beat_times):
        on = sum(1 for c in cuts if np.min(np.abs(beat_times - c)) <= BEAT_TOLERANCE_S)
        cuts_on_beat = {"count": on, "ratio": round(on / len(cuts), 3),
                        "tolerance_ms": int(BEAT_TOLERANCE_S * 1000)}

    onset_rate = len(onset_times) / max(ctx["duration"], 0.1)
    if music_cover > 0.5:
        style = "music_driven"
    elif speech_cover > 0.5:
        style = "speech_driven"
    elif music_cover < 0.15 and onset_rate < 1.5:
        style = "sfx_over_silence"
    else:
        style = "mixed"

    return {
        "bpm": round(bpm, 1) if bpm else None,
        "beats": [round(float(t), 3) for t in beat_times],
        "onsets": [round(float(t), 3) for t in onset_times],
        "drops": [round(d, 2) for d in drops],
        "segments": segments,
        "cuts_on_beat": cuts_on_beat,
        "audio_style": style,
    }


def _audio_segments(y, sr, duration):
    """music / speech / silence per 0.5s window via energy + spectral flatness
    (§4.4: a heuristic is enough for v1 — no speech model)."""
    import librosa

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    t = librosa.times_like(rms, sr=sr, hop_length=hop)
    win = 0.5
    n_win = max(1, math.ceil(duration / win))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    floor = np.percentile(rms_db, 95) - 35  # silence = well below the loud parts

    kinds = []
    for i in range(n_win):
        m = (t >= i * win) & (t < (i + 1) * win)
        if not m.any():
            kinds.append("silence")
            continue
        if np.median(rms_db[m]) < floor:
            kinds.append("silence")
        elif np.median(flat[m]) < 0.02:   # strongly tonal, sustained -> music
            kinds.append("music")
        else:
            kinds.append("speech")
    # smooth single-window islands into their left neighbour
    for i in range(1, n_win - 1):
        if kinds[i] != kinds[i - 1] and kinds[i] != kinds[i + 1]:
            kinds[i] = kinds[i - 1]

    segments = []
    for i, k in enumerate(kinds):
        t0 = round(i * win, 2)
        t1 = round(min((i + 1) * win, duration), 2)
        if segments and segments[-1]["kind"] == k:
            segments[-1]["t1"] = t1
        else:
            segments.append({"t0": t0, "t1": t1, "kind": k})
    total = max(duration, 0.1)
    music = sum(s["t1"] - s["t0"] for s in segments if s["kind"] == "music") / total
    speech = sum(s["t1"] - s["t0"] for s in segments if s["kind"] == "speech") / total
    return segments, music, speech


def _motion_frames(video, t0, t1, target_fps, native_fps):
    """Grayscale 320px-wide frames over [t0, t1) at ~target_fps, read
    sequentially (seek once, then step through native frames — a per-frame
    POS_MSEC seek is both slow and keyframe-inaccurate)."""
    step = max(1, round(native_fps / target_fps))
    cap = cv2.VideoCapture(video)
    frames, times = [], []
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        n = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = t0 + n / native_fps
            if t >= t1:
                break
            if n % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h = round(g.shape[0] * 320 / g.shape[1])
                frames.append(cv2.resize(g, (320, h)))
                times.append(t)
            n += 1
    finally:
        cap.release()
    return frames, times


def _flow_stats(frames, times):
    """Per-frame-pair Farneback stats: (t, magnitude, dx, dy, divergence, rotation)."""
    h, w = frames[0].shape
    cy, cx = h / 2, w / 2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = xs - cx, ys - cy
    rnorm = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    rx, ry = rx / rnorm, ry / rnorm  # unit radial field

    out = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mag = float(np.sqrt(fx ** 2 + fy ** 2).mean())
        div = float((fx * rx + fy * ry).mean())          # + = outward = push_in
        rot = float((fx * -ry + fy * rx).mean())          # tangential = rotation
        out.append((times[i], mag, float(fx.mean()), float(fy.mean()), div, rot))
    return out


MOTION_FPS = 10.0       # motion's own sampling rate (amendment to §4.5) —
RESAMPLE_FPS = 15.0     # decoupled from the global 2fps that colour/OCR/
                        # composition keep; hot shots re-sample at 15.


def stage_motion(ctx):
    base_fps = float(ctx.get("motion_fps") or MOTION_FPS)
    frames, times = _motion_frames(ctx["video"], 0.0, ctx["duration"],
                                   base_fps, ctx["fps"])
    if len(frames) < 2:
        raise RuntimeError("could not read frames for motion")
    w = frames[0].shape[1]
    per_frame = _flow_stats(frames, times)

    # Clip-median per-frame magnitude is the yardstick for "hot": a shot whose
    # PEAK exceeds 2x it gets a second pass at RESAMPLE_FPS, so whips and speed
    # ramps that live between 10fps samples are actually captured.
    clip_median = float(np.median([r[1] for r in per_frame])) or 1e-6

    bounds = ctx.get("shot_bounds") or [(0.0, ctx["duration"])]
    per_shot = []
    resampled = []
    handheld = 0
    for si, (s, e) in enumerate(bounds):
        rows = [r for r in per_frame if s <= r[0] < e]
        if not rows:
            per_shot.append({"shot": si, "label": "unknown", "confidence": 0.0, "magnitude": 0.0})
            continue

        flow_curve = None
        peak = max(r[1] for r in rows)
        if peak > 2 * clip_median and base_fps < RESAMPLE_FPS:
            rframes, rtimes = _motion_frames(ctx["video"], s, e, RESAMPLE_FPS, ctx["fps"])
            if len(rframes) >= 2:
                rows = _flow_stats(rframes, rtimes)
                flow_curve = [round(r[1], 3) for r in rows]
                resampled.append(si)

        label, conf, mag = _camera_label(rows, w)
        if label == "handheld":
            handheld += 1
        entry = {"shot": si, "label": label,
                 "confidence": round(conf, 2), "magnitude": round(mag, 3)}
        if flow_curve is not None:
            entry["flow_curve"] = flow_curve
            entry["sample_fps"] = RESAMPLE_FPS
        per_shot.append(entry)

    # motion energy per second, normalised by p95 (§4.5: used later to check
    # "does the output move when the reference moves")
    n_sec = max(1, math.ceil(ctx["duration"]))
    curve = np.zeros(n_sec)
    counts = np.zeros(n_sec)
    for t, mag, *_ in per_frame:
        b = min(int(t), n_sec - 1)
        curve[b] += mag
        counts[b] += 1
    curve = curve / np.maximum(counts, 1)
    p95 = np.percentile(curve, 95) or 1.0
    curve = np.clip(curve / p95, 0, 1)

    return {
        "sample_fps": base_fps,
        "resampled_shots": resampled,
        "per_shot": per_shot,
        "energy_curve": [round(float(v), 3) for v in curve],
        "handheld_ratio": round(handheld / len(bounds), 2),
    }


def _has_speed_ramp(mags, frame_w):
    """A sustained monotonic run of flow magnitude covering >3x, at real speed.

    Smooth (window 3), split at direction reversals, then accept any segment of
    >=8 samples (~0.5s at 15fps — filters the sub-second text slides that
    false-positived as ramps) whose fast end is real motion and whose slow end,
    floored at a noise epsilon, sits >3x below it. A ramp decelerating into a
    hold legitimately bottoms out near zero, so the low end is floored, never
    gated.
    """
    if len(mags) < 10:
        return False
    smoothed = np.convolve(np.asarray(mags, dtype=float), np.ones(3) / 3, mode="valid")
    if len(smoothed) < 8:
        return False
    steps = np.diff(smoothed)
    seg_start = 0
    direction = 0
    segments = []
    for i, s in enumerate(steps):
        d = 1 if s > 0 else (-1 if s < 0 else direction)
        if direction == 0:
            direction = d
        elif d != direction:
            segments.append((seg_start, i))
            seg_start = i
            direction = d
    segments.append((seg_start, len(smoothed) - 1))

    floor = frame_w * 0.001
    for a, b in segments:
        if b - a + 1 < 8:
            continue
        seg = smoothed[a : b + 1]
        hi, lo = float(seg.max()), float(seg.min())
        if hi > frame_w * 0.008 and hi / max(lo, floor) > 3:
            return True
    return False


def _camera_label(rows, frame_w):
    """CAM-family label from per-frame flow stats. Wrong labels are worse than
    missing ones, so thresholds are conservative and 'unknown' is the fallback.
    Pan/tilt labels name the CAMERA move: scene flow right = camera pans left."""
    mags = np.array([r[1] for r in rows])
    dxs = np.array([r[2] for r in rows])
    dys = np.array([r[3] for r in rows])
    divs = np.array([r[4] for r in rows])
    mag = float(mags.mean())
    if mag < frame_w * 0.004:  # ~1.3px on 320w
        return "static", 0.9, mag
    if len(rows) >= 2 and mags.max() > 4 * max(mags.mean(), 1e-6) and mags.max() > frame_w * 0.06:
        return "whip", 0.6, mag
    # speed_ramp (§4.5 amendment): magnitude accelerating or decelerating
    # monotonically by >3x with no cut. Detected per sustained monotonic
    # SEGMENT rather than whole-shot, because the real reference clip turned
    # out to be four consecutive decel ramps inside one detected shot (each
    # 3.4→0.27-style, separated by jump-backs the cut detector missed) — a
    # whole-shot test scores that 0.33 monotonic and misses it, while 5-sample
    # micro-shots trivially pass one. Checked before the direction split — a
    # ramped push-in reads as the ramp, the more actionable label.
    if _has_speed_ramp(mags, frame_w):
        return "speed_ramp", 0.7, mag
    adx, ady, adiv = abs(dxs.mean()), abs(dys.mean()), abs(divs.mean())
    total = adx + ady + adiv + 1e-6
    # translation/zoom means cancel out under shake; strong magnitude with no
    # dominant direction is the handheld signature
    if max(adx, ady, adiv) < mag * 0.25:
        return ("handheld", 0.5, mag) if mag > frame_w * 0.008 else ("unknown", 0.3, mag)
    if adiv >= adx and adiv >= ady:
        conf = min(0.9, adiv / total + 0.2)
        return ("push_in" if divs.mean() > 0 else "pull_out"), conf, mag
    if adx >= ady:
        conf = min(0.9, adx / total + 0.2)
        return ("pan_l" if dxs.mean() > 0 else "pan_r"), conf, mag
    conf = min(0.9, ady / total + 0.2)
    return ("tilt_u" if dys.mean() > 0 else "tilt_d"), conf, mag


def stage_grade(ctx):
    from sklearn.cluster import KMeans

    kfs = ctx.get("keyframes") or []
    bounds = ctx.get("shot_bounds") or []
    if not kfs:
        raise RuntimeError("needs keyframes from the shots stage")
    durations = {i: (e - s) for i, (s, e) in enumerate(bounds)} if bounds else {}
    total_dur = sum(durations.values()) or len(kfs)

    pix, lumas, sats, a_means, b_means = [], [], [], [], []
    for kf in kfs:
        im = cv2.imread(kf["path"])
        im = cv2.resize(im, (160, round(im.shape[0] * 160 / im.shape[1])))
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # sample pixels proportional to the shot's duration weight
        weight = durations.get(kf["index"], 1.0) / total_dur
        n = max(200, int(8000 * weight))
        idx = np.random.default_rng(kf["index"]).choice(len(lab), size=min(n, len(lab)), replace=False)
        pix.append(lab[idx])
        lumas.append(lab[:, 0].mean() / 255)
        sats.append(hsv[..., 1].mean() / 255)
        a_means.append(lab[:, 1].mean() - 128)
        b_means.append(lab[:, 2].mean() - 128)

    allpix = np.concatenate(pix)
    km = KMeans(n_clusters=min(5, len(allpix)), n_init=4, random_state=0).fit(allpix)
    counts = np.bincount(km.labels_, minlength=km.n_clusters).astype(float)
    weights = counts / counts.sum()
    palette = []
    for center, wgt in sorted(zip(km.cluster_centers_, weights), key=lambda p: -p[1]):
        lab_px = np.clip(center, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        b, g, r = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0, 0]
        palette.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "weight": round(float(wgt), 3)})

    all_l = allpix[:, 0] / 255
    warm = float(np.mean(a_means) + np.mean(b_means))
    return {
        "palette": palette,
        "luma_mean": round(float(np.mean(lumas)), 3),
        "contrast": round(float(np.percentile(all_l, 95) - np.percentile(all_l, 5)), 3),
        "saturation": round(float(np.mean(sats)), 3),
        "temperature": "warm" if warm > 4 else ("cool" if warm < -4 else "neutral"),
        "black_point": round(float(np.percentile(all_l, 1)), 3),
        "white_point": round(float(np.percentile(all_l, 99)), 3),
    }


def stage_text(ctx):
    import easyocr

    frames = ctx.get("frames") or []
    times = ctx.get("frame_times") or []
    if not frames:
        raise RuntimeError("needs the frames stage")
    if ctx["ocr_fps"] and ctx["ocr_fps"] < ctx["sample_rate_fps"]:
        step = max(1, round(ctx["sample_rate_fps"] / ctx["ocr_fps"]))
        frames, times = frames[::step], times[::step]

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    tracks = []  # {text, t0, t1, boxes:[bbox], texts:[(t,text)], confs:[..]}
    W, H = ctx["w"], ctx["h"]
    for path, t in zip(frames, times):
        try:
            dets = reader.readtext(path, paragraph=False)
        except Exception:
            continue
        for quad, text, conf in dets:
            if conf < 0.35 or not text.strip():
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bbox = (min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H)
            _track_text(tracks, text.strip(), bbox, float(conf), t)

    blocks = [_finish_track(tr) for tr in tracks
              if (tr["t1"] - tr["t0"]) >= 0.4 or len(tr["texts"]) >= 2]
    inside = [b for b in blocks if SAFE_TOP <= (b["bbox"][1] + b["bbox"][3]) / 2 <= SAFE_BOTTOM]
    wps_peak = _words_per_second_peak(blocks, ctx["duration"])
    return {
        "blocks": blocks,
        "safe_zone_ratio": round(len(inside) / len(blocks), 2) if blocks else None,
        "words_per_second_peak": wps_peak,
    }


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(area, 1e-9)


def _track_text(tracks, text, bbox, conf, t):
    import difflib

    best, score = None, 0.0
    for tr in tracks:
        if t - tr["t1"] > 1.5:  # track went dark, leave it closed
            continue
        sim = difflib.SequenceMatcher(None, tr["texts"][-1][1].lower(), text.lower()).ratio()
        prefixy = tr["texts"][-1][1].lower() in text.lower() or text.lower() in tr["texts"][-1][1].lower()
        s = _iou(tr["boxes"][-1], bbox) + (0.6 if (sim > 0.6 or prefixy) else 0)
        if s > score:
            best, score = tr, s
    if best is not None and score >= 0.5:
        best["t1"] = t
        best["boxes"].append(bbox)
        best["texts"].append((t, text))
        best["confs"].append(conf)
    else:
        tracks.append({"t0": t, "t1": t, "boxes": [bbox], "texts": [(t, text)], "confs": [conf]})


def _finish_track(tr):
    boxes = np.array(tr["boxes"])
    bbox = [round(float(v), 3) for v in
            (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())]
    final = max(tr["texts"], key=lambda p: len(p[1]))[1]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zone = f"{_third(cy, 'top', 'centre', 'bottom')}-{_third(cx, 'left', 'centre', 'right')}"
    return {
        "text": final,
        "t0": round(tr["t0"], 2), "t1": round(tr["t1"], 2),
        "bbox": bbox,
        "size_ratio": round(float((boxes[:, 3] - boxes[:, 1]).mean()), 3),
        "zone": zone,
        "reveal": _reveal_style(tr),
        "case": "upper" if final.isupper() else ("lower" if final.islower() else "mixed"),
        "weight": "bold" if (boxes[:, 3] - boxes[:, 1]).mean() > 0.055 else "regular",
    }


def _third(v, lo, mid, hi):
    return lo if v < 1 / 3 else (mid if v < 2 / 3 else hi)


def _reveal_style(tr):
    """block vs word_by_word vs char_by_char from how the tracked text grew.
    karaoke (colour change on a subset) needs pixel history we don't keep — that
    guess stays 'unknown' here rather than wrong (§4.7)."""
    texts = [t for _, t in tr["texts"]]
    if len(texts) < 2:
        return "block"
    words = [len(t.split()) for t in texts]
    chars = [len(t) for t in texts]
    grew_words = words[-1] > words[0]
    grew_chars = chars[-1] > chars[0]
    monotonic = all(b >= a for a, b in zip(chars, chars[1:]))
    if grew_words and monotonic:
        return "word_by_word"
    if grew_chars and not grew_words and monotonic:
        return "char_by_char"
    if not grew_chars:
        return "block"
    return "unknown"


def _words_per_second_peak(blocks, duration):
    if not blocks:
        return None
    n_sec = max(1, math.ceil(duration))
    per_sec = np.zeros(n_sec)
    for b in blocks:
        words = len(b["text"].split())
        span = max(b["t1"] - b["t0"], 0.5)
        for s in range(int(b["t0"]), min(int(b["t1"]) + 1, n_sec)):
            per_sec[s] += words / span
    return round(float(per_sec.max()), 1)


def stage_composition(ctx):
    kfs = ctx.get("keyframes") or []
    if not kfs:
        raise RuntimeError("needs keyframes from the shots stage")
    out = []
    for kf in kfs:
        im = cv2.imread(kf["path"], cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (320, round(im.shape[0] * 320 / im.shape[1])))
        # saliency proxy (§4.8, v1 minimal): largest high-contrast region
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(im, (5, 5), 0), cv2.CV_32F))
        lap = cv2.GaussianBlur(lap, (31, 31), 0)
        thresh = np.percentile(lap, 92)
        mask = (lap >= thresh).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        m = cv2.moments(max(contours, key=cv2.contourArea))
        if not m["m00"]:
            continue
        cx = m["m10"] / m["m00"] / im.shape[1]
        cy = m["m01"] / m["m00"] / im.shape[0]
        on_thirds = any(abs(v - t) < 0.08 for v in (cx, cy) for t in (1 / 3, 2 / 3))
        out.append({"t": kf["t"], "subject_centre": [round(cx, 3), round(cy, 3)],
                    "on_thirds": on_thirds, "path": kf["path"]})
    return {"keyframes": out, "contact_sheet": ctx.get("contact_sheet")}


STAGE_FNS = {
    "probe": stage_probe, "frames": stage_frames, "shots": stage_shots,
    "audio": stage_audio, "motion": stage_motion, "grade": stage_grade,
    "text": stage_text, "composition": stage_composition,
}
SECTION_FOR = {"probe": "probe", "shots": "shots", "audio": "audio",
               "motion": "motion", "grade": "grade", "text": "text",
               "composition": "composition"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--reference-id", default=None)
    ap.add_argument("--threshold", type=float, default=27.0)
    ap.add_argument("--ocr-fps", type=float, default=None)
    ap.add_argument("--motion-fps", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global FFMPEG, FFPROBE
    FFMPEG = _find_ffmpeg_tool("ffmpeg")
    FFPROBE = _find_ffmpeg_tool("ffprobe")
    requested = [s for s in args.stages.split(",") if s.strip()]
    stages = [s for s in ALL_STAGES if s in requested]  # canonical order
    for s in requested:
        if s not in ALL_STAGES:
            print(f"WARN unknown stage ignored: {s}")

    ctx = {"video": args.video, "out_dir": args.out_dir,
           "threshold": args.threshold, "ocr_fps": args.ocr_fps,
           "motion_fps": args.motion_fps}
    spec = {
        "spec_version": "1.0",
        "reference_id": args.reference_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "timings": {},
    }
    # probe always runs (duration/fps feed everything). Shots feed audio's
    # cuts_on_beat, motion's per-shot grouping, grade's duration weights and
    # composition's keyframes; the 2fps frames set now feeds ONLY text — the
    # motion stage samples the video itself at its own rate (§4.5 amendment).
    wanted = set(stages) | {"probe"}
    if wanted & {"audio", "motion", "grade", "composition"}:
        wanted |= {"shots"}
    if "text" in wanted:
        wanted |= {"frames"}
    stages = [s for s in ALL_STAGES if s in wanted]

    for s in stages:
        t0 = time.monotonic()
        try:
            result = STAGE_FNS[s](ctx)
            section = {"status": "ok", **(result or {})}
        except Exception as e:
            if s == "probe":  # fatal: nothing downstream can run
                print(f"STAGE probe failed: {e}")
                raise
            section = {"status": "failed", "error": str(e)[:300]}
            spec["status"] = "partial"
        dt = round(time.monotonic() - t0, 2)
        spec["timings"][s] = dt
        if s != "frames":  # frames is plumbing, not a schema section
            spec[SECTION_FOR[s]] = section
        print(f"STAGE {s} {dt}s {section['status']}")
        print(f"PROGRESS {PROGRESS_AT[s]}")

    if ctx.get("contact_sheet"):
        spec["contact_sheet"] = ctx["contact_sheet"]

    out_path = os.path.join(args.out_dir, "motion_spec.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(spec), f, ensure_ascii=False, indent=1)
    print("PROGRESS 100")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    sys.exit(main())
