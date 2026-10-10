"""Clip edit, pixel side: surgical MiniMax H3 edits of an existing (generated) clip.

Every mode is the same move: lay out a timeline of frames, mark what H3 may change
(a per-frame video mask and audio mask), let H3 regenerate only that, and put the
untouched pixels and sound back exactly. Modes:

  region   remove or replace what is inside boxes (tracked with the car, or held
           still for overlays) over a frame range; the prompt describes what should
           be there. Crop mode (default for small areas) generates only a window
           around the boxes that zooms with the car, like lettering_fix; full mode
           generates the whole frame.
  extend   add N seconds after the clip that continue its motion and sound
  prepend  add N seconds before it
  bridge   generate the N seconds between clip A and clip B, so one shot flows into
           the next instead of a crossfade between two renderings of the car
  audio    keep every pixel, regenerate the sound over a range (foley retake)

Same graph for all of them (videogen/graphs_inpaint.py, vlo-style latent noise mask);
only the masks differ. Two subcommands driven by a spec JSON:

  prep     build the timeline + masks, write the H3 inputs and plan.json
  compose  put each take back, score it, write every take, the best one, a proof
           sheet and a report

Run in the planar_patch venv (opencv-python-headless + numpy); ffmpeg as lettering.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import wave

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "lettering"))
import lettering as L  # noqa: E402  (io, grid, tracking: one implementation)

FPS = L.FPS
SR = 48000
MODES = ("region", "extend", "prepend", "bridge", "audio")
MAX_WINDOW = L.MAX_WINDOW
# What a 16 GB card holds in one H3 pass, in canvas pixels x frames: the follow-crop
# lettering pass (0.52 MP x 192 f) and video_gen's 768p quarters (1.03 MP x ~96 f) both
# sit near 1e8. Full-frame canvases shrink to stay under it.
PIXEL_FRAMES = 1.1e8
MAX_AREA = 1344 * 768     # H3's native canvas; never generate above it
CROP_AREA = L.CANVAS_AREA
HANDOFF = 6               # frames blended from original to generated at a seam


emit = L.emit
log = lambda s: print(s, flush=True)  # noqa: E731


# ---------------------------------------------------------------- audio (PCM16 48k stereo)

def read_audio(path, seconds):
    """The clip's audio as float32 (n, 2) at 48 kHz, padded/cut to `seconds`; silence if none."""
    n = int(round(seconds * SR))
    out = np.zeros((n, 2), np.float32)
    if not L.has_audio(path):
        return out
    ffmpeg = L.find_ffmpeg_tool("ffmpeg")
    r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-f", "s16le",
                        "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "2", "-"], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
    out[:min(n, len(a))] = a[:n]
    return out


def write_wav(path, a):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def frames_to_samples(f):
    return int(round(f / FPS * SR))


# ---------------------------------------------------------------- grid / canvas

def snap_up(n):
    """Smallest 17k+5 >= n."""
    return 5 if n <= 5 else 5 + int(math.ceil((n - 5) / 17)) * 17


def canvas_for(aspect, length, cap_area, src_area):
    """32-aligned canvas of this aspect, under the card's pixel-frame budget, H3's native
    area and the source's own area (upscaling only costs time)."""
    area = min(cap_area, MAX_AREA, PIXEL_FRAMES / max(1, length), max(src_area, 256 * 256))
    return L.gen_dims(aspect, area)


def window_around(n, s, e, ctx):
    """(lo, length) for a 17k+5 window inside [0, n) covering [s, e] plus up to ctx frames
    of context each side."""
    need = (e - s + 1) + 2 * ctx
    length = min(snap_up(need), L.snap_len(n), MAX_WINDOW)
    if length < e - s + 1:
        raise RuntimeError(f"the edit spans {e - s + 1} frames; one H3 pass holds {min(L.snap_len(n), MAX_WINDOW)} "
                           f"(~{MAX_WINDOW / FPS:.0f} s). Split it into shorter edits.")
    lo = int(round((s + e) / 2 - (length - 1) / 2))
    lo = max(0, min(lo, n - length))
    return lo, length


# ---------------------------------------------------------------- prep

def _boxes_per_frame(frames, regions, lo, hi):
    """{frame: [box, ...]} for every region over its range (tracked or held still),
    padded to at least a latent cell (masks that hug an object leave slivers)."""
    fh, fw = frames[0].shape[:2]
    grays = None
    out = {}
    for r in regions:
        b = [float(v) for v in r["box"]]
        f0 = int(r.get("frame", lo))
        s, e = int(r["start_frame"]), int(r["end_frame"])
        if r.get("track", True):
            if grays is None:
                grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            m = max(40.0, 0.6 * max(b[2] - b[0], b[3] - b[1]))
            motion = L.track_motion(grays, f0, L.clamp_box([b[0] - m, b[1] - m, b[2] + m, b[3] + m], fw, fh),
                                    max(lo, s), min(hi, e))
            tr = L.track_element(grays, motion, f0, b, max(lo, s), min(hi, e))
            # frames the track did not reach (left the frame, lost): hold the nearest box
            known = sorted(tr)
            track = {i: tr[i][:4] if i in tr else tr[min(known, key=lambda k: abs(k - i))][:4]
                     for i in range(max(lo, s), min(hi, e) + 1)}
        else:
            track = {i: b for i in range(max(lo, s), min(hi, e) + 1)}
        for i, p in track.items():
            w, h = p[2] - p[0], p[3] - p[1]
            pad = max(float(r.get("pad_px", 0)) or 0.0, 0.12 * max(w, h), 16.0)
            out.setdefault(i, []).append(L.clamp_box([p[0] - pad, p[1] - pad, p[2] + pad, p[3] + pad], fw, fh))
    return out


def _resize_to(f, w, h):
    return f if f.shape[1] == w and f.shape[0] == h else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)


def prep(spec):
    mode = spec["mode"]
    if mode not in MODES:
        raise RuntimeError(f"mode must be one of {MODES}")
    work = spec["work_dir"]
    emit("PHASE", "decoding")
    frames, fps = L.read_clip(spec["clip"])
    fh, fw = frames[0].shape[:2]
    if abs(fps - FPS) > 0.5:
        log(f"warning: clip is {fps:.2f} fps; H3 works at {FPS}, frames are taken one for one")
    n = len(frames)
    audio = read_audio(spec["clip"], n / FPS)
    warnings = []

    # timeline: list of frames (None = to generate), audio, and where the original lives
    pad = 0
    if mode in ("region", "audio"):
        # a clip off H3's 17k+5 grid (193 f) cannot be one window; pad with copies of the
        # last frame up to the grid and trim them off in compose
        if (n - 5) % 17 and snap_up(n) <= MAX_WINDOW:
            pad = snap_up(n) - n
        tl = list(frames) + [frames[-1]] * pad
        tl_audio = np.concatenate([audio, np.zeros((frames_to_samples(pad), 2), np.float32)])
        src_ranges = [(0, n)]
    elif mode in ("extend", "prepend"):
        want = max(1, int(round(float(spec["seconds"]) * FPS)))
        ctx = min(n, max(12, int(round(float(spec.get("context_s", 2.0)) * FPS))))
        length = min(snap_up(ctx + want), MAX_WINDOW)
        gen = length - ctx
        if gen < want:
            warnings.append(f"asked for {want} new frames; one pass holds {gen} after {ctx} frames of context")
        blanks = [None] * gen
        silence = np.zeros((frames_to_samples(gen), 2), np.float32)
        if mode == "extend":
            tl, tl_audio, src_ranges = list(frames) + blanks, np.concatenate([audio, silence]), [(0, n)]
            lo = n - ctx
        else:
            tl, tl_audio, src_ranges = blanks + list(frames), np.concatenate([silence, audio]), [(gen, gen + n)]
            lo = 0
    elif mode == "bridge":
        fb, _ = L.read_clip(spec["clip_b"])
        fb = [_resize_to(f, fw, fh) for f in fb]
        nb = len(fb)
        audio_b = read_audio(spec["clip_b"], nb / FPS)
        want = max(1, int(round(float(spec["seconds"]) * FPS)))
        c = max(12, int(round(float(spec.get("context_s", 1.5)) * FPS)))
        ca, cb = min(n, c), min(nb, c)
        length = min(snap_up(ca + want + cb), MAX_WINDOW)
        gen = length - ca - cb
        if gen < want:
            warnings.append(f"asked for {want} bridge frames; one pass holds {gen} with the context")
        tl = list(frames) + [None] * gen + list(fb)
        tl_audio = np.concatenate([audio, np.zeros((frames_to_samples(gen), 2), np.float32), audio_b])
        src_ranges = [(0, n), (n + gen, n + gen + nb)]
        lo = n - ca
    N = len(tl)

    # window + masks (video: list of box lists or "full"; audio: per-frame 0/1)
    vmask, amask = {}, {}
    crop_mode = "full"
    regs = None
    if mode == "region":
        regions = spec["regions"]
        if not 1 <= len(regions) <= 4:
            raise RuntimeError("region mode needs 1-4 regions")
        for r in regions:
            r.setdefault("start_frame", 0)
            r.setdefault("end_frame", n - 1)
            r["start_frame"] = max(0, int(r["start_frame"]))
            r["end_frame"] = min(n - 1, int(r["end_frame"]))
        s = min(r["start_frame"] for r in regions)
        e = max(r["end_frame"] for r in regions)
        lo, length = window_around(n + pad, s, e, int(spec.get("context_frames", 24)))
        hi = lo + length - 1
        emit("PHASE", "tracking regions")
        vmask = _boxes_per_frame(frames, regions, lo, min(hi, n - 1))
        for i in range(n, n + pad):   # padding frames hold the last frame's mask
            if n - 1 in vmask:
                vmask[i] = vmask[n - 1]
        if not vmask:
            raise RuntimeError("nothing to edit: no region box falls inside the clip")
        allb = np.array([b for bs in vmask.values() for b in bs])
        ub = [allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max()]
        big = (ub[2] - ub[0]) * (ub[3] - ub[1]) > 0.35 * fw * fh
        crop_mode = spec.get("crop", "auto")
        if crop_mode == "auto":
            crop_mode = "full" if big else "crop"
    elif mode == "audio":
        s = max(0, int(spec.get("start_frame", 0)))
        e = min(n - 1, int(spec.get("end_frame", n - 1)))
        lo, length = window_around(n + pad, s, e, int(spec.get("context_frames", 24)))
        hi = lo + length - 1
        amask = {i: 1 for i in range(s, e + 1)}
    else:
        hi = lo + length - 1
        for i in range(lo, hi + 1):
            if tl[i] is None:
                vmask[i] = "full"
                amask[i] = 1
    win = range(lo, hi + 1)
    length = hi - lo + 1
    print(f"MEASURED mode {mode} timeline {N} f, window {lo}-{hi} ({length} f), "
          f"masked video {len(vmask)} f, audio {len(amask)} f", flush=True)

    # canvas + per-frame crop regions (source coords)
    if crop_mode == "crop":
        # follow-crop: a 16:9 window on the union of the boxes, centred and sized per frame
        # (it zooms with the car, so H3 sees the area at a steady size), smoothed so H3
        # sees a steady camera
        W, Hc = canvas_for(16 / 9, length, CROP_AREA, 10 ** 9)
        cen, size = [], []
        for i in win:
            bs = vmask.get(i)
            if not bs:
                cen.append(None)
                size.append(None)
                continue
            a = np.array(bs)
            u = [a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()]
            cen.append(((u[0] + u[2]) / 2, (u[1] + u[3]) / 2))
            size.append(max((u[2] - u[0]) * 1.6, (u[3] - u[1]) * 1.6 * W / Hc, 200.0))
        known = [k for k, c in enumerate(cen) if c is not None]
        near = lambda k: min(known, key=lambda j: abs(j - k))  # noqa: E731
        cen = np.array([cen[k] if cen[k] is not None else cen[near(k)] for k in range(len(cen))], np.float32)
        size = np.array([size[k] if size[k] is not None else size[near(k)] for k in range(len(size))], np.float32)
        sm = 3
        ker = np.ones(2 * sm + 1) / (2 * sm + 1)
        smooth = lambda v: np.convolve(np.pad(v, sm, mode="edge"), ker, mode="valid")  # noqa: E731
        cen = np.stack([smooth(cen[:, d]) for d in range(2)], 1)
        size = smooth(size)
        regs = []
        for (cx, cy), sz in zip(cen, size):
            rw = min(float(sz), fw, fh * W / Hc)
            rh = rw * Hc / W
            rw_i, rh_i = int(round(rw)), int(round(rh))
            regs.append((int(round(min(max(0, cx - rw / 2), fw - rw))), int(round(min(max(0, cy - rh / 2), fh - rh))),
                         rw_i, rh_i))
    else:
        # audio retakes keep every pixel: the picture only steers the sound, so a small
        # canvas does
        W, Hc = canvas_for(fw / fh, length, 480 * 832 if mode == "audio" else MAX_AREA, fw * fh)
        regs = [(0, 0, fw, fh)] * length
    rws = [r[2] for r in regs]
    print(f"MEASURED canvas {W}x{Hc} crop={crop_mode} region {min(rws)}-{max(rws)} px wide "
          f"(x{W / max(rws):.2f}-x{W / min(rws):.2f})", flush=True)

    # generation inputs
    emit("PHASE", "writing generation inputs")
    grey = np.full((fh, fw, 3), 127, np.uint8)
    src_c, vm_c, am_c = [], [], []
    for k, i in enumerate(win):
        rx, ry, rw_, rh_ = regs[k]
        f = tl[i] if tl[i] is not None else grey
        src_c.append(cv2.resize(f[ry:ry + rh_, rx:rx + rw_], (W, Hc), interpolation=cv2.INTER_AREA))
        m = np.zeros((fh, fw), np.uint8)
        v = vmask.get(i)
        if v == "full":
            m[:] = 255
        elif v:
            for b in v:
                m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
        vm_c.append(cv2.cvtColor(cv2.resize(m[ry:ry + rh_, rx:rx + rw_], (W, Hc), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR))
        am_c.append(np.full((Hc, W, 3), 255 if amask.get(i) else 0, np.uint8))
    wav = os.path.join(work, "timeline.wav")
    write_wav(wav, tl_audio)
    L.write_clip(os.path.join(work, "gen_src.mp4"), src_c, FPS, audio_from=wav, audio_start=lo / FPS, lossless=True)
    L.write_clip(os.path.join(work, "gen_mask.mp4"), vm_c, FPS, lossless=True)
    L.write_clip(os.path.join(work, "gen_amask.mp4"), am_c, FPS, lossless=True)
    # the timeline itself (grey where H3 fills in) for compose
    L.write_clip(os.path.join(work, "timeline.mp4"), [f if f is not None else grey for f in tl], FPS,
                 audio_from=wav, lossless=True)

    plan = {"mode": mode, "fps": FPS, "frames": N, "width": fw, "height": fh, "window": [lo, hi],
            "length": length, "canvas": [W, Hc], "regs": regs, "crop": crop_mode,
            "vmask": {str(k): v for k, v in vmask.items()}, "amask": sorted(amask),
            "generated": [i for i in range(N) if tl[i] is None], "src_ranges": src_ranges, "pad_tail": pad,
            "warnings": warnings}
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
    emit("PROGRESS", "100")


# ---------------------------------------------------------------- compose

def _gen_audio(take, length):
    return read_audio(take, length / FPS)


def compose_take(work, plan, take, out_path):
    """Put one H3 take back on the timeline. Returns (frames, audio, metrics)."""
    tl, _ = L.read_clip(os.path.join(work, "timeline.mp4"))
    tl_audio = read_audio(os.path.join(work, "timeline.mp4"), len(tl) / FPS)
    gen, _ = L.read_clip(take)
    gaud = _gen_audio(take, plan["length"])
    lo, hi = plan["window"]
    fh, fw = plan["height"], plan["width"]
    mode = plan["mode"]
    vmask = {int(k): v for k, v in plan["vmask"].items()}
    out = [f.copy() for f in tl]
    feather = 8
    for k, i in enumerate(range(lo, hi + 1)):
        if k >= len(gen):
            break
        v = vmask.get(i)
        if not v:
            continue
        rx, ry, rw, rh = plan["regs"][k]
        back = cv2.resize(gen[k], (rw, rh), interpolation=cv2.INTER_CUBIC)
        if v == "full":
            out[i][ry:ry + rh, rx:rx + rw] = back
            continue
        m = np.zeros((fh, fw), np.uint8)
        for b in v:
            m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
        mf = cv2.GaussianBlur(cv2.dilate(m, np.ones((feather, feather), np.uint8)), (0, 0), feather / 2.0)
        a = (mf[ry:ry + rh, rx:rx + rw].astype(np.float32) / 255.0)[..., None]
        reg = out[i][ry:ry + rh, rx:rx + rw].astype(np.float32)
        out[i][ry:ry + rh, rx:rx + rw] = np.clip(reg * (1 - a) + back.astype(np.float32) * a, 0, 255).astype(np.uint8)

    # seams (extend / prepend / bridge): hand off from the original to H3's own
    # reconstruction of the context over a few frames, so the cut into generated
    # frames is between two frames H3 made to follow each other
    gen_set = set(plan["generated"])
    seams = []
    for i in range(lo, hi):
        if (i in gen_set) != (i + 1 in gen_set):
            seams.append(i + 0.5)
            side = range(i - HANDOFF + 1, i + 1) if i + 1 in gen_set else range(i + 1, i + 1 + HANDOFF)
            ordered = list(side) if i + 1 in gen_set else list(side)[::-1]
            for j, f in enumerate(ordered):
                if not lo <= f <= hi or f in gen_set:
                    continue
                t = (j + 1) / (HANDOFF + 1)
                g = cv2.resize(gen[f - lo], (fw, fh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                out[f] = np.clip(out[f].astype(np.float32) * (1 - t) + g * t, 0, 255).astype(np.uint8)

    # audio: generated sound where the audio mask is on (with 20 ms ramps), original elsewhere
    aud = tl_audio.copy()
    amask = set(plan["amask"])
    if amask:
        on = np.zeros(len(aud), np.float32)
        for i in amask:
            a0, a1 = frames_to_samples(i), frames_to_samples(i + 1)
            on[a0:a1] = 1.0
        ramp = int(0.02 * SR)
        on = np.convolve(on, np.ones(2 * ramp + 1) / (2 * ramp + 1), mode="same")
        w0 = frames_to_samples(lo)
        g = np.zeros_like(aud)
        g[w0:w0 + len(gaud)] = gaud[:max(0, min(len(gaud), len(aud) - w0))]
        aud = aud * (1 - on[:, None]) + g * on[:, None]

    # metrics: seam jump vs the clip's own motion, flicker inside the mask
    diffs = [float(np.abs(out[i + 1].astype(np.int16) - out[i].astype(np.int16)).mean()) for i in range(lo, hi)]
    med = float(np.median(diffs)) if diffs else 0.0
    seam_jump = max([diffs[int(s - 0.5) - lo] / max(med, 0.5) for s in seams], default=0.0)
    flick = []
    for i in range(lo, hi):
        v0, v1 = vmask.get(i), vmask.get(i + 1)
        if v0 and v1 and v0 != "full" and v1 != "full":
            b = v0[0]
            x0, y0, x1, y1 = [int(round(t)) for t in b]
            if x1 - x0 > 4 and y1 - y0 > 4:
                flick.append(float(np.abs(out[i + 1][y0:y1, x0:x1].astype(np.int16)
                                          - out[i][y0:y1, x0:x1].astype(np.int16)).mean()))
    metrics = {"seam_jump": round(seam_jump, 3), "motion_median": round(med, 3),
               "mask_flicker": round(float(np.mean(flick)), 3) if flick else 0.0}
    if plan.get("pad_tail"):
        out = out[:len(out) - plan["pad_tail"]]
        aud = aud[:frames_to_samples(len(out))]
    wav = out_path[:-4] + ".wav"
    write_wav(wav, aud)
    L.write_clip(out_path, out, FPS, audio_from=wav)
    os.remove(wav)
    return out, metrics


def compose(spec):
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    results = []
    for ti, take in enumerate(spec["takes"]):
        emit("PHASE", f"compositing take {ti + 1}")
        path = os.path.join(work, f"take{ti + 1}.mp4")
        _, m = compose_take(work, plan, take, path)
        # lower is better: a visible jump at a seam, then shimmer inside a patch
        score = -(m["seam_jump"] + 0.05 * m["mask_flicker"])
        results.append({"take": ti + 1, "seed": spec["seeds"][ti], "score": round(score, 4), **m, "path": path})
        print(f"MEASURED take {ti + 1} seed {spec['seeds'][ti]} {m}", flush=True)
        emit("PROGRESS", str(int(100 * (ti + 1) / (len(spec["takes"]) + 1))))
    best = max(results, key=lambda r: r["score"])
    shutil.copy(best["path"], spec["out"])
    emit("PHASE", "proof sheet")
    proof_sheet(spec["proof"], plan, os.path.join(work, "timeline.mp4"), [r["path"] for r in results], results, best)
    report = {"mode": plan["mode"], "best_take": best["take"],
              "takes": [{k: v for k, v in r.items() if k != "path"} for r in results],
              "window": plan["window"], "frames": plan["frames"], "generated_frames": len(plan["generated"]),
              "canvas": plan["canvas"], "crop": plan["crop"], "warnings": plan["warnings"]}
    json.dump(report, open(spec["report"], "w"), indent=1)
    emit("PROGRESS", "100")


def proof_sheet(path, plan, timeline_path, take_paths, results, best):
    """Rows: before (grey = to be generated), then every take; columns: 7 frames over the
    edited span (crop window for region edits, whole frame otherwise)."""
    tl, _ = L.read_clip(timeline_path)
    lo, hi = plan["window"]
    edited = sorted(set(int(k) for k in plan["vmask"]) | set(plan["amask"]))
    if not edited:
        edited = list(range(lo, hi + 1))
    span = [edited[0] - 6, edited[-1] + 6]
    last = plan["frames"] - plan.get("pad_tail", 0) - 1
    picks = sorted(set(int(round(t)) for t in np.linspace(max(lo, span[0]), min(hi, span[1], last), 7)))
    tile_w = 220
    rows = []
    for label, frames in [("before", tl)] + [(f"take {r['take']}" + (" (best)" if r is best else ""), L.read_clip(p)[0])
                                              for r, p in zip(results, take_paths)]:
        tiles = []
        for i in picks:
            rx, ry, rw, rh = plan["regs"][i - lo]
            # one tile shape (the canvas's) for every frame: crops change size with the car
            t = cv2.resize(frames[i][ry:ry + rh, rx:rx + rw], (tile_w, int(tile_w * plan['canvas'][1] / plan['canvas'][0])),
                           interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(t, f"{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(t)
        band = np.hstack(tiles)
        head = np.full((22, band.shape[1], 3), 30, np.uint8)
        cv2.putText(head, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([head, band]))
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("prep", "compose"):
        sys.exit("usage: clip_edit.py prep|compose <spec.json>")
    s = json.load(open(sys.argv[2]))
    prep(s) if sys.argv[1] == "prep" else compose(s)
