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
LATENT_CLIP = 17          # H3's VAE clip: its first frame is a latent of its own (anchors)


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
        if r.get("keys"):
            # keyframed: the box at a few times, linear in between, held past the ends.
            # What an editor would do; predictable where tracking an object at the frame
            # edge on a moving camera is not (it latches onto the hero car)
            keys = r["keys"]
            track = {}
            for i in range(max(lo, s), min(hi, e) + 1):
                if i <= keys[0]["frame"]:
                    bx = keys[0]["box"]
                elif i >= keys[-1]["frame"]:
                    bx = keys[-1]["box"]
                else:
                    j = next(j for j in range(1, len(keys)) if keys[j]["frame"] >= i)
                    k0, k1 = keys[j - 1], keys[j]
                    t = (i - k0["frame"]) / max(1, k1["frame"] - k0["frame"])
                    bx = [a + (c - a) * t for a, c in zip(k0["box"], k1["box"])]
                track[i] = L.clamp_box([float(v) for v in bx], fw, fh)
        elif r.get("track", True):
            if grays is None:
                grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            # features on the object itself (small margin): a wide search area pulls in the
            # hero car beside it, whose motion then wins
            m = max(8.0, 0.15 * min(b[2] - b[0], b[3] - b[1]))
            motion = L.track_motion(grays, f0, L.clamp_box([b[0] - m, b[1] - m, b[2] + m, b[3] + m], fw, fh),
                                    max(lo, s), min(hi, e))
            # the box rides the scene's motion and is clipped at the frame edge, not
            # stopped there: things to remove are often half out of frame (a parked car at
            # the side of an orbiting shot). Where the track ends (the object left the
            # frame), the box keeps its last velocity, so it slides out instead of parking
            # over whatever is there.
            q = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2)
            raw = {i: L.box_of(cv2.perspectiveTransform(q, T)) for i, T in motion.items()}
            known = sorted(raw)
            first, last = known[0], known[-1]

            def vel(ks):
                ks = [k for k in ks if k in raw]
                if len(ks) < 2:
                    return np.zeros(4)
                return (np.array(raw[ks[-1]]) - np.array(raw[ks[0]])) / (ks[-1] - ks[0])
            v_head = vel(list(range(first, min(last, first + 5) + 1)))
            v_tail = vel(list(range(max(first, last - 5), last + 1)))
            track = {}
            for i in range(max(lo, s), min(hi, e) + 1):
                if i in raw:
                    bx = raw[i]
                elif i < first:
                    bx = list(np.array(raw[first]) + v_head * (i - first))
                elif i > last:
                    bx = list(np.array(raw[last]) + v_tail * (i - last))
                else:
                    bx = raw[min(known, key=lambda k: abs(k - i))]
                track[i] = L.clamp_box(bx, fw, fh)
        else:
            track = {i: b for i in range(max(lo, s), min(hi, e) + 1)}
        for i, p in track.items():
            w, h = p[2] - p[0], p[3] - p[1]
            if w < 2 or h < 2:   # wholly out of frame here
                continue
            # about a latent cell (~32 canvas px) past the object, from its SHORT side and
            # capped: a margin from the long side turned a full-width graphics band into a
            # mask over the whole car (2026-10-10, the car warped)
            pad = float(r["pad_px"]) if r.get("pad_px") else min(max(0.1 * min(w, h), 16.0), 40.0)
            out.setdefault(i, []).append(L.clamp_box([p[0] - pad, p[1] - pad, p[2] + pad, p[3] + pad], fw, fh))
    return out


def _box_mask(boxes, fh, fw, grow=0):
    m = np.zeros((fh, fw), np.uint8)
    for b in boxes:
        m[max(0, int(b[1]) - grow):int(math.ceil(b[3])) + grow, max(0, int(b[0]) - grow):int(math.ceil(b[2])) + grow] = 255
    return m


def _cover(img, w, h):
    """Scale and centre-crop an image to exactly w x h (an image model may return another shape)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, int(round(iw * s))), max(h, int(round(ih * s)))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def _align_to(img, ref, keep, scale=0.5):
    """Warp img (a cleaned copy of ref, possibly redrawn whole by an image model) onto ref,
    fitted on the pixels outside the edit (keep = 255). Identity if the fit is implausible."""
    fh, fw = ref.shape[:2]
    a = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale).astype(np.float32)
    b = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale).astype(np.float32)
    m = cv2.resize(keep, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)
    warp = np.eye(3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(b, a, warp, cv2.MOTION_HOMOGRAPHY,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), m, 5)
    except cv2.error:
        return img, False
    S = np.diag([scale, scale, 1.0])
    full = np.linalg.inv(S) @ warp.astype(np.float64) @ S
    corners = np.array([[0, 0, 1], [fw, 0, 1], [0, fh, 1], [fw, fh, 1]], np.float64).T
    moved = full @ corners
    moved = moved[:2] / moved[2]
    if np.abs(moved - corners[:2]).max() > 0.08 * max(fw, fh):
        return img, False
    return cv2.warpPerspective(img, full, (fw, fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE), True


def _match_colour(clean, ref, mask):
    """Match the cleaned patch's level and contrast to the original in a ring around the edit
    (an image model often shifts the grade a little)."""
    ring = (cv2.dilate(mask, np.ones((61, 61), np.uint8)) > 0) & ~(cv2.dilate(mask, np.ones((11, 11), np.uint8)) > 0)
    if ring.sum() < 200:
        return clean
    c, r = clean[ring].astype(np.float32), ref[ring].astype(np.float32)
    gain = np.clip(r.std(0) / np.maximum(c.std(0), 1.0), 0.8, 1.25)
    out = (clean.astype(np.float32) - c.mean(0)) * gain + r.mean(0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _bg_motion(frames, a, targets, vmask, scale=0.5):
    """Homography taking frame a to each target frame, from corners tracked outside the edit,
    chained frame to frame (so a cleaned anchor can ride the camera move)."""
    fh, fw = frames[0].shape[:2]
    S = np.diag([scale, scale, 1.0])
    Si = np.linalg.inv(S)
    gray = lambda i: cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)  # noqa: E731

    def bg(i):
        v = vmask.get(i)
        m = np.full((fh, fw), 255, np.uint8)
        if v and v != "full":
            m[_box_mask(v, fh, fw, grow=20) > 0] = 0
        return cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    out = {a: np.eye(3)}
    want = set(targets)
    for step in (1, -1):
        far = max([t for t in want if (t - a) * step > 0], key=lambda t: abs(t - a), default=None)
        if far is None:
            continue
        acc, prev, pg = np.eye(3), a, gray(a)
        for i in range(a + step, far + step, step):
            cg = gray(i)
            hs = np.eye(3)
            pts = cv2.goodFeaturesToTrack(pg, 400, 0.01, 8, mask=bg(prev))
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, pts, None)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    h, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
                    if h is not None:
                        hs = h
            acc = hs @ acc
            out[i] = Si @ acc @ S
            prev, pg = i, cg
    return out


def _anchors(spec, frames, vmask, lo, hi, warnings):
    """Cleaned frames the agent supplies (an image model's version of a frame with the object
    gone) pasted into the masked area on their frames. Those frames go to H3 unmasked, so it
    copies the emptiness instead of redrawing what the scene implies; anchor_every spreads
    them along the camera move to more frames. Returns {frame: (full-res frame, mask H3
    still fills on it)}.

    Anchors only count on KEY frames: H3's VAE encodes 17-frame clips whose first frame
    is a latent frame of its own, while the other 16 share latents four at a time, and a
    latent is masked if any of its frames is (max pooling). A lone unmasked frame anywhere
    else is swallowed by its masked neighbours, so a cleaned frame is carried (by the
    background's motion) to the key frames - window start + 17k - and only those are
    kept: the nearest one for each supplied anchor, or every round(anchor_every / 17)
    clips across the boxed span."""
    n = len(frames)
    fh, fw = frames[0].shape[:2]
    src = {}
    for an in spec.get("anchors") or []:
        a = int(round(float(an["at_s"]) * FPS)) if an.get("at_s") is not None else int(an.get("frame", 0))
        a = min(max(a, lo), min(hi, n - 1))
        v = vmask.get(a)
        if not v or v == "full":
            warnings.append(f"anchor at frame {a} skipped: no region box on that frame")
            continue
        img = cv2.imread(an["image"], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"anchor image for frame {a} is not a readable image")
        img = _cover(img, fw, fh)
        m = _box_mask(v, fh, fw)
        img, ok = _align_to(img, frames[a], 255 - cv2.dilate(m, np.ones((41, 41), np.uint8)))
        if not ok:
            warnings.append(f"anchor at frame {a}: could not line the cleaned frame up with the clip; used as is")
        src[a] = _match_colour(img, frames[a], m)
    keys = [i for i in range(lo, min(hi, n - 1) + 1) if (i - lo) % LATENT_CLIP == 0
            and vmask.get(i) and vmask[i] != "full"]
    every = int(spec.get("anchor_every") or 0)
    if not src or not keys:
        if src:
            warnings.append("anchors skipped: no key frame (window start + 17k) has a region box")
        return {}
    if every > 0:
        step = max(1, int(round(every / LATENT_CLIP)))
        targets = keys[::step]
    else:
        targets = sorted({min(keys, key=lambda k: abs(k - a)) for a in src})
    # where the box grows or shrinks between key frames (the object entering or leaving)
    # H3 invents the most: also anchor the middle 4-frame latent of that clip (all four
    # frames, or max pooling masks it again)
    area = lambda i: float(_box_mask(vmask[i], fh, fw).sum())  # noqa: E731
    for k in list(targets):
        k2 = k + LATENT_CLIP
        group = list(range(k + 9, k + 13))
        if k2 in vmask and vmask[k2] != "full" and all(vmask.get(g) and vmask[g] != "full" for g in group)                 and abs(area(k2) - area(k)) > 0.15 * max(area(k), area(k2)) and group[-1] < n:
            targets.extend(group)
    targets = sorted(set(targets))
    # carry only what was cleaned: outside its box the source frame is plain footage, and the
    # background's motion does not move the subject (it dragged the hero car's own front into
    # the box). Each target takes the source whose cleaned box covers most of its own box.
    motion = {a: _bg_motion(frames, a, targets, vmask) for a in src}
    pasted = {}
    for t in targets:
        box_t = _box_mask(vmask[t], fh, fw)
        best = None
        for a in src:
            cleaned = _box_mask(vmask[a], fh, fw)
            if t == a:
                cand = (src[a], cleaned)
            else:
                Hm = motion[a][t]
                cand = (cv2.warpPerspective(src[a], Hm, (fw, fh), borderMode=cv2.BORDER_REPLICATE),
                        cv2.warpPerspective(cleaned, Hm, (fw, fh), flags=cv2.INTER_NEAREST))
            cov = int(cv2.countNonZero(cv2.bitwise_and(cand[1], box_t)))
            if best is None or cov > best[0]:
                best = (cov, cand)
        pasted[t] = best[1]
    out = {}
    for a, (img, valid) in pasted.items():
        box = _box_mask(vmask[a], fh, fw)
        # H3 still draws a band inside the box edge and anything the warp could not cover,
        # so the cleaned area blends into the footage instead of sitting in it as a patch
        bs = np.array(vmask[a], np.float32)
        band = int(min(max(0.12 * float(np.min(np.minimum(bs[:, 2] - bs[:, 0], bs[:, 3] - bs[:, 1]))), 12), 40))
        keep = cv2.erode(box, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
        keep &= cv2.erode(valid, np.ones((9, 9), np.uint8))
        if keep.sum() < 0.25 * box.sum():
            warnings.append(f"anchor at frame {a} dropped: the cleaned frame covers too little of the box there")
            continue
        alpha = (cv2.GaussianBlur(keep, (0, 0), 3.0).astype(np.float32) / 255.0)[..., None]
        out[a] = (np.clip(frames[a].astype(np.float32) * (1 - alpha) + img.astype(np.float32) * alpha,
                          0, 255).astype(np.uint8), cv2.bitwise_and(box, cv2.bitwise_not(keep)))
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
    anchors = {}
    crop_mode = "full"
    regs = None
    if mode == "region":
        regions = spec["regions"]
        if not 1 <= len(regions) <= 4:
            raise RuntimeError("region mode needs 1-4 regions")
        for r in regions:
            # agents see downscaled frames: boxes may come as fractions of the frame and
            # times as seconds
            if r.get("box_norm") is not None:
                b = r["box_norm"]
                r["box"] = [b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh]
            for k in r.get("keys") or []:
                if k.get("box_norm") is not None:
                    b = k["box_norm"]
                    k["box"] = [b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh]
                if k.get("at_s") is not None:
                    k["frame"] = int(round(float(k["at_s"]) * FPS))
                if "box" not in k or "frame" not in k:
                    raise RuntimeError("each key needs box (or box_norm) and frame (or at_s)")
            if r.get("keys"):
                r["keys"] = sorted(r["keys"], key=lambda k: k["frame"])
                r.setdefault("box", r["keys"][0]["box"])
                r.setdefault("start_frame", r["keys"][0]["frame"])
                r.setdefault("end_frame", r["keys"][-1]["frame"])
            for key, fkey in (("at_s", "frame"), ("start_s", "start_frame"), ("end_s", "end_frame")):
                if r.get(key) is not None:
                    r[fkey] = int(round(float(r[key]) * FPS))
            if "box" not in r:
                raise RuntimeError("each region needs box (pixels), box_norm (0-1 of the frame) or keys")
            r["frame"] = min(max(0, int(r.get("frame", 0))), n - 1)
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
        anchors = _anchors(spec, frames, vmask, lo, hi, warnings)
        for a, (f, _) in anchors.items():
            tl[a] = f
    elif mode == "audio":
        for key, fkey in (("start_s", "start_frame"), ("end_s", "end_frame")):
            if spec.get(key) is not None:
                spec[fkey] = int(round(float(spec[key]) * FPS))
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
        unions = {}
        for i in win:
            bs = vmask.get(i)
            if bs:
                a = np.array(bs)
                unions[i] = [a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()]
        # the window takes the area's shape (a tall area in a portrait clip gets a portrait
        # canvas), within H3-friendly 1:2 .. 2:1 and the frame's own shape
        asp = float(np.median([(u[2] - u[0]) / max(1.0, u[3] - u[1]) for u in unions.values()]))
        asp = min(max(asp, 0.5, min(1.0, fw / fh) * 0.5), 2.0)
        W, Hc = canvas_for(asp, length, CROP_AREA, 10 ** 9)
        cen, size = [], []
        for i in win:
            u = unions.get(i)
            if u is None:
                cen.append(None)
                size.append(None)
                continue
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
        # anchor frames carry the cleaned area as given: H3 keeps it and fills only the band
        # around it (and the frames in between)
        v = None if i in anchors else vmask.get(i)
        if i in anchors:
            m = anchors[i][1].copy()
        elif v == "full":
            m[:] = 255
        elif v:
            for b in v:
                m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
        vm_c.append(cv2.cvtColor(cv2.resize(m[ry:ry + rh_, rx:rx + rw_], (W, Hc), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR))
        am_c.append(np.full((Hc, W, 3), 255 if amask.get(i) else 0, np.uint8))
    cv2.imwrite(os.path.join(work, "edge_first.png"), src_c[0])
    cv2.imwrite(os.path.join(work, "edge_last.png"), src_c[-1])
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
            "anchor_frames": sorted(anchors), "feather": 16 if anchors else 8, "warnings": warnings}
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
    emit("PROGRESS", "100")


# ---------------------------------------------------------------- compose

def _gen_audio(take, length):
    return read_audio(take, length / FPS)


def motion_residuals(frames, lo, hi, scale=0.25):
    """Per frame pair in [lo, hi]: what is left after warping frame i+1 onto frame i with
    optical flow (mean abs grey level, quarter resolution). A fast camera move warps away;
    a cut, a dissolve between two different shots or a lighting pop does not. Measured on
    real takes 2026-10-10: smooth extends 2-3x the clip's own median, a headlight pop 11x,
    bridges that cut 10-60x (a plain frame difference rated a fast push-in as a jump and
    missed cuts inside the generated stretch)."""
    g = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames[lo:hi + 1]]
    h, w = g[0].shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = []
    for a, b in zip(g, g[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        warped = cv2.remap(b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.append(float(np.abs(warped - a)[4:-4, 4:-4].mean()))
    return out


def continuity(jump):
    """smooth (< 5x the clip's own residual) | pop (a visible flash or jump) | cut (> 20x)."""
    return "smooth" if jump < 5 else "pop" if jump < 20 else "cut"


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
    # a removed object leaves new content against old (cleaned paving beside the car's own
    # shadow): anchor removals blend wider; overlays keep a tight edge off the car
    feather = int(plan.get("feather", 8))
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

    # metrics: the largest frame-to-frame jump anywhere in or at the edges of the edited
    # stretch, against the original footage's own motion (a bridge between very different
    # framings comes back as a hard cut INSIDE the generated frames, which a seams-only
    # measure missed), and flicker inside the mask
    seam_jump, jump_at, med = 0.0, None, 0.0
    span = sorted(gen_set & set(range(lo, hi + 1)))
    if span:
        r0, r1 = max(lo, span[0] - 1 - HANDOFF), min(hi, span[-1] + 1 + HANDOFF)
        res = motion_residuals(out, lo, hi)
        orig = [d for k, d in enumerate(res) if not (r0 <= lo + k <= r1)]
        med = float(np.median(orig or res))
        for i in range(r0, r1):
            v = res[i - lo] / max(med, 0.3)
            if v > seam_jump:
                seam_jump, jump_at = v, i
    flick = []
    for i in range(lo, hi):
        v0, v1 = vmask.get(i), vmask.get(i + 1)
        if v0 and v1 and v0 != "full" and v1 != "full":
            b = v0[0]
            x0, y0, x1, y1 = [int(round(t)) for t in b]
            if x1 - x0 > 4 and y1 - y0 > 4:
                flick.append(float(np.abs(out[i + 1][y0:y1, x0:x1].astype(np.int16)
                                          - out[i][y0:y1, x0:x1].astype(np.int16)).mean()))
    metrics = {"seam_jump": round(seam_jump, 3), "jump_frame": jump_at, "continuity": continuity(seam_jump),
               "residual_median": round(med, 3),
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
              "canvas": plan["canvas"], "crop": plan["crop"], "anchor_frames": plan.get("anchor_frames", []),
              "warnings": plan["warnings"]}
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


def clean_inputs(spec):
    """After a first prep (plan.json in work_dir): the frames and masks an image model cleans
    for anchor removal. The main frame is spec["frame"], or the one where the boxes cover the
    most (the object fully in view), a key frame in the middle of that stretch. The first and
    last boxed key frames join it when they are a clip or more away: a moving camera carries
    one cleaned frame only so far, and an object entering or leaving would otherwise pop.
    Writes <out_prefix><frame>.png and <out_prefix><frame>_mask.png at the image model's size
    (spec["size"]) and clean.json {frames: [{frame, image, mask}]}."""
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    fh, fw = plan["height"], plan["width"]
    n = plan["frames"] - plan.get("pad_tail", 0)
    lo = plan["window"][0]
    boxed = {int(k): v for k, v in plan["vmask"].items() if v != "full" and int(k) < n}
    if not boxed:
        raise RuntimeError("cleaning needs region boxes")
    keys = sorted(k for k in boxed if (k - lo) % LATENT_CLIP == 0)
    if spec.get("frame") is not None:
        a = int(spec["frame"])
        if a not in boxed:
            a = min(boxed, key=lambda k: abs(k - a))
    else:
        area = {k: float(_box_mask(v, fh, fw).sum()) for k, v in boxed.items()}
        top = max(area.values())
        full = sorted(k for k, s in area.items() if s >= 0.98 * top)
        # prefer a key frame (window start + 17k): there the cleaned frame is an anchor as is
        key = [k for k in full if k in keys]
        a = (key or full)[len(key or full) // 2]
    picks = [a] + [k for k in (keys[:1] + keys[-1:]) if abs(k - a) >= LATENT_CLIP]
    picks = sorted(set(picks[:int(spec.get("max_frames", 3))]))
    cap = cv2.VideoCapture(spec["clip"])
    got = {}
    for i in range(max(picks) + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} of the clip")
        if i in picks:
            got[i] = frame
    cap.release()
    kw, kh = spec["size"]
    out = []
    for k in picks:
        img, mask = f"{spec['out_prefix']}{k}.png", f"{spec['out_prefix']}{k}_mask.png"
        m = _box_mask(boxed[k], fh, fw, grow=int(spec.get("grow", 24)))
        cv2.imwrite(img, cv2.resize(got[k], (kw, kh), interpolation=cv2.INTER_AREA))
        cv2.imwrite(mask, cv2.resize(m, (kw, kh), interpolation=cv2.INTER_NEAREST))
        out.append({"frame": k, "image": img, "mask": mask})
    json.dump({"frames": out, "main": a, "width": fw, "height": fh}, open(os.path.join(work, "clean.json"), "w"))
    print(f"MEASURED clean frames {picks} (main {a}) at {kw}x{kh}", flush=True)


if __name__ == "__main__":
    cmds = {"prep": prep, "compose": compose, "clean_inputs": clean_inputs}
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit("usage: clip_edit.py prep|clean_inputs|compose <spec.json>")
    cmds[sys.argv[1]](json.load(open(sys.argv[2])))
