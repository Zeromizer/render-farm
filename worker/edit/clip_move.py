"""Clip move, pixel side: send a car in an existing (generated) clip along a new path.

The edit, from the 2026-10-10 spike (spikes/route/try_route_edit.py):

  SAM 3.1 tracks the car in the source (its old path, frame by frame)
  -> a clean plate: the road without it (masked temporal median; locked-off camera only)
  -> the car cut from the start frame and dragged along the new route over the source with
     the old car taken out, turned to face where it is going = the cut-and-drag reference
  -> H3 + vlo Time-to-Move renders the clip from that (videogen/graphs_ttm.py)
  -> SAM finds the car again in each render
  -> only the car comes from the render: pasted where SAM found it (H3 runs off and behind
     the drawn path, so pasting by the drawn path clipped it); the old car AND its soft
     shadow halo (H3 cars darken the road out to ~40-50 px) are covered by the clean plate,
     graded to each frame; everything else is the source, pixel for pixel, with its sound.

Three subcommands driven by a spec JSON (run in the planar_patch venv: opencv + numpy):

  prep     conform the clip to an H3 canvas on the 17k+5 grid, refuse a moving camera,
           write move_src.mp4 and plan.json
  build    pick the car's track, build the clean plate and the reference, write the H3
           inputs (move_ref.mp4, move_refmask.mp4, first.png), oldmask.mp4, poses.json,
           route.png
  compose  pick the car in each render, score the takes against the drawn path, compose
           each at the source's own resolution, write the best one, a proof sheet and a
           report
"""
import json
import math
import shutil
import os
import subprocess
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "lettering"))
import lettering as L  # noqa: E402  (io, grid: one implementation)

FPS = L.FPS
MAX_FRAMES = L.MAX_WINDOW          # ~15 s: one H3 pass
MAX_AREA = 1344 * 768              # H3's native canvas
# Canvas pixels x frames one TTM pass holds on the 16 GB card: 1344x768x124 ran (2026-10-10).
PIXEL_FRAMES = 1.3e8
# How far the background may drift (fraction of the frame width, over the clip) and still
# count as a locked-off shot. The clean plate is one image, so a camera move breaks it.
MAX_CAMERA_DRIFT = 0.012
BASE_W = 832                       # every pixel size below was tuned on 832-wide footage

emit = L.emit
log = lambda s: print(s, flush=True)  # noqa: E731


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def px(v, w):
    """A size tuned at 832 wide, for a frame w wide."""
    return max(1, int(round(v * w / BASE_W)))


def snap_up(n):
    """Smallest 17k+5 >= n."""
    return 5 if n <= 5 else 5 + int(math.ceil((n - 5) / 17)) * 17


def canvas_for(fw, fh, length):
    area = min(MAX_AREA, PIXEL_FRAMES / max(1, length), max(fw * fh, 256 * 256))
    return L.gen_dims(fw / fh, area)


def _disk(r):
    r = max(1, int(r))
    return np.ones((2 * r + 1, 2 * r + 1), np.uint8)


# ---------------------------------------------------------------- video io

class Reader:
    """Frames one at a time (long 1080p clips do not fit in memory as a list)."""

    def __init__(self, path, size=None):
        self.cap = cv2.VideoCapture(path)
        self.size = size
        self.last = None

    def next(self):
        ok, f = self.cap.read()
        if ok:
            if self.size and (f.shape[1], f.shape[0]) != tuple(self.size):
                f = cv2.resize(f, tuple(self.size), interpolation=cv2.INTER_AREA)
            self.last = f
        return self.last  # past the end: the last frame again (padding)

    def close(self):
        self.cap.release()


class Writer:
    def __init__(self, path, w, h, fps, audio_from=None, frames=None, lossless=False):
        ffmpeg = L.find_ffmpeg_tool("ffmpeg")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-"]
        if audio_from:
            cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a?"]
        if lossless:
            cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p"]
        else:
            cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
        if audio_from:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{(frames or 0) / fps:.4f}"]
        cmd += [path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.path = path

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def read_masks(path, n, size=None):
    """Bool masks from a mask video (red channel), resized to size, padded/cut to n."""
    out = []
    cap = cv2.VideoCapture(path)
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        m = f[..., 2]
        if size and (m.shape[1], m.shape[0]) != tuple(size):
            m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_LINEAR)
        out.append(m > 127)
    cap.release()
    if not out:
        raise RuntimeError(f"no frames in mask video {path}")
    while len(out) < n:
        out.append(np.zeros_like(out[0]))
    return out


def write_masks(path, masks, fps=FPS):
    h, w = masks[0].shape
    wr = Writer(path, w, h, fps, lossless=True)
    for m in masks:
        wr.write(np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2))
    wr.close()


# ---------------------------------------------------------------- camera

def camera_drift(frames, step=6, scale=0.5):
    """How far the background moves over the clip, as a fraction of the frame width: corner
    tracks frame to frame (RANSAC similarity, so a moving car does not count), chained, the
    largest displacement any frame corner reaches."""
    h, w = frames[0].shape[:2]
    S = np.diag([scale, scale, 1.0])
    gray = lambda f: cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)  # noqa: E731
    M = np.eye(3)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    worst = 0.0
    prev = gray(frames[0])
    for i in range(step, len(frames), step):
        cur = gray(frames[i])
        p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
        if p0 is None or len(p0) < 12:
            prev = cur
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
        ok = st.ravel() == 1
        if ok.sum() < 12:
            prev = cur
            continue
        A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if A is not None:
            Af = np.vstack([A, [0, 0, 1]])
            M = (np.linalg.inv(S) @ Af @ S) @ M
            moved = M @ corners
            worst = max(worst, float(np.max(np.hypot(moved[0] - corners[0], moved[1] - corners[1]))))
        prev = cur
    return worst / w


# ---------------------------------------------------------------- route geometry

def catmull_rom(pts, per=64):
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = (np.array(p[j], float) for j in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(np.array(pts[-1], float))
    return np.array(out)


def poses(route_px, n, start, hold, arrive, ease="inout"):
    """(x, y, heading) per frame: still until start + hold, along the spline by arc length
    (smoothstep ease) to arrive at `arrive`, then still at the end. Frames before start get
    the first point (the caller does not use them)."""
    dense = catmull_rom(route_px)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    go = start + hold
    moving = max(1, arrive - go)
    out = []
    for f in range(n):
        u = min(1.0, max(0.0, (f - go) / moving))
        if ease == "inout":
            u = u * u * (3 - 2 * u)
        d = u * total
        i = int(np.searchsorted(s, d, side="right") - 1)
        i = min(max(i, 0), len(dense) - 2)
        a = (d - s[i]) / max(seg[i], 1e-6)
        x, y = dense[i] + a * (dense[i + 1] - dense[i])
        j0, j1 = max(0, i - 3), min(len(dense) - 1, i + 4)
        dx, dy = dense[j1] - dense[j0]
        out.append((float(x), float(y), math.atan2(dy, dx)))
    return out


def car_axis(mask):
    """The car's long-axis angle (radians, image coords) and centre, from its mask."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    w, v = np.linalg.eigh(cov)
    vx, vy = v[:, int(np.argmax(w))]
    return math.atan2(vy, vx), (float(cx), float(cy))


def centroid(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


# ---------------------------------------------------------------- tracks

def pick_track(tracks, frame, point, w, h):
    """The track under (or nearest) the hint point at `frame` (or the frames around it)."""
    px_, py_ = point[0] * w, point[1] * h
    best = None
    for k, ms in enumerate(tracks):
        for df in (0, 1, -1, 2, -2, 4, -4, 6, -6):
            f = frame + df
            if not 0 <= f < len(ms) or not ms[f].any():
                continue
            hit = ms[f][min(max(int(py_), 0), h - 1), min(max(int(px_), 0), w - 1)]
            c = centroid(ms[f])
            d = math.hypot(c[0] - px_, c[1] - py_) / w
            score = (0 if hit else 1, abs(df), d)
            if best is None or score < best[0]:
                best = (score, k)
            break
    if best is None or (best[0][0] and best[0][2] > 0.15):
        raise RuntimeError("no tracked object at the hint point; point at the car in the frame given")
    return best[1]


def main_body(masks, w):
    """Each frame's mask cut down to the object itself: its largest piece plus any piece of
    real size (a quarter of it) close by (a car split by a pole or a lane line). SAM hands a
    car's track stray pieces on other things now and then: on a render, a stray piece over
    the white SUV got pasted as a white patch beside it (2026-10-10). Returns (masks, number
    of frames that lost a stray piece)."""
    near = _disk(px(12, w))
    out, cut = [], 0
    for m in masks:
        if not m.any():
            out.append(m)
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if n <= 2:
            out.append(m)
            continue
        big = 1 + int(np.argmax(st[1:, 4]))
        reach = cv2.dilate((lab == big).astype(np.uint8), near) > 0
        keep = np.zeros(n, bool)
        keep[big] = True
        for k in range(1, n):
            if k != big and st[k, 4] >= 0.25 * st[big, 4] and reach[lab == k].any():
                keep[k] = True
        kept = keep[lab] & m
        cut += int(kept.sum() < m.sum())
        out.append(kept)
    return out, cut


def fill_gaps(masks, max_gap=12):
    """SAM loses a fast or partly hidden car for a few frames now and then: fill short gaps
    by sliding the nearest mask along the line between the masks either side (a missed frame
    left the old car showing)."""
    n = len(masks)
    have = [m.any() for m in masks]
    out = list(masks)
    f = 0
    while f < n:
        if have[f]:
            f += 1
            continue
        g0 = f
        while f < n and not have[f]:
            f += 1
        a, b = g0 - 1, f
        if a < 0 or b >= n or b - a - 1 > max_gap:
            continue
        ca, cb = centroid(masks[a]), centroid(masks[b])
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            src = masks[a] if t < 0.5 else masks[b]
            cs = ca if t < 0.5 else cb
            tx = ca[0] + t * (cb[0] - ca[0]) - cs[0]
            ty = ca[1] + t * (cb[1] - ca[1]) - cs[1]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            h, w = src.shape
            out[k] = cv2.warpAffine(src.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    return out


# ---------------------------------------------------------------- plate + grade

def clean_plate(frames, old, grow, others=None, other_grow=None):
    """Locked-off camera: the empty road, per pixel - the median over the frames where no car
    is: not the one being moved (with its shadow halo, `grow` px round it) and not any other
    traffic SAM found (`others`, `other_grow` px round each). Guessing traffic from pixel
    statistics failed both ways (2026-10-10): with only the moved car excluded, a passing SUV
    won the median near where the car had waited; a median-agreement filter then let the car
    itself in where it had sat most of the clip. Where every frame has a car, fall back to
    the frames clear of the moved car, then to the plain median. frames/old/others may be a
    sample of the clip."""
    h, w = frames[0].shape[:2]
    own = [cv2.dilate(m.astype(np.uint8), _disk(grow)) > 0 for m in old]
    body = [cv2.dilate(m.astype(np.uint8), _disk(max(2, grow // 8))) > 0 for m in old]
    traffic = [np.zeros((h, w), bool)] * len(old)
    if others is not None:
        ko = _disk(other_grow if other_grow is not None else grow // 2)
        traffic = [cv2.dilate(o.astype(np.uint8), ko) > 0 for o in others]
    # best first: no car and no shadow; then only the moved car's soft shadow (road, a little
    # shaded: far better than another car); then clear of the moved car; then anything
    tiers = ([o | t for o, t in zip(own, traffic)], [b | t for b, t in zip(body, traffic)], own)
    plate = np.zeros((h, w, 3), np.uint8)
    for y0 in range(0, h, 48):
        blk = np.stack([f[y0:y0 + 48] for f in frames]).astype(np.float32)
        med = np.median(blk, axis=0)
        for tier in reversed(tiers):
            x = blk.copy()
            x[np.stack([c[y0:y0 + 48] for c in tier])] = np.nan
            with np.errstate(all="ignore"):
                m = np.nanmedian(x, axis=0)
            med = np.where(np.isnan(m), med, m)
        plate[y0:y0 + 48] = np.clip(med, 0, 255).astype(np.uint8)
    return plate


def matched_plate(frame, plate, cover, sigma):
    """The plate brought to this frame's grade: a smooth correction field from the static
    pixels round it (frame - plate where they agree and the car is not). H3 footage drifts in
    brightness over a clip; the plain plate showed as a dark car-shaped patch. Computed at a
    quarter size (it is smooth)."""
    h, w = frame.shape[:2]
    q = 4
    sw, sh = max(8, w // q), max(8, h // q)
    fs = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    ps = cv2.resize(plate, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    cs = cv2.resize(cover.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    d = fs - ps
    valid = ((np.abs(d).max(2) < 20) & ~cs).astype(np.float32)
    num = cv2.GaussianBlur(d * valid[..., None], (0, 0), sigma / q)
    den = cv2.GaussianBlur(valid, (0, 0), sigma / q)[..., None]
    corr = cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)
    return plate.astype(np.float32) + corr


def remove_old(frame, old_m, plate, w_ref):
    """The old car and its shadow halo replaced by the graded plate (body + 48 px at 832
    wide, feathered wide); things much brighter or darker than the road outside the body
    (another car passing close) keep their own pixels."""
    if not old_m.any():
        return frame.astype(np.float32)
    g, cg = px(48, w_ref), px(60, w_ref)
    sc = g / 48.0
    h, w = frame.shape[:2]
    ys, xs = np.nonzero(old_m)
    pad = cg + int(30 * sc) + 8
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    fr = frame[y0:y1, x0:x1]
    m = old_m[y0:y1, x0:x1].astype(np.uint8)
    pl = plate[y0:y1, x0:x1]
    fill = matched_plate(fr, pl, cv2.dilate(m, _disk(cg)) > 0, 25 * sc)
    reg = cv2.dilate(m, _disk(g)) > 0
    tight = cv2.dilate(m, _disk(round(6 * sc))) > 0
    base = fr.astype(np.float32)
    other = (np.abs(base - pl.astype(np.float32)).max(2) > 45) & ~tight
    other = cv2.dilate(other.astype(np.uint8), _disk(round(4 * sc))) > 0
    a = cv2.GaussianBlur((reg & ~other).astype(np.float32), (0, 0), 10 * sc)[..., None]
    a = np.maximum(a, cv2.GaussianBlur(tight.astype(np.float32), (0, 0), 2 * sc)[..., None])
    out = frame.astype(np.float32)
    out[y0:y1, x0:x1] = base * (1 - a) + fill * a
    return out


def paste_car(base, raw, car_m, w_ref):
    if not car_m.any():
        return base
    g = px(8, w_ref)
    a = cv2.GaussianBlur((cv2.dilate(car_m.astype(np.uint8), _disk(g)) > 0).astype(np.float32),
                         (0, 0), 2.5 * g / 8.0)[..., None]
    return base * (1 - a) + raw.astype(np.float32) * a


# ---------------------------------------------------------------- prep

def prep(spec):
    work = spec["work_dir"]
    emit("PHASE", "reading the clip")
    frames, fps = L.read_clip(spec["clip"])
    n = len(frames)
    warnings = []
    if abs(fps - FPS) > 0.5:
        warnings.append(f"clip is {fps:.2f} fps; H3 works at {FPS}, frames are taken one for one")
    if n > MAX_FRAMES:
        raise RuntimeError(f"the clip is {n} frames (~{n / FPS:.1f} s); a move edit handles up to {MAX_FRAMES} "
                           f"(~{MAX_FRAMES / FPS:.0f} s). Trim it first.")
    if n < 17:
        raise RuntimeError("the clip is too short for a move edit (under a second)")
    fh, fw = frames[0].shape[:2]
    emit("PHASE", "checking the camera")
    drift = camera_drift(frames)
    log(f"MEASURED camera drift {drift * 100:.2f}% of the width over the clip")
    if drift > MAX_CAMERA_DRIFT:
        raise RuntimeError(f"the camera moves in this clip (the background drifts {drift * 100:.1f}% of the frame "
                           f"width); move edits need a locked-off shot for now")
    length = snap_up(n)
    W, H = canvas_for(fw, fh, length)
    emit("PROGRESS", 50)
    small = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames]
    small += [small[-1]] * (length - n)
    L.write_clip(os.path.join(work, "move_src.mp4"), small, FPS, audio_from=spec["clip"], lossless=True)
    plan = {"mode": "move", "fps": FPS, "frames": n, "width": fw, "height": fh, "length": length,
            "canvas": [W, H], "camera_drift": round(drift, 4), "warnings": warnings}
    _save(os.path.join(work, "plan.json"), plan)
    log(f"MEASURED canvas {W}x{H}, {n} frames -> {length} on the H3 grid")
    emit("PROGRESS", 100)


# ---------------------------------------------------------------- build

def _frame_of(spec, key_f, key_s, default):
    if spec.get(key_f) is not None:
        return int(spec[key_f])
    if spec.get(key_s) is not None:
        return int(round(float(spec[key_s]) * FPS))
    return default


def build(spec):
    work = spec["work_dir"]
    plan = _load(os.path.join(work, "plan.json"))
    W, H = plan["canvas"]
    n, length = plan["frames"], plan["length"]
    emit("PHASE", "reading tracks")
    frames, _ = L.read_clip(os.path.join(work, "move_src.mp4"))
    frames = (frames + [frames[-1]] * length)[:length]
    tracks = [read_masks(p, length, (W, H)) for p in spec["sam_masks"]]
    if not tracks:
        raise RuntimeError(f"SAM found no {spec.get('object', 'object')!r} in the clip")
    s0 = min(max(0, _frame_of(spec, "start_frame", "start_s", 0)), n - 2)
    k = pick_track(tracks, s0, spec["point_norm"], W, H)
    body, cut = main_body(tracks[k], W)
    if cut:
        log(f"MEASURED old track: dropped stray pieces in {cut} frame(s)")
    old = fill_gaps(body)
    old = [m if f >= s0 else np.zeros_like(m) for f, m in enumerate(old)]
    if not old[s0].any():
        raise RuntimeError("the car is not tracked at the start frame; give a start where it is in view")
    seen = sum(1 for m in old[s0:n] if m.any())
    log(f"MEASURED old track: object {k}, in {seen}/{n - s0} frames from frame {s0}")
    emit("PROGRESS", 15)

    # other traffic (SAM's generic "car" pass): kept out of the clean plate
    others = [np.zeros((H, W), bool) for _ in range(length)]
    for pth in spec.get("other_masks") or []:
        for f, m in enumerate(read_masks(pth, length, (W, H))):
            others[f] |= m
    write_masks(os.path.join(work, "othermask.mp4"), others)

    # clean plate (a sample of up to ~96 frames is plenty for a median)
    step = max(1, n // 96)
    idx = list(range(0, n, step))
    plate = clean_plate([frames[i] for i in idx], [old[i] for i in idx], px(60, W),
                        [others[i] for i in idx], px(30, W))
    cv2.imwrite(os.path.join(work, "plate.png"), plate)
    emit("PROGRESS", 35)

    # route: from the car's centre at s0 through the given points (0-1 of the frame)
    m0 = old[s0]
    ang, (cx, cy) = car_axis(m0)
    pts = [(float(x) * W, float(y) * H) for x, y in spec["route"]]
    if pts and math.hypot(pts[0][0] - cx, pts[0][1] - cy) < 0.04 * W:
        pts = pts[1:]  # the agent gave the car's own position first
    if not pts:
        raise RuntimeError("the route needs at least one point after the car's position")
    route = [(cx, cy)] + pts
    hold = max(0, _frame_of(spec, "hold_frames", "hold_s", int(round(0.25 * FPS))))
    arrive = _frame_of(spec, "arrive_frame", "arrive_s", n - 1)
    arrive = min(max(arrive, s0 + hold + 6), n - 1)
    ps = poses(route, length, s0, hold, arrive, spec.get("ease", "inout"))
    turn = spec.get("turn", True) is not False
    # the nose: the way the car was going in the source (old track after s0, else before),
    # else the way the new route starts
    h_route = ps[min(length - 1, s0 + hold + 2)][2]
    h_src = None
    for f in list(range(s0 + 12, min(n, s0 + 25))) + list(range(max(0, s0 - 12), s0)):
        src_m = tracks[k][f] if f < len(tracks[k]) else None
        c = centroid(src_m) if src_m is not None and src_m.any() else None
        if c and math.hypot(c[0] - cx, c[1] - cy) > 3:
            h_src = math.atan2(c[1] - cy, c[0] - cx) if f > s0 else math.atan2(cy - c[1], cx - c[0])
            break
    if math.cos(ang - (h_src if h_src is not None else h_route)) < 0:
        ang += math.pi
    emit("PHASE", "building the reference")
    alpha = cv2.GaussianBlur(m0.astype(np.float32), (0, 0), 1.2)
    car = frames[s0].astype(np.float32)
    cover_k = _disk(px(15, W))
    k10 = _disk(px(10, W))
    refs, refmask, new = [], [], []
    for f in range(length):
        if f < s0:
            refs.append(frames[f])
            refmask.append(cv2.dilate(tracks[k][f].astype(np.uint8), k10) > 0)
            new.append(tracks[k][f])
            continue
        cov = cv2.GaussianBlur((cv2.dilate(old[f].astype(np.uint8), cover_k) > 0).astype(np.float32), (0, 0), 3)[..., None]
        bg = frames[f].astype(np.float32) * (1 - cov) + plate.astype(np.float32) * cov
        x, y, hd = ps[f]
        dth = 0.0
        if turn:
            ramp = min(1.0, max(f - s0 - hold, 0) / 8.0)
            dth = math.atan2(math.sin(hd - ang), math.cos(hd - ang)) * ramp
        M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(dth), 1.0)
        M[0, 2] += x - cx
        M[1, 2] += y - cy
        c = cv2.warpAffine(car, M, (W, H), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        refs.append(np.clip(bg * (1 - a) + c * a, 0, 255).astype(np.uint8))
        nm = a[..., 0] > 0.5
        new.append(nm)
        refmask.append(cv2.dilate(nm.astype(np.uint8), k10) > 0)
        if f % 24 == 0:
            emit("PROGRESS", 40 + int(50 * f / length))
    L.write_clip(os.path.join(work, "move_ref.mp4"), refs, FPS, audio_from=os.path.join(work, "move_src.mp4"),
                 lossless=True)
    write_masks(os.path.join(work, "move_refmask.mp4"), refmask)
    write_masks(os.path.join(work, "oldmask.mp4"), old)
    cv2.imwrite(os.path.join(work, "first.png"), frames[0])
    drawn = []
    for f in range(length):
        c = centroid(new[f]) if f >= s0 and new[f].any() else None
        drawn.append(None if c is None else [round(c[0], 1), round(c[1], 1)])
    _save(os.path.join(work, "poses.json"), {"start": s0, "hold": hold, "arrive": arrive, "drawn": drawn, "track": k,
                                             "route": [[round(x, 1), round(y, 1)] for x, y in route]})
    pic = frames[s0].copy()
    for f in range(s0, n, 3):
        cs, _ = cv2.findContours(tracks[k][f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pic, cs, -1, (0, 200, 255), 1)
    for (x0, y0, _), (x1, y1, _) in zip(ps[s0:n], ps[s0 + 1:n]):
        cv2.line(pic, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    for x, y in route:
        cv2.circle(pic, (int(x), int(y)), 4, (0, 255, 255), -1)
    cv2.imwrite(os.path.join(work, "route.png"), pic)
    emit("PROGRESS", 100)


# ---------------------------------------------------------------- compose

def pick_render_car(tracks, drawn, s0, n, w):
    """The render's track that sits on the drawn path most, and its score: mean distance
    (832-scale px) from the drawn car, plus a miss penalty for frames where the drawn car is
    in view and the render's is not."""
    near = px(60, w)
    best = None
    for k, ms in enumerate(tracks):
        d, miss, hits = [], 0, 0
        for f in range(s0, n):
            want = drawn[f]
            if want is None:
                continue
            c = centroid(ms[f]) if f < len(ms) and ms[f].any() else None
            if c is None:
                miss += 1
                continue
            dist = math.hypot(c[0] - want[0], c[1] - want[1])
            d.append(dist)
            hits += dist < near
        if not hits:
            continue
        score = (float(np.mean(d)) + 40.0 * miss / max(1, len(d) + miss)) * BASE_W / w
        if best is None or score < best[0]:
            best = (score, k, float(np.mean(d)) * BASE_W / w, miss)
    return best


def compose(spec):
    work = spec["work_dir"]
    plan = _load(os.path.join(work, "plan.json"))
    pz = _load(os.path.join(work, "poses.json"))
    W, H = plan["canvas"]
    n, length = plan["frames"], plan["length"]
    fw, fh = plan["width"], plan["height"]
    s0 = pz["start"]
    emit("PHASE", "scoring takes")
    results = []
    for i, (take, masks) in enumerate(zip(spec["takes"], spec["take_masks"])):
        tracks = [read_masks(p, length, (W, H)) for p in masks]
        pick = pick_render_car(tracks, pz["drawn"], s0, n, W)
        if pick is None:
            results.append({"take": i + 1, "seed": spec["seeds"][i], "ok": False,
                            "why": "the car was not found on the drawn path in this render"})
            continue
        score, k, dist, miss = pick
        results.append({"take": i + 1, "seed": spec["seeds"][i], "ok": True, "score": round(score, 1),
                        "path_error_px832": round(dist, 1), "missing_frames": miss, "track": k})
        log(f"take {i + 1}: car track {k}, {dist:.1f} px off the drawn path (832 scale), {miss} frames missing")
    good = [r for r in results if r["ok"]]
    if not good:
        raise RuntimeError("H3 did not draw the car on the new path in any take; try a gentler route or another seed")
    best = min(good, key=lambda r: r["score"])["take"]

    # the plate again at the source's own size (from a sample of frames + upscaled masks)
    emit("PHASE", "clean plate at full size")
    old_small = read_masks(os.path.join(work, "oldmask.mp4"), length)
    om_path = os.path.join(work, "othermask.mp4")
    other_small = read_masks(om_path, length) if os.path.exists(om_path) else None
    up = lambda m: cv2.resize(m.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_LINEAR) > 0  # noqa: E731
    step = max(1, n // 96)
    rd = Reader(spec["clip"])
    sample, sample_m, sample_o = [], [], []
    for f in range(n):
        fr = rd.next()
        if f % step == 0:
            sample.append(fr.copy())
            sample_m.append(up(old_small[f]))
            if other_small is not None:
                sample_o.append(up(other_small[f]))
    rd.close()
    plate = clean_plate(sample, sample_m, px(60, fw), sample_o if other_small is not None else None, px(30, fw))
    del sample, sample_m, sample_o

    outs = {}
    for r in good:
        i = r["take"] - 1
        emit("PHASE", f"composing take {r['take']}")
        car_small, cut = main_body(read_masks(spec["take_masks"][i][r["track"]], length), W)
        if cut:
            log(f"take {r['take']}: dropped stray pieces of the car's mask in {cut} frame(s)")
        dest = os.path.join(work, f"take{r['take']}.mp4")
        src, raw = Reader(spec["clip"]), Reader(spec["takes"][i])
        wr = Writer(dest, fw, fh, FPS, audio_from=spec["clip"], frames=n)
        for f in range(n):
            fr = src.next()
            rf = raw.next()
            if f < s0:
                wr.write(fr)
                continue
            if (rf.shape[1], rf.shape[0]) != (fw, fh):
                rf = cv2.resize(rf, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
            om = cv2.resize(old_small[f].astype(np.uint8), (fw, fh), interpolation=cv2.INTER_LINEAR) > 0
            u = np.zeros_like(car_small[0])
            for j in range(max(s0, f - 1), min(n, f + 2)):
                u |= car_small[j]
            cm = cv2.resize(u.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_LINEAR) > 0
            out = paste_car(remove_old(fr, om, plate, fw), rf, cm, fw)
            wr.write(np.clip(out, 0, 255).astype(np.uint8))
            if f % 24 == 0:
                emit("PROGRESS", int(100 * (f + 1) / n))
        wr.close()
        src.close()
        raw.close()
        outs[r["take"]] = dest
    shutil.copyfile(outs[best], spec["out"])
    proof_sheet(spec["proof"], work, spec["clip"], outs[best], s0, n)
    report = {"mode": "move", "best_take": best, "takes": results, "canvas": [W, H], "frames": n,
              "start_frame": s0, "arrive_frame": pz["arrive"], "camera_drift": plan["camera_drift"],
              "warnings": plan.get("warnings", [])}
    _save(spec["report"], report)
    log(f"best take {best}")


def proof_sheet(path, work, clip, out_path, s0, n):
    """The drawn route (old path outlined in orange, the new one in red), then six moments,
    before over after, in two rows of three: big enough to judge the edit by."""
    route = cv2.imread(os.path.join(work, "route.png"))
    picks = [int(s0 + (n - 1 - s0) * t) for t in (0.1, 0.3, 0.45, 0.6, 0.75, 0.9)]
    tile_w = 480
    th = int(tile_w * route.shape[0] / route.shape[1])
    pairs = {}
    a, b = Reader(clip), Reader(out_path)
    want = set(picks)
    for f in range(n):
        fa, fb = a.next(), b.next()
        if f in want:
            pa = cv2.resize(fa, (tile_w, th), interpolation=cv2.INTER_AREA)
            pb = cv2.resize(fb, (tile_w, th), interpolation=cv2.INTER_AREA)
            for img, label in ((pa, f"before  {f / FPS:.1f}s"), (pb, f"after  {f / FPS:.1f}s")):
                cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            pairs[f] = np.vstack([pa, pb, np.zeros((6, tile_w, 3), np.uint8)])
    a.close()
    b.close()
    cols = [pairs[f] for f in picks if f in pairs]
    while len(cols) < 6:
        cols.append(np.zeros_like(cols[0]))
    sep = lambda img: np.hstack([img, np.zeros((img.shape[0], 6, 3), np.uint8)])  # noqa: E731
    grid = np.vstack([np.hstack([sep(c) for c in cols[:3]]), np.hstack([sep(c) for c in cols[3:]])])
    rt = cv2.resize(route, (int(grid.shape[0] * route.shape[1] / route.shape[0] * 0.5), grid.shape[0] // 2))
    rt = np.vstack([rt, np.zeros((grid.shape[0] - rt.shape[0], rt.shape[1], 3), np.uint8)])
    cv2.putText(rt, "route: old path (orange), new (red)", (8, rt.shape[0] // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.imwrite(path, np.hstack([sep(rt), grid]))


def main():
    sub, spec_path = sys.argv[1], sys.argv[2]
    spec = _load(spec_path)
    {"prep": prep, "build": build, "compose": compose}[sub](spec)


if __name__ == "__main__":
    main()
