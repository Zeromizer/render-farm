"""Clip move, pixel side: send a car in an existing (generated) clip along a new path.

The edit, from the 2026-10-10 spike (spikes/route/try_route_edit.py):

  SAM 3.1 tracks the car in the source (its old path, frame by frame)
  -> a clean plate video: the road without it, frame by frame - the LTX 2.5 Clean-Plate
     IC-LoRA (videogen/graphs_ltx_plate.py, 2026-10-11), which takes out every vehicle and its
     shadow and follows a moving camera. Without the LTX models on the PC: a masked temporal
     median over a locked-off shot (with every other car SAM finds kept out of it)
  -> the camera's motion, frame to frame (a ground-plane homography; identity when locked off):
     the route is a path on the ground, given in the start frame (or as timed points, each in
     the frame at its own time), so the car stays on the road while the camera moves
  -> the car cut from the start frame and dragged along the new route over the source with
     the old car taken out, turned to face where it is going = the cut-and-drag reference
  -> H3 + vlo Time-to-Move renders the clip from that (videogen/graphs_ttm.py)
  -> SAM finds the car again in each render
  -> only the car comes from the render: pasted where SAM found it (H3 runs off and behind
     the drawn path, so pasting by the drawn path clipped it); the old car AND its soft
     shadow halo (H3 cars darken the road out to ~40-50 px) are covered by the clean plate,
     graded to each frame; everything else is the source, pixel for pixel, with its sound.

Three subcommands driven by a spec JSON (run in the planar_patch venv: opencv + numpy):

  prep     conform the clip to an H3 canvas on the 17k+5 grid, track the camera (cam.json),
           write move_src.mp4, the LTX plate windows (ltx_src_<i>.mp4) and plan.json
  build    pick the car's track, stitch the LTX plate windows (plate_ltx.mp4) or build the
           median plate, build the reference, write the H3 inputs (move_ref.mp4,
           move_refmask.mp4, first.png), oldmask.mp4, poses.json, route.png
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
# count as a locked-off shot (then the camera is held still: no homography jitter). A moving
# camera needs the LTX plate: the median plate is one image.
MAX_CAMERA_DRIFT = 0.012
# LTX plate windows (videogen/graphs_ltx_plate.py WINDOW_*: what the 16 GB card ran) and how
# much consecutive windows overlap (crossfaded).
LTX_PIXELS = 1024 * 576
LTX_FRAMES = 121
LTX_OVERLAP = 16
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
        if audio_from == "silence":  # the LTX plate graph encodes audio: give it a silent track
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a"]
        elif audio_from:
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

def camera_track(frames, scale=0.5):
    """The camera's motion: per frame, the 3x3 homography taking frame 0's pixels to that
    frame's (the ground plane, as RANSAC sees most of the picture; a moving car is an
    outlier), chained frame to frame. Returns (homographies, drift, frames it lost track):
    drift = the largest displacement any frame corner reaches, as a fraction of the width.
    A step it cannot measure repeats the last one."""
    h, w = frames[0].shape[:2]
    S = np.diag([scale, scale, 1.0])
    Si = np.linalg.inv(S)
    gray = lambda f: cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)  # noqa: E731
    Hs, fails, last = [np.eye(3)], 0, np.eye(3)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    worst = 0.0
    prev = gray(frames[0])
    for i in range(1, len(frames)):
        cur = gray(frames[i])
        step = None
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8)
        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
            ok = st.ravel() == 1
            if ok.sum() >= 12:
                Hm, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.5)
                if Hm is not None and inl is not None and inl.sum() >= 30:
                    step = Si @ Hm @ S
                else:  # few points: a similarity is steadier than a homography
                    A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.5)
                    if A is not None and inl is not None and inl.sum() >= 10:
                        step = Si @ np.vstack([A, [0, 0, 1]]) @ S
        if step is None:
            fails += 1
            step = last
        last = step
        Hs.append(step @ Hs[-1])
        moved = Hs[-1] @ corners
        moved = moved[:2] / moved[2]
        worst = max(worst, float(np.max(np.hypot(moved[0] - corners[0], moved[1] - corners[1]))))
        prev = cur
    return Hs, worst / w, fails


def camera_drift(frames):
    """How far the background moves over the clip, as a fraction of the frame width."""
    return camera_track(frames)[1]


def static_drift(frames, step=6, scale=0.5):
    """Is the camera locked off? Every step-th frame registered DIRECTLY to frame 0 (a
    similarity, RANSAC), not chained: chaining adds up each step's error (a locked-off H3
    clip read 1.1% chained against 0.75% here, 2026-10-11). The largest corner displacement
    as a fraction of the width; 1.0 when a frame cannot be registered at all."""
    h, w = frames[0].shape[:2]
    gray = lambda f: cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)  # noqa: E731
    g0 = gray(frames[0])
    p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 8)
    if p0 is None or len(p0) < 12:
        return 0.0  # nothing to track: a featureless frame cannot show a camera move either
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) * scale
    worst = 0.0
    for i in range(step, len(frames), step):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, gray(frames[i]), p0, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        if ok.sum() < 12:
            return 1.0
        A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if A is None or inl is None or inl.sum() < 10:
            return 1.0
        moved = corners @ A[:, :2].T + A[:, 2]
        worst = max(worst, float(np.max(np.hypot(*(moved - corners).T))) / scale)
    return worst / w


def to_canvas(Hs, fw, fh, W, H):
    """Source-pixel homographies as canvas-pixel ones."""
    S = np.diag([W / fw, H / fh, 1.0])
    Si = np.linalg.inv(S)
    return [S @ Hm @ Si for Hm in Hs]


def relative(Hs, s0):
    """G[f]: start-frame pixels -> frame f pixels."""
    inv0 = np.linalg.inv(Hs[s0])
    return [Hm @ inv0 for Hm in Hs]


def warp_pt(G, x, y):
    v = G @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


# ---------------------------------------------------------------- LTX plate windows

def ltx_dims(fw, fh, area=LTX_PIXELS):
    """The LTX canvas: the clip's aspect, 32-aligned, about `area` pixels."""
    ar = fw / fh
    w = max(256, int(round(math.sqrt(area * ar) / 32)) * 32)
    h = max(256, int(round(math.sqrt(area / ar) / 32)) * 32)
    while w * h > area * 1.04:
        if w >= h:
            w -= 32
        else:
            h -= 32
    return w, h


def ltx_windows(n, size=LTX_FRAMES, overlap=LTX_OVERLAP):
    """[(start, count)] covering n frames: one window of 8k+1 >= n frames for a short clip
    (padded with the last frame), else windows of `size` overlapping by at least `overlap`,
    the last one ending on the clip's last frame."""
    if n <= size:
        return [(0, 1 + 8 * int(math.ceil((n - 1) / 8)))]
    out, s = [(0, size)], 0
    while s + size < n:
        s = s + size - overlap
        if s + size >= n:
            s = n - size
        out.append((s, size))
    return out


def stitch_plate(paths, windows, n, out_path, size):
    """The plate windows as one n-frame video at `size` (lossless), overlaps crossfaded."""
    wr = Writer(out_path, size[0], size[1], FPS, lossless=True)
    pending = {}
    for i, ((s, c), pth) in enumerate(zip(windows, paths)):
        nxt = windows[i + 1][0] if i + 1 < len(windows) else None
        end = s + c - 1
        rd = Reader(pth, size)
        for j in range(c):
            f = s + j
            if f >= n:
                break
            img = rd.next().astype(np.float32)
            if f in pending:  # second half of an overlap with the previous window
                pw, acc = pending.pop(f)
                img = acc + img * (1.0 - pw)
            if nxt is not None and f >= nxt:
                wprev = 1.0 - (f - nxt + 1) / (end - nxt + 2)
                pending[f] = (wprev, img * wprev)
                continue
            wr.write(np.clip(img, 0, 255).astype(np.uint8))
        rd.close()
    wr.close()


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


PER = 64  # spline samples per route segment


def _along(dense, s, seg, d):
    """(x, y, heading) at arc length d along the dense spline."""
    i = int(np.searchsorted(s, d, side="right") - 1)
    i = min(max(i, 0), len(dense) - 2)
    a = (d - s[i]) / max(seg[i], 1e-6)
    x, y = dense[i] + a * (dense[i + 1] - dense[i])
    j0, j1 = max(0, i - 3), min(len(dense) - 1, i + 4)
    dx, dy = dense[j1] - dense[j0]
    return float(x), float(y), math.atan2(dy, dx)


def _spline(route_px):
    dense = catmull_rom(route_px, PER)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    return dense, seg, np.concatenate([[0], np.cumsum(seg)])


def poses(route_px, n, start, hold, arrive, ease="inout", v0=0.0):
    """(x, y, heading) per frame: still until start + hold, along the spline by arc length to
    arrive at `arrive`, then still at the end. ease inout: from v0 px/frame (0 = from rest:
    smoothstep; a car already driving keeps its speed) easing to a stop; linear: constant
    speed. Frames before start get the first point (the caller does not use them)."""
    dense, seg, s = _spline(route_px)
    total = s[-1]
    go = start + hold
    moving = max(1, arrive - go)
    m0 = min(max(0.0, v0) * moving, 3.0 * total)  # Hermite slope in u; <= 3x the mean stays monotone
    out = []
    for f in range(n):
        u = min(1.0, max(0.0, (f - go) / moving))
        if ease == "inout":
            d = (3 * u * u - 2 * u ** 3) * total + (u ** 3 - 2 * u * u + u) * m0
        else:
            d = u * total
        out.append(_along(dense, s, seg, d))
    return out


def _pchip(xk, yk, x, m0=None, m1=None):
    """Monotone cubic through (xk, yk) (Fritsch-Carlson), slopes m0/m1 at the ends (default
    the end secants): distance along the route over time, so the car never backs up."""
    xk, yk = np.asarray(xk, float), np.asarray(yk, float)
    h = np.diff(xk)
    dl = np.diff(yk) / h
    m = np.zeros(len(xk))
    m[0] = dl[0] if m0 is None else m0
    m[-1] = dl[-1] if m1 is None else m1
    for k in range(1, len(xk) - 1):
        if dl[k - 1] * dl[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k])
    k = min(max(int(np.searchsorted(xk, x, side="right") - 1), 0), len(xk) - 2)
    t = (x - xk[k]) / h[k]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * yk[k] + h10 * h[k] * m[k] + h01 * yk[k + 1] + h11 * h[k] * m[k + 1]


def poses_timed(route_px, key_frames, n, go, ease="inout", v0=0.0):
    """(x, y, heading) per frame for a timed route: route_px[0] is the car at the start (it
    leaves at frame go), route_px[i] is reached at key_frames[i - 1]. Distance along the
    spline follows a monotone curve through those times (with ease inout from v0 px/frame -
    0 = from rest - stopping at the last point), still before go and after the last key."""
    dense, seg, s = _spline(route_px)
    xs = [float(go)] + [float(k) for k in key_frames]
    ys = [0.0] + [float(s[min(i * PER, len(s) - 1)]) for i in range(1, len(route_px))]
    first = (ys[1] - ys[0]) / max(1e-6, xs[1] - xs[0])
    m0, m1 = (min(max(0.0, v0), 3.0 * first), 0.0) if ease == "inout" else (None, None)
    out = []
    for f in range(n):
        if f <= xs[0]:
            d = 0.0
        elif f >= xs[-1]:
            d = ys[-1]
        else:
            d = float(np.clip(_pchip(xs, ys, f, m0, m1), 0, ys[-1]))
        out.append(_along(dense, s, seg, d))
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
    (another car passing close) keep their own pixels - but not the car's own shadow (darker,
    same tint, within 36 px of the body)."""
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
    plf = pl.astype(np.float32)
    # the old car's own hard shadow beside its body: clearly darker than the road, about its
    # tint (H3's shadows run warm: (56,40,32) on (89,87,87) asphalt). Kept as "another object"
    # it left a dark crescent on the LTX plate (2026-10-11)
    bs, ps = base.sum(2), plf.sum(2)
    tint = np.abs(base / (bs[..., None] + 1) - plf / (ps[..., None] + 1)).max(2)
    shade = (bs < ps - 45) & (bs < 0.85 * ps) & (tint < 0.12) & (cv2.dilate(m, _disk(round(36 * sc))) > 0)
    other = (np.abs(base - plf).max(2) > 45) & ~tight & ~shade
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
    emit("PHASE", "tracking the camera")
    still = static_drift(frames)
    moving = still > MAX_CAMERA_DRIFT
    Hs, drift, lost = camera_track(frames) if moving else (None, still, 0)
    log(f"MEASURED camera drift {drift * 100:.2f}% of the width over the clip "
        f"({'moving' if moving else 'locked off'}); lost track in {lost} frame(s)")
    if moving and lost > max(3, 0.1 * n):
        raise RuntimeError(f"the camera moves in a way the edit cannot follow (lost track of the background in {lost} "
                           f"of {n} frames: too fast, too blurred or too little texture); pick a steadier shot")
    length = snap_up(n)
    W, H = canvas_for(fw, fh, length)
    emit("PROGRESS", 40)
    # the camera, in canvas pixels, one homography per frame on the H3 grid (held still when
    # locked off: no estimate jitter)
    Hc = to_canvas(Hs, fw, fh, W, H) if moving else [np.eye(3)] * n
    Hc = list(Hc) + [Hc[-1]] * (length - n)
    _save(os.path.join(work, "cam.json"), {"moving": moving, "H": [[round(float(v), 8) for v in m.ravel()] for m in Hc]})
    small = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames]
    small += [small[-1]] * (length - n)
    L.write_clip(os.path.join(work, "move_src.mp4"), small, FPS, audio_from=spec["clip"], lossless=True)
    del small
    emit("PROGRESS", 70)
    # the LTX clean-plate windows: the clip at the LTX canvas, 8k+1 frames each, silent track
    lw, lh = ltx_dims(fw, fh, min(LTX_PIXELS, max(fw * fh, 512 * 288)))  # no bigger than the source
    wins = ltx_windows(n)
    files = []
    for i, (s0, c) in enumerate(wins):
        name = f"ltx_src_{i}.mp4"
        wr = Writer(os.path.join(work, name), lw, lh, FPS, audio_from="silence", frames=c)
        for j in range(c):
            wr.write(cv2.resize(frames[min(s0 + j, n - 1)], (lw, lh), interpolation=cv2.INTER_AREA))
        wr.close()
        files.append(name)
    plan = {"mode": "move", "fps": FPS, "frames": n, "width": fw, "height": fh, "length": length,
            "canvas": [W, H], "camera_drift": round(drift, 4), "camera_moving": moving, "warnings": warnings,
            "ltx": {"canvas": [lw, lh], "windows": [[a, b] for a, b in wins], "files": files}}
    _save(os.path.join(work, "plan.json"), plan)
    log(f"MEASURED canvas {W}x{H}, {n} frames -> {length} on the H3 grid; LTX plate {lw}x{lh} in {len(wins)} window(s)")
    emit("PROGRESS", 100)


# ---------------------------------------------------------------- build

def _frame_of(spec, key_f, key_s, default):
    if spec.get(key_f) is not None:
        return int(spec[key_f])
    if spec.get(key_s) is not None:
        return int(round(float(spec[key_s]) * FPS))
    return default


def _route_keys(spec, n):
    """Timed route points as [(frame, x, y)] (x, y: 0-1 of THAT frame), or None for a plain
    route ([x, y] points in the start frame)."""
    raw = spec["route"]
    if not any(isinstance(q, dict) for q in raw):
        return None
    keys = []
    for q in raw:
        if not isinstance(q, dict):
            raise RuntimeError("the route mixes timed points ({at_s, x, y}) and plain ones ([x, y]); use one kind")
        f = int(q["frame"]) if q.get("frame") is not None else int(round(float(q["at_s"]) * FPS))
        keys.append((min(max(f, 0), n - 1), float(q["x"]), float(q["y"])))
    return sorted(keys)


def _v0(route, v_dir, speed, driving, hold):
    """The speed the drag starts at: the car's own, along the way the route starts (none if it
    turns back on itself, or after a wait)."""
    if not driving or hold or v_dir is None:
        return 0.0
    r_dir = math.atan2(route[1][1] - route[0][1], route[1][0] - route[0][0])
    return speed * max(0.0, math.cos(r_dir - v_dir))


def build(spec):
    work = spec["work_dir"]
    plan = _load(os.path.join(work, "plan.json"))
    W, H = plan["canvas"]
    n, length = plan["frames"], plan["length"]
    cam_path = os.path.join(work, "cam.json")
    cam = _load(cam_path) if os.path.exists(cam_path) else {"moving": False}
    moving = bool(cam.get("moving"))
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

    plate_path = os.path.join(work, "plate_ltx.mp4")
    if spec.get("plate_windows"):
        # the LTX clean plate, frame by frame (follows the camera; every car and shadow out)
        emit("PHASE", "stitching the clean plate")
        ltx = plan["ltx"]
        stitch_plate(spec["plate_windows"], [tuple(w) for w in ltx["windows"]], n, plate_path, tuple(ltx["canvas"]))
        rd = Reader(plate_path, (W, H))
        plate = rd.next().copy()
        rd.close()
        log(f"MEASURED plate: LTX clean plate, {len(spec['plate_windows'])} window(s)")
    else:
        if moving:
            raise RuntimeError("the camera moves in this clip and the LTX clean-plate models are not on the render "
                               "PC; without them move edits need a locked-off shot")
        if os.path.exists(plate_path):
            os.remove(plate_path)
        # other traffic (SAM's generic "car" pass): kept out of the median plate
        others = [np.zeros((H, W), bool) for _ in range(length)]
        for pth in spec.get("other_masks") or []:
            for f, m in enumerate(read_masks(pth, length, (W, H))):
                others[f] |= m
        write_masks(os.path.join(work, "othermask.mp4"), others)
        # a sample of up to ~96 frames is plenty for a median
        step = max(1, n // 96)
        idx = list(range(0, n, step))
        plate = clean_plate([frames[i] for i in idx], [old[i] for i in idx], px(60, W),
                            [others[i] for i in idx], px(30, W))
        log("MEASURED plate: masked temporal median")
    cv2.imwrite(os.path.join(work, "plate.png"), plate)
    emit("PROGRESS", 35)

    # the camera: G[f] takes start-frame pixels (the route's ground) to frame f's
    G = relative([np.array(h, float).reshape(3, 3) for h in cam["H"]], s0) if moving else None
    Gi = [np.linalg.inv(g) for g in G] if G else None
    to_world = (lambda f, x, y: warp_pt(Gi[f], x, y)) if G else (lambda f, x, y: (x, y))  # noqa: E731

    # route: from the car's centre at s0 through the given points, on the ground (start-frame
    # pixels). Plain points are 0-1 of the start frame; timed ones 0-1 of the frame at their time.
    m0 = old[s0]
    ang, (cx, cy) = car_axis(m0)
    # how fast the car is going at s0, on the ground (px/frame): a car already driving leaves
    # at that speed with no wait; a parked one waits hold_s (default 0.25 s) and pulls away
    a_f, b_f = (max(0, s0 - 6), s0) if s0 >= 3 else (s0, min(n - 1, s0 + 6))
    ca = centroid(tracks[k][a_f]) if tracks[k][a_f].any() else None
    cb = centroid(tracks[k][b_f]) if tracks[k][b_f].any() else None
    v_dir, speed = None, 0.0
    if ca and cb and b_f > a_f:
        wa, wb = to_world(a_f, *ca), to_world(b_f, *cb)
        speed = math.hypot(wb[0] - wa[0], wb[1] - wa[1]) / (b_f - a_f)
        v_dir = math.atan2(wb[1] - wa[1], wb[0] - wa[0])
    driving = speed > px(0.6, W)
    log(f"MEASURED car speed at the start: {speed:.1f} px/frame on the ground ({'driving' if driving else 'parked'})")
    hold = max(0, _frame_of(spec, "hold_frames", "hold_s", 0 if driving else int(round(0.25 * FPS))))
    keys = _route_keys(spec, n)
    if keys is None:
        pts = [(float(x) * W, float(y) * H) for x, y in spec["route"]]
        if pts and math.hypot(pts[0][0] - cx, pts[0][1] - cy) < 0.04 * W:
            pts = pts[1:]  # the agent gave the car's own position first
        if not pts:
            raise RuntimeError("the route needs at least one point after the car's position")
        route = [(cx, cy)] + pts
        arrive = _frame_of(spec, "arrive_frame", "arrive_s", n - 1)
        arrive = min(max(arrive, s0 + hold + 6), n - 1)
        ps = poses(route, length, s0, hold, arrive, spec.get("ease", "inout"), _v0(route, v_dir, speed, driving, hold))
    else:
        wk = [(f, *to_world(f, x * W, y * H)) for f, x, y in keys]
        if wk and wk[0][0] <= s0 + 2 and math.hypot(wk[0][1] - cx, wk[0][2] - cy) < 0.04 * W:
            wk = wk[1:]  # the car's own position at the start
        wk = [k for k in wk if k[0] > s0]
        if not wk:
            raise RuntimeError("the timed route needs at least one point after start_s")
        hold = min(hold, max(0, wk[0][0] - s0 - 4))
        frames_k, last = [], s0 + hold
        for f, _, _ in wk:  # strictly later than the one before
            last = max(f, last + 2)
            frames_k.append(min(last, length - 1))
        route = [(cx, cy)] + [(x, y) for _, x, y in wk]
        arrive = frames_k[-1]
        ps = poses_timed(route, frames_k, length, s0 + hold, spec.get("ease", "inout"),
                         _v0(route, v_dir, speed, driving, hold))
    turn = spec.get("turn", True) is not False
    # the nose: the way the car was going in the source (old track after s0, else before),
    # else the way the new route starts
    h_route = ps[min(length - 1, s0 + hold + 2)][2]
    h_src = None
    for f in list(range(s0 + 12, min(n, s0 + 25))) + list(range(max(0, s0 - 12), s0)):
        src_m = tracks[k][f] if f < len(tracks[k]) else None
        c = centroid(src_m) if src_m is not None and src_m.any() else None
        c = to_world(f, *c) if c else None
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
    pr = Reader(plate_path, (W, H)) if os.path.exists(plate_path) else None
    for f in range(length):
        pf = pr.next() if pr is not None else plate
        if f < s0:
            refs.append(frames[f])
            refmask.append(cv2.dilate(tracks[k][f].astype(np.uint8), k10) > 0)
            new.append(tracks[k][f])
            continue
        cov = cv2.GaussianBlur((cv2.dilate(old[f].astype(np.uint8), cover_k) > 0).astype(np.float32), (0, 0), 3)[..., None]
        bg = frames[f].astype(np.float32) * (1 - cov) + pf.astype(np.float32) * cov
        x, y, hd = ps[f]
        dth = 0.0
        if turn:
            ramp = min(1.0, max(f - s0 - hold, 0) / 8.0)
            dth = math.atan2(math.sin(hd - ang), math.cos(hd - ang)) * ramp
        M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(dth), 1.0)
        M[0, 2] += x - cx
        M[1, 2] += y - cy
        M3 = np.vstack([M, [0, 0, 1]])
        if G is not None:  # the ground pose, seen through the camera at frame f
            M3 = G[f] @ M3
        c = cv2.warpPerspective(car, M3, (W, H), flags=cv2.INTER_LINEAR)
        a = cv2.warpPerspective(alpha, M3, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        refs.append(np.clip(bg * (1 - a) + c * a, 0, 255).astype(np.uint8))
        nm = a[..., 0] > 0.5
        new.append(nm)
        refmask.append(cv2.dilate(nm.astype(np.uint8), k10) > 0)
        if f % 24 == 0:
            emit("PROGRESS", 40 + int(50 * f / length))
    if pr is not None:
        pr.close()
    L.write_clip(os.path.join(work, "move_ref.mp4"), refs, FPS, audio_from=os.path.join(work, "move_src.mp4"),
                 lossless=True)
    write_masks(os.path.join(work, "move_refmask.mp4"), refmask)
    write_masks(os.path.join(work, "newmask.mp4"), [m if f >= s0 else np.zeros_like(m) for f, m in enumerate(new)])
    write_masks(os.path.join(work, "oldmask.mp4"), old)
    cv2.imwrite(os.path.join(work, "first.png"), frames[0])
    drawn = []
    for f in range(length):
        c = centroid(new[f]) if f >= s0 and new[f].any() else None
        drawn.append(None if c is None else [round(c[0], 1), round(c[1], 1)])
    _save(os.path.join(work, "poses.json"), {"start": s0, "hold": hold, "arrive": arrive, "drawn": drawn, "track": k,
                                             "route": [[round(x, 1), round(y, 1)] for x, y in route],
                                             "camera_moving": moving})
    # the route picture, on the ground as the start frame sees it
    pic = frames[s0].copy()
    for f in range(s0, n, 3):
        cs, _ = cv2.findContours(tracks[k][f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if Gi is not None:
            cs = [cv2.perspectiveTransform(c.astype(np.float32), Gi[f]).astype(np.int32) for c in cs]
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
        d, miss, hits, off = [], 0, 0, 0
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
            off += dist > 2 * near
        if not hits:
            continue
        score = (float(np.mean(d)) + 40.0 * (miss + off) / max(1, len(d) + miss)) * BASE_W / w
        if best is None or score < best[0]:
            best = (score, k, float(np.mean(d)) * BASE_W / w, miss, off)
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
        score, k, dist, miss, off = pick
        results.append({"take": i + 1, "seed": spec["seeds"][i], "ok": True, "score": round(score, 1),
                        "path_error_px832": round(dist, 1), "missing_frames": miss, "off_path_frames": off,
                        "track": k})
        log(f"take {i + 1}: car track {k}, {dist:.1f} px off the drawn path (832 scale), {miss} frames missing, "
            f"{off} well off it")
    good = [r for r in results if r["ok"]]
    if not good:
        raise RuntimeError("H3 did not draw the car on the new path in any take; try a gentler route or another seed")
    best = min(good, key=lambda r: r["score"])["take"]

    old_small = read_masks(os.path.join(work, "oldmask.mp4"), length)
    plate_path = os.path.join(work, "plate_ltx.mp4")
    use_ltx = os.path.exists(plate_path)
    plate = None
    if not use_ltx:
        # the median plate again at the source's own size (a sample of frames + upscaled masks)
        emit("PHASE", "clean plate at full size")
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
        # and only near where the car was drawn (H3 runs ~30 px off it at 832 wide; a stray piece
        # on other traffic sits much further away): nothing else in the render gets pasted
        nm_path = os.path.join(work, "newmask.mp4")
        if os.path.exists(nm_path):
            reach = _disk(px(110, W))
            drawn_m = read_masks(nm_path, length)
            kept, dropped = [], 0
            for c, d in zip(car_small, drawn_m):
                if d.any() and c.any():
                    near_c = c & (cv2.dilate(d.astype(np.uint8), reach) > 0)
                    # the render's car mostly beyond reach (H3 kept it somewhere else): pasting
                    # the part inside left a sliver of car (2026-10-11); paste none of it
                    if near_c.sum() < 0.6 * c.sum():
                        near_c = np.zeros_like(c)
                        dropped += 1
                    c = near_c
                kept.append(c)
            car_small = kept
            if dropped:
                log(f"take {r['take']}: the render's car was off the drawn path in {dropped} frame(s): not pasted there")
                r["unpasted_frames"] = dropped
        dest = os.path.join(work, f"take{r['take']}.mp4")
        src, raw = Reader(spec["clip"]), Reader(spec["takes"][i])
        pv = Reader(plate_path) if use_ltx else None
        wr = Writer(dest, fw, fh, FPS, audio_from=spec["clip"], frames=n)
        pf = plate
        for f in range(n):
            fr = src.next()
            rf = raw.next()
            if pv is not None:
                pf = pv.next()
                if (pf.shape[1], pf.shape[0]) != (fw, fh):
                    pf = cv2.resize(pf, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
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
            out = paste_car(remove_old(fr, om, pf, fw), rf, cm, fw)
            wr.write(np.clip(out, 0, 255).astype(np.uint8))
            if f % 24 == 0:
                emit("PROGRESS", int(100 * (f + 1) / n))
        wr.close()
        src.close()
        raw.close()
        if pv is not None:
            pv.close()
        outs[r["take"]] = dest
    shutil.copyfile(outs[best], spec["out"])
    proof_sheet(spec["proof"], work, spec["clip"], outs[best], s0, n)
    report = {"mode": "move", "best_take": best, "takes": results, "canvas": [W, H], "frames": n,
              "start_frame": s0, "arrive_frame": pz["arrive"], "camera_drift": plan["camera_drift"],
              "camera_moving": bool(plan.get("camera_moving")), "plate": "ltx" if use_ltx else "median",
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
