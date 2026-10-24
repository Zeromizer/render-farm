"""Planar-door infinite zoom. Each level (except the last) contains a flat chroma-green "door"
(screen, billboard, lightbox). The next level is mapped into that quad by a homography, and the
camera flies into the quad: its viewport corners ease from the full frame onto the quad, so the
door ends up filling the screen exactly when the next level takes over. Seamless by construction.

usage: python warpzoom.py <cfg.json> <out.mp4>
cfg: {"levels": ["l1.png", "l2.png", ..., "hero.png"], "W": 1080, "H": 1920, "fps": 30,
      "sec_per_level": 2.2, "hold_start": 0.8, "hold_end": 2.2, "ease": 0.18,
      "quads": {"l1.png": [[x,y],...]}   # optional manual override (TL, TR, BR, BL)
      "only_frames": [..]}               # optional: write PNG stills instead of a video
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

cfg = json.load(open(sys.argv[1]))
base = os.path.dirname(os.path.abspath(sys.argv[1]))
OUT = sys.argv[2]
W, H, fps = cfg.get("W", 1080), cfg.get("H", 1920), cfg.get("fps", 30)
names = cfg["levels"]
N = len(names)


def load(p):
    im = cv2.imread(os.path.join(base, p), cv2.IMREAD_COLOR)
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) if im.shape[:2] != (H, W) else im


def green_mask(im):
    b, g, r = [im[..., i].astype(np.int32) for i in range(3)]
    return ((g > 110) & (g - np.maximum(r, b) > 55)).astype(np.uint8)


def find_quad(im):
    """Largest green blob -> 4 corners (TL, TR, BR, BL), pushed ~1.5 px outward to eat the fringe."""
    m = green_mask(im)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise RuntimeError("no green door found")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.08, 60):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            break
    else:
        # fall back to the extreme points along the diagonals
        pts = hull[:, 0, :].astype(np.float32)
        s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
        ap = np.array([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])[:, None]
    q = ap[:, 0, :].astype(np.float32)
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # clockwise from top-left in image coords: TL, TR, BR, BL
    q = np.roll(q, -int(np.argmin(q.sum(1))), axis=0)
    # refine: fit a line to the blob's contour points along each side (skipping the rounded
    # corners) and intersect neighbours, so the quad hugs the straight edges, not the corners
    pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    lines = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        t = ((pts - a) @ d) / (L * L)
        dist = np.abs(np.cross(d, pts - a)) / L
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.04 * L + 4)]
        if len(sel) < 10:
            lines.append((a, d / L))
            continue
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01)[:, 0]
        lines.append((np.float32([x0, y0]), np.float32([vx, vy])))
    ref = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
        try:
            u = np.linalg.solve(A, (p2 - p1).astype(np.float64))
            ref.append(p1 + u[0] * d1)
        except np.linalg.LinAlgError:
            ref.append(q[k])
    q = np.float32(ref)
    c = q.mean(0)
    q = c + (q - c) * (1 + 2.0 / np.linalg.norm(q - c, axis=1, keepdims=True))
    fill = blob.sum() / max(1, cv2.contourArea(q.astype(np.float32)))
    return q, fill


def despill(im, q, near_px=0.25, far_px=0.7):
    """Neutralise the chroma panel's green glow: average-despill (g <= (r+b)/2) at full strength
    within near_px*H of the door, fading out by far_px*H, so foliage further away keeps its green."""
    inside = np.ones(im.shape[:2], np.uint8)
    cv2.fillPoly(inside, [q.astype(np.int32)], 0)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5) / im.shape[0]
    wgt = np.clip((far_px - dist) / (far_px - near_px), 0, 1)
    out = im.astype(np.float32)
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    cap = (r + b) / 2 + 4
    out[..., 1] = np.where(g > cap, g - (g - cap) * wgt, g)
    return np.clip(out, 0, 255).astype(np.uint8)


levels = [load(p) for p in names]
RECT = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
quads = []
for i in range(N - 1):
    over = cfg.get("quads", {}).get(names[i])
    if over:
        q, fill = np.float32(over), 1.0
    else:
        q, fill = find_quad(levels[i])
    quads.append(q)
    levels[i] = despill(levels[i], q)
    print(f"{names[i]} door {np.round(q).astype(int).tolist()} fill {fill:.3f}", flush=True)

# P[i]: level i+1 coords -> level i coords
P = [cv2.getPerspectiveTransform(RECT, q) for q in quads]
# mip pyramids (each step halves)
pyr = []
for im in levels:
    p = [im]
    while min(p[-1].shape[:2]) > 64:
        p.append(cv2.pyrDown(p[-1]))
    pyr.append(p)


def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def draw(canvas, j, S):
    """Composite level j given S: screen -> level-j coords; recurse into its door."""
    corners = cv2.perspectiveTransform(RECT[None], np.linalg.inv(S))[0]  # level frame on screen
    area = poly_area(corners)
    if area < 64:
        return
    shrink = math.sqrt(W * H / area)  # level px per screen px (approx.)
    lvl = int(np.clip(math.floor(math.log2(max(shrink, 1.0))), 0, len(pyr[j]) - 1))
    src = pyr[j][lvl]
    D = np.diag([0.5 ** lvl, 0.5 ** lvl, 1.0]) @ S  # screen -> mip coords
    layer = cv2.warpPerspective(src, D, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
    if canvas is None:
        canvas = layer.astype(np.float32)
    else:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(corners * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        a = (m.astype(np.float32) / 255)[..., None]
        canvas[:] = canvas * (1 - a) + layer * a
    if j + 1 < N:
        draw_into = canvas
        draw(draw_into, j + 1, np.linalg.inv(P[j]) @ S)
    return canvas


def frame_at(pos):
    """pos in [0, N-1]: integer part = level, fraction = progress into its door."""
    i = min(int(math.floor(pos)), N - 1)
    u = pos - i
    S = np.eye(3)
    if i < N - 1 and u > 0:
        q = quads[i]
        a = math.sqrt(poly_area(q) / (W * H))
        p = (1 - a ** u) / (1 - a)  # viewport shrinks geometrically -> constant-feeling zoom
        V = ((1 - p) * RECT + p * q).astype(np.float32)
        S = cv2.getPerspectiveTransform(RECT, V)  # screen -> level i
    canvas = np.zeros((H, W, 3), np.float32)
    first = cv2.warpPerspective(levels[i], S, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    canvas[:] = first
    if i + 1 < N:
        draw(canvas, i + 1, np.linalg.inv(P[i]) @ S)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def pos_at(t):
    hs, spl, he = cfg.get("hold_start", 0.8), cfg.get("sec_per_level", 2.2), cfg.get("hold_end", 2.2)
    run = (N - 1) * spl
    u = min(max((t - hs) / run, 0), 1)
    e = cfg.get("ease", 0.18)
    v = 1 / (1 - e)
    if u < e:
        p = v * u * u / (2 * e)
    elif u > 1 - e:
        p = 1 - v * (1 - u) ** 2 / (2 * e)
    else:
        p = v * (u - e / 2)
    return (N - 1) * p


def creep(t, img):
    """Slow push on the final hold so the last frame isn't frozen."""
    hs, spl, he = cfg.get("hold_start", 0.8), cfg.get("sec_per_level", 2.2), cfg.get("hold_end", 2.2)
    t0 = hs + (N - 1) * spl
    if t <= t0:
        return img
    z = 1 + 0.04 * min((t - t0) / he, 1)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


total = cfg.get("hold_start", 0.8) + (N - 1) * cfg.get("sec_per_level", 2.2) + cfg.get("hold_end", 2.2)
nf = int(round(total * fps))
only = cfg.get("only_frames")
ff = None
if not only:
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
for f in range(nf):
    if only and f not in only:
        continue
    t = f / fps
    pos = pos_at(t)
    ns, shutter = int(cfg.get("blur_samples", 1)), float(cfg.get("shutter", 0.5))
    acc = None
    for k in range(ns):
        tk = t + (k / max(ns - 1, 1) - 0.5) * shutter / fps if ns > 1 else t
        pk = pos_at(tk)
        im = creep(tk, levels[N - 1].copy()) if pk >= N - 1 else frame_at(pk)
        acc = im.astype(np.float32) if acc is None else acc + im
    img = np.clip(acc / ns + 0.5, 0, 255).astype(np.uint8)
    if only:
        cv2.imwrite(OUT.replace(".mp4", f"_f{f:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        ff.stdin.write(img.tobytes())
    if f % 60 == 0:
        print(f"frame {f}/{nf} pos {pos:.2f}", flush=True)
if ff:
    ff.stdin.close()
    ff.wait()
print("done", OUT, f"{total:.1f}s {nf} frames")
