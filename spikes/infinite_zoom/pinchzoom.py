"""Drive the door-zoom with real pinch gestures from a hand clip shot over a green tablet screen.

1. track thumb/index tips (MediaPipe HandLandmarker) -> per-frame pinch spread
2. zoom only grows while fingers spread (closing = lifting/resetting, ignored), with iOS-like
   momentum after release; the whole gesture track is scaled so the last pinch lands on the hero
3. render each zoom frame (warpzoom.frame_at), crop it to the screen's aspect, warp it into the
   tablet's green quad, key the hands out of the green and lay them back on top as dark silhouettes

usage: mpvenv python pinchzoom.py <zoom cfg.json> <hand clip.mp4> <out.mp4> [--track-only]
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

ZCFG, CLIP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TRACK_ONLY = "--track-only" in sys.argv
def _arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
CUT_S, TAIL_S, PUSH = _arg("--cut", 0.0), _arg("--tail", 1.6), _arg("--push", 1.0)
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "mpmodel", "hand_landmarker.task")

# --- read the hand clip --------------------------------------------------------------------------
cap = cv2.VideoCapture(CLIP)
fps = cap.get(cv2.CAP_PROP_FPS) or 24
frames = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
h0, w0 = frames[0].shape[:2]
OW = 1080
OH = int(round(OW * h0 / w0 / 2)) * 2
frames = [cv2.resize(f, (OW, OH), interpolation=cv2.INTER_LANCZOS4) for f in frames]
NF = len(frames)
print(f"clip {w0}x{h0} @ {fps:.2f} fps, {NF} frames -> {OW}x{OH}", flush=True)

# --- fingertip tracking --------------------------------------------------------------------------
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO,
    num_hands=2, min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))
track = []  # per frame: list of (thumb_xy, index_xy, wrist_xy)
for i, fr in enumerate(frames):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, int(i * 1000 / fps))
    hands = []
    for hl in res.hand_landmarks or []:
        pt = lambda k: np.array([hl[k].x * OW, hl[k].y * OH])  # noqa: E731
        hands.append((pt(4), pt(8), pt(0)))
    track.append(hands)

# per-frame spread channels: each hand's thumb-index distance, and index-to-index for two hands
def spreads(hands):
    d = {}
    for h in hands:
        key = "L" if h[2][0] < OW / 2 else "R"  # identify hands by side of the wrist
        d[key] = float(np.linalg.norm(h[0] - h[1]))
    if len(hands) == 2:
        d["2H"] = float(np.linalg.norm(hands[0][1] - hands[1][1]))
    return d

raw = [spreads(h) for h in track]
keys = sorted({k for d in raw for k in d})
inc = np.zeros(NF)
for k in keys:
    s = np.array([d.get(k, np.nan) for d in raw])
    # 5-frame median then EMA, only over contiguous runs where the channel exists
    sm = s.copy()
    for i in range(NF):
        win = s[max(0, i - 2):i + 3]
        win = win[~np.isnan(win)]
        sm[i] = np.median(win) if len(win) and not np.isnan(s[i]) else np.nan
    for i in range(1, NF):
        if not np.isnan(sm[i]) and not np.isnan(sm[i - 1]):
            sm[i] = 0.6 * sm[i] + 0.4 * sm[i - 1]
    for i in range(1, NF):
        a, b = sm[i - 1], sm[i]
        if np.isnan(a) or np.isnan(b) or a < 15:
            continue
        g = math.log(b / a)
        if g > 0.004:  # spreading; closing (lift/reset) is ignored
            inc[i] = max(inc[i], min(g, 0.08))

# momentum: when the spread stops, keep gliding and decay (iOS-like)
tau = 0.22 * fps
mom = np.zeros(NF)
v = 0.0
for i in range(NF):
    v = max(inc[i], v * math.exp(-1 / tau))
    mom[i] = v
L_hand = np.cumsum(mom)
print("pinch log-zoom total", round(float(L_hand[-1]), 3), "channels", keys, flush=True)

# --- zoom chain ----------------------------------------------------------------------------------
sys.argv = [sys.argv[0], ZCFG, OUT]
sys.path.insert(0, HERE)
import warpzoom as wz  # noqa: E402  (loads levels + quads)

seg = [math.log(1 / math.sqrt(wz.poly_area(q) / (wz.W * wz.H))) for q in wz.quads]
L_chain = sum(seg)
# finish the chain with ~0.6 s of clip left so the hero gets a hold under the resting hand
NCUT = int(round(CUT_S * fps)) if CUT_S else NF
end_i = int(np.searchsorted(L_hand[:NCUT], L_hand[NCUT - 1] * 0.995))
gain = L_chain / max(L_hand[end_i], 1e-6)
print(f"chain log-zoom {L_chain:.3f}, gain {gain:.2f}x, chain ends at frame {end_i}/{NF}", flush=True)
cum = np.concatenate([[0], np.cumsum(seg)])


def pos_from_L(L):
    L = min(max(L, 0.0), L_chain)
    i = min(int(np.searchsorted(cum, L, side="right") - 1), len(seg) - 1)
    return i + (L - cum[i]) / seg[i]


pos = [pos_from_L(L_hand[i] * gain) for i in range(NF)]
json.dump({"fps": fps, "pos": pos, "inc": inc.tolist(), "keys": keys,
           "tips": [[[h[0].tolist(), h[1].tolist()] for h in t] for t in track]},
          open(OUT.replace(".mp4", "_track.json"), "w"))
if TRACK_ONLY:
    sys.exit(0)

# --- tablet screen quad (median over the clip; hands occlude different parts each frame) -------
union = np.zeros((OH, OW), np.uint8)
for fr in frames[:NCUT]:
    union |= wz.green_mask(fr)
fake = np.zeros((OH, OW, 3), np.uint8)
fake[union > 0] = (0, 255, 0)
SQ, _ = wz.find_quad(fake)
SQ = SQ.astype(np.float32)
sw = (np.linalg.norm(SQ[1] - SQ[0]) + np.linalg.norm(SQ[2] - SQ[3])) / 2
sh = (np.linalg.norm(SQ[3] - SQ[0]) + np.linalg.norm(SQ[2] - SQ[1])) / 2
asp = sw / sh
print("screen quad", np.round(SQ).astype(int).tolist(), f"aspect {asp:.3f}", flush=True)
# crop the 9:16 zoom frame to the screen's aspect (centre crop), then map it onto the quad
if asp > wz.W / wz.H:
    cw, ch = wz.W, wz.W / asp
else:
    cw, ch = wz.H * asp, wz.H
crop = np.float32([[(wz.W - cw) / 2, (wz.H - ch) / 2], [(wz.W + cw) / 2, (wz.H - ch) / 2],
                   [(wz.W + cw) / 2, (wz.H + ch) / 2], [(wz.W - cw) / 2, (wz.H + ch) / 2]])
Mz = cv2.getPerspectiveTransform(crop, SQ)
scr = np.zeros((OH, OW), np.uint8)
cv2.fillPoly(scr, [np.round(SQ * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
scr = (scr.astype(np.float32) / 255)[..., None]


def hand_alpha(fr):
    b, g, r = [fr[..., i].astype(np.float32) for i in range(3)]
    key = np.clip((g - np.maximum(r, b) - 25) / 55, 0, 1)  # 1 = pure screen green
    a = 1 - key
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    return a[..., None]


def silhouette(fr):
    """Backlit-hand look: despill, desaturate, crush to near-black but keep a hint of skin."""
    f = fr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    f[..., 1] = np.minimum(g, (r + b) / 2)
    grey = f.mean(2, keepdims=True)
    f = 0.55 * f + 0.45 * grey
    return f * 0.32


nfo = NCUT + int(round(TAIL_S * fps))
ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                       "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", f"{fps}", "-i", "-",
                       "-i", CLIP, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "160k",
                       "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                       "-t", f"{nfo / fps:.3f}", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
for i in range(nfo):
    fr = frames[min(i, NCUT - 1)]
    p = pos[min(i, NCUT - 1)]
    fade = 1.0 if i < NCUT else max(0.0, 1 - (i - NCUT) / (0.6 * fps))
    if p >= len(wz.levels) - 1:
        tc = max(0, i - end_i) / fps  # slow push on the hero once the chain has landed
        zz = 1 + 0.035 * min(tc / 2.5, 1)
        M = cv2.getRotationMatrix2D((wz.W / 2, wz.H / 2), 0, zz)
        z = cv2.warpAffine(wz.levels[-1], M, (wz.W, wz.H), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    else:
        z = wz.frame_at(p)
    zw = cv2.warpPerspective(z, Mz, (OW, OH), flags=cv2.INTER_CUBIC).astype(np.float32)
    base = silhouette(fr)  # dark room: hands crossing the screen edge stay one tone
    out = base * (1 - scr) + zw * scr
    a = hand_alpha(fr) * scr * fade  # hands over the screen
    out = out * (1 - a) + silhouette(fr) * a
    if PUSH > 1:  # frame the tablet tighter, centred on the screen
        cx, cy = SQ[:, 0].mean(), SQ[:, 1].mean()
        Mp = np.float32([[PUSH, 0, OW / 2 - PUSH * cx], [0, PUSH, OH / 2 - PUSH * cy]])
        out = cv2.warpAffine(out, Mp, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ff.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
    if i % 48 == 0:
        print(f"frame {i}/{nfo} pos {p:.2f}", flush=True)
ff.stdin.close()
ff.wait()
print("done", OUT)
