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
ONLY_RIGHT = "--only-right" in sys.argv
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
track = []  # per frame: list of 21x2 landmark arrays (px)
for i, fr in enumerate(frames):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, int(i * 1000 / fps))
    hands = []
    for hl in res.hand_landmarks or []:
        hands.append(np.array([[p.x * OW, p.y * OH] for p in hl]))
    track.append(hands)

# per-frame spread channels: each hand's thumb-index distance, and index-to-index for two hands
def spreads(hands):
    d = {}
    for h in hands:
        key = "L" if h[0][0] < OW / 2 else "R"  # identify hands by side of the wrist
        d[key] = float(np.linalg.norm(h[4] - h[8]))
    if len(hands) == 2 and not ONLY_RIGHT:
        d["2H"] = float(np.linalg.norm(hands[0][8] - hands[1][8]))
    if ONLY_RIGHT:
        d = {k: v for k, v in d.items() if k == "R"}
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
           "tips": [[[h[4].tolist(), h[8].tolist()] for h in t] for t in track]},
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


def hand_region(h, finger_px=38):
    """Hand hull + a forearm strip running from the wrist away from the fingers, as 0..1 float."""
    m = np.zeros((OH, OW), np.uint8)
    wr, mid = h[0], h[9]
    d = (wr - mid) / max(np.linalg.norm(wr - mid), 1)
    perp = np.array([-d[1], d[0]])
    w = 0.75 * max(np.linalg.norm(h[5] - h[17]), 60)
    far = wr + d * 2500
    arm = np.array([wr + perp * w, wr - perp * w, far - perp * w * 1.3, far + perp * w * 1.3])
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.vstack([h, arm[:2]])).astype(np.int32)), 255)
    cv2.fillConvexPoly(m, np.round(arm).astype(np.int32), 255)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * finger_px + 1,) * 2))
    return m.astype(np.float32) / 255


LM = None
if ONLY_RIGHT:
    LM, RM = [], []
    lastL = lastR = None
    for t in track[:NCUT]:
        L = [h for h in t if h[0][0] < OW / 2]
        R = [h for h in t if h[0][0] >= OW / 2]
        lastL = hand_region(L[0], 48) if L else (cv2.dilate(lastL, np.ones((15, 15), np.uint8)) if lastL is not None else np.zeros((OH, OW), np.float32))
        lastR = hand_region(R[0], 10) if R else (lastR if lastR is not None else np.zeros((OH, OW), np.float32))
        LM.append(lastL)
        RM.append(lastR)
    # clean plate: per-pixel median over frames where neither hand covers the pixel
    idx = list(range(0, NCUT, 3))
    plate = np.zeros((OH, OW, 3), np.float32)
    hole = np.zeros((OH, OW), np.uint8)
    for y0 in range(0, OH, 96):
        y1 = min(OH, y0 + 96)
        st = np.stack([frames[i][y0:y1].astype(np.float32) for i in idx])
        bad = np.stack([(LM[i][y0:y1] > 0.02) | (RM[i][y0:y1] > 0.02) for i in idx])[..., None]
        st[np.broadcast_to(bad, st.shape)] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(st, axis=0)
        hole[y0:y1] = np.isnan(med[..., 0]).astype(np.uint8)
        plate[y0:y1] = np.nan_to_num(med)
    plate = cv2.inpaint(np.clip(plate, 0, 255).astype(np.uint8), hole * 255, 15, cv2.INPAINT_TELEA)
    print(f"clean plate: {int(hole.sum())} px never uncovered (inpainted)", flush=True)
    cv2.imwrite(OUT.replace(".mp4", "_plate.png"), plate)
    LM = [np.clip(cv2.GaussianBlur(lm, (0, 0), 6) * (1 - (rm > 0.5)), 0, 1)[..., None] for lm, rm in zip(LM, RM)]

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
    lm = LM[min(i, NCUT - 1)] if LM is not None else 0.0
    if LM is not None:
        base = base * (1 - lm) + silhouette(plate) * lm
    out = base * (1 - scr) + zw * scr
    a = hand_alpha(fr) * scr * fade * (1 - lm)  # hands over the screen (minus a removed hand)
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
