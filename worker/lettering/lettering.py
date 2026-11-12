"""Lettering fix, pixel side: correct number plates and badges on a generated clip by
letting MiniMax H3 propagate the REAL lettering from a reference photo.

Why this and not a corner pin (planar_patch): a pasted artwork never sits in the
footage the way the model's own pixels do (perspective drift through a turn, sharp
letters on motion blur). H3 cannot spell from a prompt, but it holds a video
consistent: whatever lettering is visible on the plate in some frames, it carries
into the masked ones. So the real lettering is pasted where the car's pose matches
the reference (the "anchor" frames, left unmasked), the element is masked
everywhere else, and H3 redraws it there with its own perspective, blur and light.
Proven on an ATTO 1 take 2026-10-10: 5/5 seeds read "BYD ATTO 1" through the
turn-in; text in the prompt alone gave the wrong font or garbled glyphs.

Two subcommands, both driven by a spec JSON the runner writes:

  prep     align the reference to the anchor frame, track every element, paste the
           real lettering on the anchor run, cut a follow-crop window (the elements
           stay large and nearly still on the generation canvas), write the H3
           inputs (source / video mask / audio mask) and plan.json
  compose  composite each H3 take back over the anchored clip, score the takes
           against the reference lettering, write the best one, every take, a proof
           sheet and a report

Run in the planar_patch venv (opencv-python-headless + numpy); ffmpeg from PATH or
the worker's usual lookup.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract"))
from common import find_ffmpeg_tool  # noqa: E402

FPS = 24
MAX_WINDOW = 362          # H3's trained range tops out near 15 s at 24 fps
CANVAS_AREA = 960 * 544   # follow-crop canvas: ~0.52 MP, plenty for a 16 GB card at 15 s
KEEP_ANCHOR_MIN = 8       # shorter anchor runs are rebuilt from their neighbours in compose
                          # (two of H3's ~4-frame latent groups; see prep)
# Multi-anchor mode (anchor_mode "multi"): when the car never holds the reference pose
# (driving at or away from the camera), the real lettering is warped into short runs
# spread over the window instead of one settled run. The layout is the one the Ole 911
# spike measured (2026-11-07, farm job 64fddd5c): 8 frames at window offset 17j+5..17j+12,
# i.e. aligned to H3's latent groups so the max-pooled latent mask leaves them visible.
ANCHOR_PERIOD = 17
ANCHOR_OFFSET = 5
ANCHOR_LEN = 8


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


# ---------------------------------------------------------------- io

def read_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode any frame from {path}")
    return frames, fps


def write_clip(path, frames, fps, audio_from=None, audio_start=0.0, lossless=False):
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-"]
    if audio_from:
        cmd += ["-ss", f"{audio_start:.4f}", "-i", audio_from, "-map", "0:v:0", "-map", "1:a?"]
    if lossless:
        cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if audio_from:
        # -t, not -shortest: a source whose audio ends a hair early must not cost the
        # clip its last frame (a 158-frame take came back as 157)
        cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{len(frames) / fps:.4f}"]
    cmd += [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def has_audio(path):
    ffprobe = find_ffmpeg_tool("ffprobe")
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(r.stdout.strip())


def silent_wav(path, seconds):
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=stereo", "-t", f"{seconds:.4f}", path], check=True)


# ---------------------------------------------------------------- geometry

def snap_len(n):
    """Largest 17k+5 <= n (H3's frame grid)."""
    return max(5, n - ((n - 5) % 17))


def gen_dims(aspect, area=CANVAS_AREA):
    w = math.sqrt(area * aspect)
    return max(256, int(round(w / 32)) * 32), max(256, int(round(w / aspect / 32)) * 32)


def window_for(n_frames, anchor):
    """(start, length): the largest 17k+5 window inside the clip that contains the anchor,
    preferring to keep the anchor at the end it is nearest to."""
    length = min(snap_len(n_frames), MAX_WINDOW)
    if anchor >= n_frames - length:
        return n_frames - length, length
    if anchor < length:
        return 0, length
    return anchor - length + 1, length


def box_of(quad):
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def clamp_box(b, w, h):
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]


# ---------------------------------------------------------------- reference alignment

def align_reference(ref, frame, boxes_ref, log):
    """Homography reference -> frame. AKAZE features + RANSAC for the gross fit (the
    reference may be another size, crop or aspect), then ECC refined on the elements'
    neighbourhood. Returns (H, ecc_score, inliers)."""
    fh, fw = frame.shape[:2]
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # AKAZE in OpenCV 4 (the worker venv pins <5); OpenCV 5 moved it out of the main
    # namespace, so fall back to ORB there (both binary descriptors -> Hamming).
    make = getattr(cv2, "AKAZE_create", None) or (lambda: cv2.ORB_create(4000))
    det = make()
    k1, d1 = det.detectAndCompute(rg, None)
    k2, d2 = det.detectAndCompute(fg, None)
    H, inliers = None, 0
    if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.8 * n.distance]
        if len(good) >= 8:
            src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 10:
        # same framing (the usual i2v case: the reference IS the last frame): plain resize
        log(f"feature fit weak ({inliers} inliers); assuming the reference is the frame resized")
        H = np.array([[fw / ref.shape[1], 0, 0], [0, fh / ref.shape[0], 0], [0, 0, 1]], np.float64)
    # ECC refine on the union of the elements (+ margin) in FRAME coords
    quads = [cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), H)
             for b in boxes_ref]
    u = box_of(np.concatenate([q.reshape(-1, 2) for q in quads]))
    m = max(40.0, 0.6 * max(u[2] - u[0], u[3] - u[1]))
    u = clamp_box([u[0] - m, u[1] - m, u[2] + m, u[3] + m], fw, fh)
    warped = cv2.warpPerspective(rg, H, (fw, fh), flags=cv2.INTER_LINEAR)
    tmask = np.zeros((fh, fw), np.uint8)
    tmask[int(u[1]):int(u[3]), int(u[0]):int(u[2])] = 255
    try:
        cc, W = cv2.findTransformECC(fg.astype(np.float32), warped.astype(np.float32), np.eye(3, dtype=np.float32),
                                     cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                                     tmask, 3)
        # template = frame, input = warped ref: warped(W x) ~ frame(x)  ->  frame = W^-1 applied to warped
        H = np.linalg.inv(W.astype(np.float64)) @ H
    except cv2.error:
        cc = -1.0
    return H / H[2, 2], float(cc), inliers


# ---------------------------------------------------------------- tracking

def track_motion(grays, anchor, roi, lo, hi):
    """Similarity transform (scale, rotation, translation) anchor -> frame for every frame
    the car stays trackable, chained frame to frame from LK flow on features around the
    elements. Scale matters: a push-in or a car driving at the camera can double the
    lettering's size over a clip. Returns {frame: 3x3}."""
    H, W = grays[0].shape
    corners = np.float32([[roi[0], roi[1]], [roi[2], roi[1]], [roi[2], roi[3]], [roi[0], roi[3]]]).reshape(-1, 1, 2)
    lk = {"winSize": (21, 21), "maxLevel": 3,
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    out = {anchor: np.eye(3)}
    for rng in (range(anchor - 1, lo - 1, -1), range(anchor + 1, hi + 1)):
        T, prev = np.eye(3), anchor
        for i in rng:
            r = clamp_box(box_of(cv2.perspectiveTransform(corners, T)), W, H)
            if r[2] - r[0] < 24 or r[3] - r[1] < 24:
                break
            m = np.zeros((H, W), np.uint8)
            m[int(r[1]):int(r[3]), int(r[0]):int(r[2])] = 255
            pts = cv2.goodFeaturesToTrack(grays[prev], 400, 0.01, 5, mask=m)
            if pts is None or len(pts) < 12:
                break
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[prev], grays[i], pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[i], grays[prev], nxt, None, **lk)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pts - back).reshape(-1, 2), axis=1) < 1.0)
            if ok.sum() < 12:
                break
            A, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if A is None or int(inl.sum()) < 10:
                break
            T = np.vstack([A, [0, 0, 1]]) @ T
            out[i], prev = T.copy(), i
    return out


def _peak(res, loc):
    """Sub-pixel peak of a matchTemplate response (parabola through the neighbours)."""
    x, y = loc
    def off(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dx = off(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    dy = off(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    return x + dx, y + dy


def track_element(grays, motion, anchor, box, lo, hi):
    """Each element rides the car's motion, then a local template match (template updated
    every frame, cut sub-pixel at the predicted size) absorbs parallax and pose change.
    The correction carries forward, so an element on another plane (badge on the bonnet,
    plate on the bumper) does not drift. Returns {frame: (x0, y0, x1, y1, score)}."""
    H, W = grays[0].shape
    q0 = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)
    out = {anchor: (*box, 1.0)}
    for rng in (range(anchor - 1, lo - 1, -1), range(anchor + 1, hi + 1)):
        cx = cy = 0.0
        prev, pc = anchor, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for i in rng:
            if i not in motion:
                break
            p = box_of(cv2.perspectiveTransform(q0, motion[i]))
            pw, ph = p[2] - p[0], p[3] - p[1]
            mx, my = (p[0] + p[2]) / 2 + cx, (p[1] + p[3]) / 2 + cy
            if mx - pw / 2 <= 0 or my - ph / 2 <= 0 or mx + pw / 2 >= W - 1 or my + ph / 2 >= H - 1 or pw < 6 or ph < 4:
                break   # touching the frame edge: entering/leaving, not fully visible -> leave it alone
            w, h = int(round(pw)), int(round(ph))
            # previous frame's patch, resampled to this frame's scale/rotation (sub-pixel)
            D = motion[prev][:2, :2] @ np.linalg.inv(motion[i][:2, :2])
            c0 = np.array([(w - 1) / 2, (h - 1) / 2])
            M = np.hstack([D, (np.array(pc) - D @ c0)[:, None]]).astype(np.float32)
            tpl = cv2.warpAffine(grays[prev], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            sx, sy = max(6, int(0.15 * w)), max(4, int(0.3 * h))
            ox, oy = int(round(mx - w / 2)) - sx, int(round(my - h / 2)) - sy
            score = 0.0
            if ox >= 0 and oy >= 0 and ox + w + 2 * sx <= W and oy + h + 2 * sy <= H and float(tpl.std()) >= 4.0:
                res = cv2.matchTemplate(grays[i][oy:oy + h + 2 * sy, ox:ox + w + 2 * sx], tpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if score >= 0.5:   # else: flat or lost -> ride the motion alone this frame
                    lx, ly = _peak(res, loc)
                    nx, ny = ox + lx + (w - 1) / 2, oy + ly + (h - 1) / 2
                    cx, cy = cx + nx - mx, cy + ny - my
                    mx, my = nx, ny
            out[i] = tuple(float(v) for v in (mx - pw / 2, my - ph / 2, mx + pw / 2, my + ph / 2, score))
            prev, pc = i, (mx, my)
    return out


def motion_scale(T):
    return math.sqrt(abs(np.linalg.det(T[:2, :2])))


# ---------------------------------------------------------------- multi-anchor

def _quad(b, pad=0.0):
    return np.float32([[b[0] - pad, b[1] - pad], [b[2] + pad, b[1] - pad],
                       [b[2] + pad, b[3] + pad], [b[0] - pad, b[3] + pad]]).reshape(-1, 1, 2)


def plan_multi_runs(lo, hi, tracks, fw, fh, every=1, min_w=56.0, min_h=10.0):
    """Anchor runs on H3's latent grid inside the window [lo, hi]: 8 frames at window offset
    17j+5. A run is kept only when every element is tracked, fully inside the frame and big
    enough to carry legible lettering on all 8 frames (56 px wide by default: below that the
    letters are blobs, and on a car still in the dark a pasted plate floats; NIO s1, 2026-11-12).
    every=2 keeps every other slot."""
    runs, j = [], 0
    while True:
        s = lo + ANCHOR_PERIOD * j + ANCHOR_OFFSET
        e = s + ANCHOR_LEN - 1
        if e > hi:
            break
        if j % max(1, every) == 0:
            ok = True
            for t in tracks:
                for i in range(s, e + 1):
                    p = t.get(i)
                    if (p is None or p[0] < 1 or p[1] < 1 or p[2] > fw - 2 or p[3] > fh - 2
                            or p[2] - p[0] < min_w or p[3] - p[1] < min_h):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                runs.append([s, e])
        j += 1
    return runs


def element_homography(grays, anchor, i, box, tbox):
    """Homography anchor-frame coords -> frame i for the plane an element sits on.
    Seeded with the similarity implied by the tracked boxes (it carries the scale change of a
    car driving at or away from the camera, which a cold ECC cannot find), then refined by
    ECC on the element's neighbourhood in frame i. Returns (H, ecc); ecc -1 = seed kept."""
    S = _box_similarity(tbox[:4], box)            # anchor -> i
    if i == anchor:
        return np.eye(3), 1.0
    fh, fw = grays[0].shape
    w, h = tbox[2] - tbox[0], tbox[3] - tbox[1]
    m = max(30.0, 0.6 * max(w, h))
    x0, y0, x1, y1 = [int(round(v)) for v in clamp_box([tbox[0] - m, tbox[1] - m, tbox[2] + m, tbox[3] + m], fw, fh)]
    if x1 - x0 < 24 or y1 - y0 < 24:
        return S, -1.0
    tpl = cv2.GaussianBlur(grays[i][y0:y1, x0:x1], (0, 0), 1.0).astype(np.float32)
    inp = cv2.GaussianBlur(grays[anchor], (0, 0), 1.0).astype(np.float32)
    Tsh = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    W0 = np.linalg.inv(S) @ Tsh                    # crop-of-i coords -> anchor coords
    W0 = (W0 / W0[2, 2]).astype(np.float32)
    try:
        cc, W = cv2.findTransformECC(tpl, inp, W0, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5), None, 3)
    except cv2.error:
        return S, -1.0
    H = np.linalg.inv(W.astype(np.float64) @ np.linalg.inv(Tsh))
    H /= H[2, 2]
    q = _quad(box)
    d = float(np.abs(cv2.perspectiveTransform(q, H) - cv2.perspectiveTransform(q, S)).max())
    if cc < 0.6 or d > 0.35 * max(w, 1.0):
        return S, float(cc)
    return H, float(cc)


def smooth_run(Hs, box):
    """Fit each corner of the element's quad with a quadratic over the run, so the pasted
    lettering does not jitter frame to frame (per-frame ECC is noisy at plate size)."""
    idx = sorted(Hs)
    if len(idx) < 4:
        return Hs
    q = _quad(box)
    pts = np.stack([cv2.perspectiveTransform(q, Hs[i]).reshape(4, 2) for i in idx])   # n x 4 x 2
    t = np.array(idx, np.float64) - idx[0]
    fit = np.empty_like(pts)
    for c in range(4):
        for d in range(2):
            fit[:, c, d] = np.polyval(np.polyfit(t, pts[:, c, d], 2), t)
    src = q.reshape(4, 2).astype(np.float32)
    return {i: cv2.getPerspectiveTransform(src, fit[k].astype(np.float32)).astype(np.float64)
            for k, i in enumerate(idx)}


def _velocity(track, i):
    def c(p):
        return np.array([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
    a, b = track.get(i - 1), track.get(i + 1)
    if a is not None and b is not None:
        return (c(b) - c(a)) / 2
    if b is not None:
        return c(b) - c(track[i])
    if a is not None:
        return c(track[i]) - c(a)
    return np.zeros(2)


def paste_element(frame, ref_warp, anchor_img, H, box, vel, pad=4.0, shutter=0.5):
    """Warp the reference lettering (aligned to the anchor frame) into this frame along H.
    Gain-matched per channel to the frame's light from a ring around the element (warped
    anchor vs this frame), motion-blurred along the element's own velocity with a 180-degree
    shutter, so the anchors sit in the clip instead of on it."""
    fh, fw = frame.shape[:2]
    qi = cv2.perspectiveTransform(_quad(box, pad), H).reshape(4, 2)
    blur = shutter * float(np.hypot(*vel))
    m = int(math.ceil(blur)) + 12
    x0 = max(0, int(qi[:, 0].min()) - m); y0 = max(0, int(qi[:, 1].min()) - m)
    x1 = min(fw, int(math.ceil(qi[:, 0].max())) + m); y1 = min(fh, int(math.ceil(qi[:, 1].max())) + m)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    cw, ch = x1 - x0, y1 - y0
    Hc = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ H
    patch = cv2.warpPerspective(ref_warp, Hc, (cw, ch), flags=cv2.INTER_AREA if np.linalg.det(H[:2, :2]) < 1 else cv2.INTER_CUBIC).astype(np.float32)
    orig = cv2.warpPerspective(anchor_img, Hc, (cw, ch), flags=cv2.INTER_LINEAR).astype(np.float32)
    mk = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(mk, [np.round(qi - [x0, y0]).astype(np.int32)], 255)
    r = max(3, int(round(0.25 * (qi[:, 1].max() - qi[:, 1].min()))))
    ring = (cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0) & (mk == 0)
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    if ring.sum() > 20:
        gain = (crop[ring].mean(0) + 1.0) / (orig[ring].mean(0) + 1.0)
        patch *= np.clip(gain, 0.2, 2.0)
    a = cv2.GaussianBlur(mk, (0, 0), 1.2).astype(np.float32) / 255.0
    if blur >= 1.0:
        n = int(math.ceil(blur)) | 1
        k = np.zeros((n, n), np.float32)
        ang = math.atan2(vel[1], vel[0])
        cx = cy = n // 2
        dx, dy = math.cos(ang) * (n // 2), math.sin(ang) * (n // 2)
        cv2.line(k, (int(round(cx - dx)), int(round(cy - dy))), (int(round(cx + dx)), int(round(cy + dy))), 1.0, 1)
        k /= max(k.sum(), 1e-6)
        patch = cv2.filter2D(patch, -1, k)
        a = cv2.filter2D(a, -1, k)
    a = a[..., None]
    frame = frame.copy()
    frame[y0:y1, x0:x1] = np.clip(crop * (1 - a) + patch * a, 0, 255).astype(np.uint8)
    return frame


def _mask_boxes(tracks, i, fw, fh, pad_frac):
    boxes = []
    for t in tracks:
        p = t.get(i)
        if p is None:
            continue
        w, h = p[2] - p[0], p[3] - p[1]
        pad = max(10.0, pad_frac * max(w, h) * 0.5 + 0.15 * h)
        boxes.append(clamp_box([p[0] - pad, p[1] - pad, p[2] + pad, p[3] + pad], fw, fh))
    return boxes


# ---------------------------------------------------------------- prep

def prep(spec):
    emit("PHASE", "decoding")
    frames, fps = read_clip(spec["clip"])
    n = len(frames)
    fh, fw = frames[0].shape[:2]
    ref = cv2.imread(spec["reference"], cv2.IMREAD_COLOR)
    if ref is None:
        raise RuntimeError("cannot read the reference image")
    anchor = int(spec.get("anchor_frame", -1))
    anchor = n - 1 if anchor < 0 else min(anchor, n - 1)
    elements = spec["elements"]
    log = lambda s: print(s, flush=True)
    print(f"{n} frames {fw}x{fh} @ {fps:.3f}, anchor {anchor}", flush=True)

    emit("PHASE", "aligning reference")
    boxes_ref = [e["box"] for e in elements]
    H, cc, inliers = align_reference(ref, frames[anchor], boxes_ref, log)
    print(f"MEASURED align ecc={cc:.3f} inliers={inliers}", flush=True)
    if cc < float(spec.get("min_align", 0.8)):
        raise RuntimeError(f"the reference does not line up with frame {anchor} of the clip (alignment score "
                           f"{cc:.2f} < {spec.get('min_align', 0.8)}). Pass the frame whose pose matches the "
                           f"reference as anchor_frame, or a reference of the same shot.")
    ref_warp = cv2.warpPerspective(ref, H, (fw, fh), flags=cv2.INTER_CUBIC)

    # element boxes on the anchor frame
    anchor_boxes = []
    for e in elements:
        b = e["box"]
        q = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), H)
        anchor_boxes.append(clamp_box(box_of(q), fw, fh))

    start, length = window_for(n, anchor)
    lo, hi = start, start + length - 1
    emit("PHASE", "tracking elements")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    u = box_of(np.array([[b[0], b[1]] for b in anchor_boxes] + [[b[2], b[3]] for b in anchor_boxes]))
    m = max(40.0, 0.6 * max(u[2] - u[0], u[3] - u[1]))
    motion = track_motion(grays, anchor, clamp_box([u[0] - m, u[1] - m, u[2] + m, u[3] + m], fw, fh), lo, hi)
    tracks = [track_element(grays, motion, anchor, b, lo, hi) for b in anchor_boxes]
    scales = {i: motion_scale(T) for i, T in motion.items()}
    win = range(lo, hi + 1)
    print(f"MEASURED motion frames={len(motion)} scale {min(scales.values()):.2f}-{max(scales.values()):.2f}", flush=True)
    for e, t in zip(elements, tracks):
        sc = [v[4] for v in t.values()]
        print(f"MEASURED track {e['name']} frames={len(t)} min_score={min(sc):.2f} "
              f"weak={sum(1 for s in sc if s < 0.5)}", flush=True)

    # anchor run: contiguous frames around the anchor where every element's box (all four
    # edges, so size too) sits within `settle` px of the anchor box -> the real lettering
    # is pasted there, unmasked
    settle = float(spec.get("settle_px", 2.5))
    def settled(i):
        for k, b in enumerate(anchor_boxes):
            p = tracks[k].get(i)
            if p is None or max(abs(p[j] - b[j]) for j in range(4)) > settle:
                return False
        return True
    a0 = a1 = anchor
    while a0 - 1 >= lo and settled(a0 - 1):
        a0 -= 1
    while a1 + 1 <= hi and settled(a1 + 1):
        a1 += 1
    print(f"MEASURED anchor_run {a0}-{a1} ({a1 - a0 + 1} frames)", flush=True)
    run_len = a1 - a0 + 1
    keep_min = int(spec.get("keep_anchor_min", KEEP_ANCHOR_MIN))
    mode = str(spec.get("anchor_mode", "settle"))
    runs = []
    if mode == "multi" or (mode == "auto" and run_len < keep_min):
        runs = plan_multi_runs(lo, hi, tracks, fw, fh, every=int(spec.get("anchor_every", 1)),
                               min_w=float(spec.get("anchor_min_w", 56)))
        print(f"MEASURED multi anchor runs {runs}", flush=True)
    multi = bool(runs)

    emit("PHASE", "pasting reference lettering")
    anchored = [f.copy() for f in frames]
    pad_a = int(spec.get("paste_pad", 4))
    ref_elements = []
    for b in anchor_boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        ref_elements.append(ref_warp[y0:y1, x0:x1].copy())
    anchor_eccs = []
    if multi:
        # warp the real lettering into every frame of every run along that frame's own
        # homography (perspective, not just scale), light-matched and motion-blurred
        for s, e in runs:
            for k, b in enumerate(anchor_boxes):
                Hs, ccs = {}, []
                for i in range(s, e + 1):
                    Hs[i], cc = element_homography(grays, anchor, i, b, tracks[k][i])
                    ccs.append(cc)
                Hs = smooth_run(Hs, b)
                for i in range(s, e + 1):
                    anchored[i] = paste_element(anchored[i], ref_warp, frames[anchor], Hs[i], b,
                                                _velocity(tracks[k], i), pad=pad_a)
                anchor_eccs.append({"run": [s, e], "element": k, "ecc_min": round(min(ccs), 3),
                                    "seeded": sum(1 for c in ccs if c < 0.6)})
                print(f"MEASURED run {s}-{e} element {k}: ecc min {min(ccs):.3f}, "
                      f"{sum(1 for c in ccs if c < 0.6)} frame(s) on the box seed", flush=True)
        anchor_set = {i for s, e in runs for i in range(s, e + 1)}
    else:
        # paste the real lettering on the settled anchor run (per element, moved with the car
        # and its own correction)
        for k, b in enumerate(anchor_boxes):
            mk = np.zeros((fh, fw), np.uint8)
            cv2.rectangle(mk, (int(b[0]) - pad_a, int(b[1]) - pad_a), (int(b[2]) + pad_a, int(b[3]) + pad_a), 255, -1)
            mf = cv2.GaussianBlur(mk, (0, 0), 2.5).astype(np.float32) / 255.0
            for i in range(a0, a1 + 1):
                p = tracks[k][i]
                s = (p[2] - p[0]) / max(1e-6, b[2] - b[0])
                M = np.float32([[s, 0, p[0] - s * b[0]], [0, s, p[1] - s * b[1]]])
                r = cv2.warpAffine(ref_warp, M, (fw, fh), flags=cv2.INTER_CUBIC)
                al = cv2.warpAffine(mf, M, (fw, fh))[..., None]
                anchored[i] = np.clip(anchored[i] * (1 - al) + r * al, 0, 255).astype(np.uint8)
        anchor_set = set(range(a0, a1 + 1))

    # masks: every window frame outside the anchor frames where the element was located
    pad_frac = float(spec.get("mask_pad", 0.25))
    masks = {}
    for i in win:
        if i in anchor_set:
            continue
        boxes = _mask_boxes(tracks, i, fw, fh, pad_frac)
        if boxes:
            masks[i] = boxes

    # follow-crop: a window riding the elements that zooms with the car, so H3 sees the
    # lettering at a steady size and resolution (compose maps each frame back by its reg)
    uw, uh = u[2] - u[0], u[3] - u[1]
    rw0 = max(2.4 * uw, 2.2 * uh * 16 / 9, 240.0)
    W, Hc = gen_dims(16 / 9)
    off = ((u[0] + u[2]) / 2, (u[1] + u[3]) / 2)
    cen, sz = [], []
    for i in win:
        T = motion.get(i)
        if T is None:   # car lost (left the frame): hold the nearest known pose
            T = motion[min(motion, key=lambda j: abs(j - i))]
        c = cv2.perspectiveTransform(np.float32([[off]]), T)[0, 0]
        cen.append(c)
        sz.append(motion_scale(T))
    cen, sz = np.array(cen, np.float32), np.array(sz, np.float32)
    sm = 3
    ker = np.ones(2 * sm + 1) / (2 * sm + 1)
    cen = np.stack([np.convolve(np.pad(cen[:, d], sm, mode="edge"), ker, mode="valid") for d in range(2)], 1)
    sz = np.convolve(np.pad(sz, sm, mode="edge"), ker, mode="valid")
    regs = []
    for (cx, cy), s in zip(cen, sz):
        rw = min(rw0 * s, fw, fh * W / Hc)
        rh = rw * Hc / W
        rw, rh = int(round(rw)), int(round(rh))
        regs.append((int(round(min(max(0, cx - rw / 2), fw - rw))), int(round(min(max(0, cy - rh / 2), fh - rh))), rw, rh))
    rws = [r[2] for r in regs]
    print(f"MEASURED window {lo}-{hi} ({length} f) crop {min(rws)}-{max(rws)} px wide -> canvas {W}x{Hc} "
          f"(x{W / max(rws):.2f}-x{W / min(rws):.2f}), masked frames {len(masks)}", flush=True)

    emit("PHASE", "writing generation inputs")
    work = spec["work_dir"]
    src_c, vm_c, am_c = [], [], []
    for k, i in enumerate(win):
        rx, ry, w_, h_ = regs[k]
        src_c.append(cv2.resize(anchored[i][ry:ry + h_, rx:rx + w_], (W, Hc), interpolation=cv2.INTER_AREA))
        m = np.zeros((fh, fw), np.uint8)
        for b in masks.get(i, []):
            m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
        mc = cv2.resize(m[ry:ry + h_, rx:rx + w_], (W, Hc), interpolation=cv2.INTER_NEAREST)
        vm_c.append(cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR))
        am_c.append(np.zeros((Hc, W, 3), np.uint8))
    audio = spec["clip"] if has_audio(spec["clip"]) else None
    if audio is None:
        audio = os.path.join(work, "silence.wav")
        silent_wav(audio, n / fps + 1)
    write_clip(os.path.join(work, "gen_src.mp4"), src_c, FPS, audio_from=audio, audio_start=lo / fps, lossless=True)
    write_clip(os.path.join(work, "gen_mask.mp4"), vm_c, FPS, lossless=True)
    write_clip(os.path.join(work, "gen_amask.mp4"), am_c, FPS, lossless=True)
    write_clip(os.path.join(work, "anchored.mp4"), anchored, fps, lossless=True)

    plan = {"fps": fps, "frames": n, "width": fw, "height": fh, "anchor": anchor, "window": [lo, hi],
            "length": length, "canvas": [W, Hc], "regs": regs, "anchor_run": [a0, a1],
            "masks": {str(k): v for k, v in masks.items()}, "anchor_boxes": anchor_boxes,
            "tracks": [{str(i): list(p[:4]) for i, p in t.items()} for t in tracks],
            "names": [e["name"] for e in elements], "align": {"ecc": cc, "inliers": inliers},
            "warnings": []}
    for k, crop in enumerate(ref_elements):
        cv2.imwrite(os.path.join(work, f"ref_element{k}.png"), crop)
    if multi:
        plan["anchor_policy"] = "multi"
        plan["lettering_source"] = "reference"
        plan["anchor_runs"] = runs
        plan["anchor_eccs"] = anchor_eccs
    elif mode in ("multi", "auto") and run_len < keep_min:
        plan["warnings"].append("multi-anchor mode found no run where the element is tracked, inside the frame "
                                "and large enough on all 8 frames; fell back to the settled anchor")
    if multi:
        pass
    elif run_len < keep_min:
        # H3 works in ~4-frame latent groups and the latent mask is max-pooled, so a run this
        # short is masked in latent space: H3 never sees the pasted lettering and draws the
        # element from the PROMPT. Keeping the pasted frames then leaves a one-frame pop (a
        # different plate, size and font) in the middle of H3's own rendering - job a56e505c
        # frame 98, 2026-11-07. So compose rebuilds them from their neighbours instead.
        plan["anchor_policy"] = "rebuild"
        plan["lettering_source"] = "prompt"
        plan["warnings"].append(
            f"only {run_len} anchor frame(s) (the car never holds the reference pose): H3 could not use the "
            f"reference, so the lettering was drawn from the PROMPT - check the spelling on the proof sheet. "
            f"The pasted anchor frame(s) {a0}-{a1} were rebuilt from their neighbours so they do not pop.")
    else:
        plan["anchor_policy"] = "keep"
        plan["lettering_source"] = "reference"
    if not masks:
        raise RuntimeError("nothing to fix: the elements match the reference pose in every frame of the window")
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
    emit("PROGRESS", "100")


def prep_harmonise(spec):
    """Second pass for multi-anchor: the anchor frames are pasted, so they can still read as
    a sticker. Mask ONLY the anchor runs on the composited first pass (spec["base"]) and let
    H3 redraw them from the frames around them, which now carry the right lettering in H3's
    own perspective, blur and light. Same window and follow-crop as the first pass; writes
    gen2_*.mp4 and plan2.json."""
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    runs = plan.get("anchor_runs") or []
    if not runs:
        raise RuntimeError("harmonise needs a multi-anchor plan (no anchor_runs in plan.json)")
    base, fps = read_clip(spec["base"])
    fh, fw = base[0].shape[:2]
    lo, hi = plan["window"]
    W, Hc = plan["canvas"]
    regs = plan["regs"]
    tracks = [{int(k): v for k, v in t.items()} for t in plan["tracks"]]
    pad_frac = float(spec.get("mask_pad", 0.25))
    masks = {}
    for s, e in runs:
        for i in range(s, e + 1):
            boxes = _mask_boxes(tracks, i, fw, fh, pad_frac)
            if boxes:
                masks[i] = boxes
    emit("PHASE", "writing harmonise inputs")
    src_c, vm_c, am_c = [], [], []
    for k, i in enumerate(range(lo, hi + 1)):
        rx, ry, w_, h_ = regs[k]
        src_c.append(cv2.resize(base[i][ry:ry + h_, rx:rx + w_], (W, Hc), interpolation=cv2.INTER_AREA))
        m = np.zeros((fh, fw), np.uint8)
        for b in masks.get(i, []):
            m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
        vm_c.append(cv2.cvtColor(cv2.resize(m[ry:ry + h_, rx:rx + w_], (W, Hc), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR))
        am_c.append(np.zeros((Hc, W, 3), np.uint8))
    audio = spec["clip"] if has_audio(spec["clip"]) else os.path.join(work, "silence.wav")
    if not os.path.exists(audio):
        silent_wav(audio, len(base) / fps + 1)
    write_clip(os.path.join(work, "gen2_src.mp4"), src_c, FPS, audio_from=audio, audio_start=lo / fps, lossless=True)
    write_clip(os.path.join(work, "gen2_mask.mp4"), vm_c, FPS, lossless=True)
    write_clip(os.path.join(work, "gen2_amask.mp4"), am_c, FPS, lossless=True)
    plan2 = dict(plan, masks={str(k): v for k, v in masks.items()}, anchor_policy="harmonise", anchor_runs=[],
                 harmonised_runs=runs)
    json.dump(plan2, open(os.path.join(work, "plan2.json"), "w"), indent=1)
    print(f"MEASURED harmonise masked frames {len(masks)} over runs {runs}", flush=True)
    emit("PROGRESS", "100")


# ---------------------------------------------------------------- compose

def _grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def _pair_for_run(a0, a1, n):
    """Two frames outside the anchor run to rebuild it from: one each side when possible
    (interpolation), else the two nearest on the one side there is (extrapolation)."""
    if a0 - 1 >= 0 and a1 + 1 < n:
        return a0 - 1, a1 + 1
    if a0 - 2 >= 0:
        return a0 - 2, a0 - 1
    if a1 + 2 < n:
        return a1 + 1, a1 + 2
    return None


def _ecc_homography(img_p, img_q, roi):
    """Homography W mapping frame-q coords -> frame-p coords, solved by ECC in `roi` of q.
    Returns (W, cc) or (None, -1) when it does not converge."""
    x0, y0, x1, y1 = [int(round(v)) for v in roi]
    gp = cv2.GaussianBlur(cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY), (0, 0), 0.8).astype(np.float32)
    gq = cv2.GaussianBlur(cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY), (0, 0), 0.8).astype(np.float32)
    try:
        cc, W = cv2.findTransformECC(gq[y0:y1, x0:x1], gp[y0:y1, x0:x1], np.eye(3, dtype=np.float32),
                                     cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 3)
    except cv2.error:
        return None, -1.0
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    full = np.linalg.inv(T) @ W.astype(np.float64) @ T
    return full / full[2, 2], float(cc)


def _box_similarity(bp, bq):
    """Fallback motion from two tracked boxes: scale about the centre + shift, as a
    q-coords -> p-coords 3x3 (same direction as _ecc_homography)."""
    sp = max(bp[2] - bp[0], 1e-3) / max(bq[2] - bq[0], 1e-3)
    cpx, cpy = (bp[0] + bp[2]) / 2, (bp[1] + bp[3]) / 2
    cqx, cqy = (bq[0] + bq[2]) / 2, (bq[1] + bq[3]) / 2
    return np.array([[sp, 0, cpx - sp * cqx], [0, sp, cpy - sp * cqy], [0, 0, 1]], np.float64)


def rebuild_run(frames, run, tracks, anchor_boxes, pad_frac=0.25, feather=2.5):
    """Replace the elements on every frame of `run` (a0, a1) with frame p's pixels warped
    along the measured p->q motion, interpolated (or extrapolated) to that frame's place in
    time. p and q lie outside the run, so the rebuilt frames inherit the surrounding frames'
    size, angle, blur and lettering instead of the pasted reference. In place; returns
    [{frame, from, t, ecc}]."""
    a0, a1 = run
    fh, fw = frames[0].shape[:2]
    pair = _pair_for_run(a0, a1, len(frames))
    if pair is None:
        return []
    p, q = pair

    def box(e, i):
        return tracks[e].get(i, anchor_boxes[e])

    # ECC window: all elements at q, with room for the car body around them
    bs = [box(e, q) for e in range(len(anchor_boxes))]
    m = 30
    roi = clamp_box([min(b[0] for b in bs) - m, min(b[1] for b in bs) - m,
                     max(b[2] for b in bs) + m, max(b[3] for b in bs) + m], fw, fh)
    W, cc = _ecc_homography(frames[p], frames[q], roi)
    if W is None or cc < 0.8:
        W = _box_similarity(box(0, p), box(0, q))
    done = []
    for i in range(a0, a1 + 1):
        t = (i - p) / float(q - p)
        Wt = np.eye(3) + (W - np.eye(3)) * t
        pred = cv2.warpPerspective(frames[p], Wt, (fw, fh), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        mk = np.zeros((fh, fw), np.uint8)
        for e in range(len(anchor_boxes)):
            b = box(e, i)
            w, h = b[2] - b[0], b[3] - b[1]
            pad = max(10.0, pad_frac * max(w, h) * 0.5 + 0.15 * h)
            x0, y0, x1, y1 = clamp_box([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], fw, fh)
            mk[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))] = 255
        a = (cv2.GaussianBlur(mk, (0, 0), feather).astype(np.float32) / 255.0)[..., None]
        frames[i] = np.clip(frames[i].astype(np.float32) * (1 - a) + pred.astype(np.float32) * a,
                            0, 255).astype(np.uint8)
        done.append({"frame": i, "from": p, "t": round(t, 3), "ecc": round(cc, 4)})
    return done


def compose(spec):
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, spec.get("plan", "plan.json"))))
    anchored, fps = read_clip(spec.get("base") or os.path.join(work, "anchored.mp4"))
    fh, fw = anchored[0].shape[:2]
    lo, hi = plan["window"]
    regs = plan["regs"]
    masks = {int(k): v for k, v in plan["masks"].items()}
    feather = int(spec.get("feather", 8))
    names = plan["names"]
    anchor = plan["anchor"]
    a0, a1 = plan["anchor_run"]
    tag = str(spec.get("tag", ""))

    # reference appearance of each element for scoring: the reference crop prep saved (in
    # multi-anchor mode the anchor frame itself is masked, so it no longer carries it)
    refs = []
    for k, b in enumerate(plan["anchor_boxes"]):
        p = os.path.join(work, f"ref_element{k}.png")
        img = cv2.imread(p) if os.path.exists(p) else None
        if img is None:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            img = anchored[anchor][y0:y1, x0:x1]
        refs.append(_grad(img))
    tracks = [{int(k): v for k, v in t.items()} for t in plan["tracks"]]

    results = []
    for ti, take in enumerate(spec["takes"]):
        emit("PHASE", f"compositing take {ti + 1}")
        gen, _ = read_clip(take)
        out = [f.copy() for f in anchored]
        for k, i in enumerate(range(lo, hi + 1)):
            if k >= len(gen) or i not in masks:
                continue
            rx, ry, rw, rh = regs[k]
            m = np.zeros((fh, fw), np.uint8)
            for b in masks[i]:
                m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
            mf = cv2.GaussianBlur(cv2.dilate(m, np.ones((feather, feather), np.uint8)), (0, 0), feather / 2.0)
            a = (mf[ry:ry + rh, rx:rx + rw].astype(np.float32) / 255.0)[..., None]
            back = cv2.resize(gen[k], (rw, rh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            reg = out[i][ry:ry + rh, rx:rx + rw].astype(np.float32)
            out[i][ry:ry + rh, rx:rx + rw] = np.clip(reg * (1 - a) + back * a, 0, 255).astype(np.uint8)
        rebuilt = []
        if plan.get("anchor_policy") == "rebuild":
            rebuilt = rebuild_run(out, (a0, a1), tracks, plan["anchor_boxes"])
            for r in rebuilt:
                print(f"MEASURED rebuilt anchor frame {r['frame']} from frame {r['from']} "
                      f"(t {r['t']}, ecc {r['ecc']})", flush=True)
        # score: lettering similarity to the reference on masked frames (gradient NCC at
        # the tracked box, reference resized to it) minus a flicker penalty
        sims, flick = [], []
        for e, ref_g in enumerate(refs):
            h, w = ref_g.shape[:2]
            prev = None
            for i in sorted(masks):
                p = tracks[e].get(i)
                if p is None:
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in p]
                if x0 < 0 or y0 < 0 or x1 > fw or y1 > fh or x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                crop = _grad(cv2.resize(out[i][y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA))
                sims.append(_ncc(crop, ref_g))
                if prev is not None:
                    flick.append(float(np.abs(crop - prev).mean()))
                prev = crop
        score = (float(np.mean(sims)) if sims else 0.0) - 0.002 * (float(np.mean(flick)) if flick else 0.0)
        path = os.path.join(work, f"{tag}take{ti + 1}.mp4")
        write_clip(path, out, fps, audio_from=spec["clip"] if has_audio(spec["clip"]) else None)
        results.append({"take": ti + 1, "seed": spec["seeds"][ti], "score": round(score, 4),
                        "lettering_similarity": round(float(np.mean(sims)) if sims else 0.0, 4),
                        "flicker": round(float(np.mean(flick)) if flick else 0.0, 3),
                        "rebuilt_frames": rebuilt, "path": path})
        print(f"MEASURED take {ti + 1} seed {spec['seeds'][ti]} score {score:.4f}", flush=True)
        emit("PROGRESS", str(int(100 * (ti + 1) / (len(spec["takes"]) + 1))))

    best = max(results, key=lambda r: r["score"])
    shutil.copy(best["path"], spec["out"])
    emit("PHASE", "proof sheet")
    proof_sheet(spec, plan, read_clip(spec["clip"])[0], read_clip(best["path"])[0], best, results)
    report = {"best_take": best["take"], "takes": [{k: v for k, v in r.items() if k != "path"} for r in results],
              "anchor_frame": anchor, "anchor_run": plan["anchor_run"], "window": plan["window"],
              "elements": names, "align": plan["align"], "warnings": plan["warnings"],
              "anchor_policy": plan.get("anchor_policy", "keep"),
              "lettering_source": plan.get("lettering_source", "reference"),
              "anchor_runs": plan.get("anchor_runs") or plan.get("harmonised_runs") or [],
              "anchor_eccs": plan.get("anchor_eccs", [])}
    json.dump(report, open(spec["report"], "w"), indent=1)
    emit("PROGRESS", "100")


def proof_sheet(spec, plan, original, fixed, best, results):
    """Per element: original over fixed, zoomed, at six frames spread over the masked
    range plus one anchor frame; a header with the take scores."""
    masks = sorted(int(k) for k in plan["masks"])
    tracks = [{int(k): v for k, v in t.items()} for t in plan["tracks"]]
    fh, fw = original[0].shape[:2]
    runs = plan.get("anchor_runs") or []
    anchors = {i for s, e in runs for i in range(s, e + 1)} or set(range(plan["anchor_run"][0], plan["anchor_run"][1] + 1))
    first_anchor = runs[len(runs) // 2][0] + 3 if runs else plan["anchor_run"][0]

    def label(i):
        if i not in anchors:
            return f"{i} fixed"
        return f"{i} rebuilt" if plan.get("anchor_policy") == "rebuild" else f"{i} anchor"
    tile_w = 260
    rows = []
    for e, name in enumerate(plan["names"]):
        own = [i for i in masks if i in tracks[e]] or masks
        picks = [own[int(round(t * (len(own) - 1)))] for t in np.linspace(0, 1, 6)] + [first_anchor]
        b = plan["anchor_boxes"][e]
        bw, bh = b[2] - b[0], b[3] - b[1]
        k = max(bh * 3.0, 50) / max(bw * 2.2, 80)   # one tile shape per element; crops scale with the car
        sz = (tile_w, int(tile_w * k))
        top, bot = [], []
        for i in picks:
            p = tracks[e].get(i, b)
            cw = min(max((p[2] - p[0]) * 2.2, 80), fw)
            ch = min(cw * k, fh)
            cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            x0 = int(min(max(0, cx - cw / 2), fw - cw)); y0 = int(min(max(0, cy - ch / 2), fh - ch))
            o = cv2.resize(original[i][y0:y0 + int(ch), x0:x0 + int(cw)], sz, interpolation=cv2.INTER_CUBIC)
            f = cv2.resize(fixed[i][y0:y0 + int(ch), x0:x0 + int(cw)], sz, interpolation=cv2.INTER_CUBIC)
            for t, lab in ((o, f"{i} before"), (f, label(i))):
                cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            top.append(o)
            bot.append(f)
        band = np.vstack([np.hstack(top), np.hstack(bot)])
        label = np.full((24, band.shape[1], 3), 30, np.uint8)
        cv2.putText(label, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(np.vstack([label, band]))
    width = max(r.shape[1] for r in rows)
    head = np.full((30, width, 3), 20, np.uint8)
    txt = "MEASURED takes: " + "  ".join(f"#{r['take']} seed {r['seed']} score {r['score']:.3f}" for r in results) + \
          f"   -> best #{best['take']}"
    cv2.putText(head, txt[:160], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 160), 1)
    sheet = np.vstack([head] + [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows])
    cv2.imwrite(spec["proof"], sheet)


if __name__ == "__main__":
    cmds = {"prep": prep, "harmonise": prep_harmonise, "compose": compose}
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit("usage: lettering.py prep|harmonise|compose <spec.json>")
    cmds[sys.argv[1]](json.load(open(sys.argv[2])))
