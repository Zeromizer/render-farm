"""Plate soften: carry a clean plate backwards through a drive-in and blur it with the car's motion.

The problem it solves: fix_lettering pastes correct plates only on the frames where the car holds the
reference pose (the "anchor run"). Before that, plates are prompt-drawn and garbled, and the clean plate
pops in on one frame. This script:

1. takes each plate's quad on the first clean frame (`anchor`, e.g. 108),
2. tracks it BACKWARDS frame by frame with ECC (affine) on a window of rigid bumper/grille around the plate,
   with the plate itself masked out (its garbled lettering changes every frame),
3. warps the clean anchor-frame plate into every earlier frame, gain-matched to that frame's plate,
   motion-blurred along the plate's own velocity (180-degree shutter) plus a light base softness that eases
   to zero at the anchor, and composites it inside a feathered plate mask,
4. leaves frames >= anchor untouched (copied as decoded).

params.args = [<job json>, <out dir>]. Job JSON:
  {"src": "spikes/plate_soften/in/clip.mp4", "anchor": 108,
   "plates": [{"name": "freed", "quad": [[x,y],[x,y],[x,y],[x,y]]}, ...],   # TL, TR, BR, BL at `anchor`
   "shutter": 0.5, "base_sigma": 0.6, "ease_frames": 12, "min_cc": 0.6, "crf": 12}
Outputs in <out>: plate_soften.mp4, track.json, track_sheet.png, compare_sheet.png.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def write_mp4(frames, fps, path, crf):
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")


def quad_bbox(q, pad_x, pad_y, w, h):
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    return (int(max(0, math.floor(x0 - pad_x))), int(max(0, math.floor(y0 - pad_y))),
            int(min(w, math.ceil(x1 + pad_x))), int(min(h, math.ceil(y1 + pad_y))))


def apply_affine(A, pts):
    return pts @ A[:, :2].T + A[:, 2]


def plate_contrast(lum, q):
    """Median luminance inside the quad / 75th percentile of a 3-7 px ring around it. A real plate is dark
    against the painted bumper (<< 1); the ring's 75th percentile skips the black grille under the plate.
    Plain paving or sea scores near 1."""
    h, w = lum.shape
    inner = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inner, cv2.convexHull(q.astype(np.int32)), 1)
    core = cv2.erode(inner, np.ones((3, 3), np.uint8))
    ring = cv2.dilate(inner, np.ones((15, 15), np.uint8)) - cv2.dilate(inner, np.ones((7, 7), np.uint8))
    if core.sum() < 20 or ring.sum() < 20:
        return 9.0
    return float(np.median(lum[core > 0]) + 4) / float(np.percentile(lum[ring > 0], 75) + 4)


def track_back(frames, anchor, quad0, min_cc, search=60, log=print):
    """Return {frame: quad} for frames anchor..0 while tracking holds, and per-frame cc.

    Each step: (1) coarse translation by template-matching the front-end window of frame i+1 inside a
    +-`search` px area of frame i (the cars enter fast), (2) ECC affine refine with the plate masked out,
    (3) sanity: quad inside the frame, area/aspect close to the anchor's, and still a DARK plate against its
    surroundings - two failures in a row stop the track (that is where it slid off the car)."""
    h, w = frames[0].shape[:2]
    lum = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    gray = [cv2.GaussianBlur(l, (0, 0), 0.8).astype(np.float32) for l in lum]
    quads, ccs = {anchor: quad0.copy()}, {anchor: 1.0}
    q = quad0.copy()
    area0 = cv2.contourArea(quad0)
    asp0 = (quad0[:, 0].max() - quad0[:, 0].min()) / max(1.0, quad0[:, 1].max() - quad0[:, 1].min())
    vel = np.zeros(2, np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    fails = 0
    con0 = plate_contrast(lum[anchor], quad0)
    max_contrast = min(0.85, con0 + 0.3)
    log(f"  anchor contrast {con0:.2f} -> max {max_contrast:.2f}")
    for i in range(anchor - 1, -1, -1):
        pw = q[:, 0].max() - q[:, 0].min()
        ph = q[:, 1].max() - q[:, 1].min()
        # (1) coarse shift: window (frame i+1) matched in frame i
        tx0, ty0, tx1, ty1 = quad_bbox(q, 1.6 * pw, 2.2 * ph, w, h)
        if tx1 - tx0 < 40 or ty1 - ty0 < 30:
            break
        tmpl = gray[i + 1][ty0:ty1, tx0:tx1]
        sx0, sy0 = max(0, tx0 - search), max(0, ty0 - search // 2)
        sx1, sy1 = min(w, tx1 + search), min(h, ty1 + search // 2)
        area = gray[i][sy0:sy1, sx0:sx1]
        shift = vel.copy()
        if area.shape[0] >= tmpl.shape[0] and area.shape[1] >= tmpl.shape[1]:
            r = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            if mv > 0.5:
                shift = np.array([sx0 + ml[0] - tx0, sy0 + ml[1] - ty0], np.float32)
        # (2) ECC refine. warp maps template (frame i+1) coords -> input (frame i) coords; its mask lives in
        # INPUT coords, so the plate hole goes at the predicted position
        x0, y0, x1, y1 = quad_bbox(q + shift, 1.6 * pw, 2.2 * ph, w, h)
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 255
        cv2.fillConvexPoly(mask, cv2.convexHull((q + shift).astype(np.int32)), 0)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        A = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
        try:
            cc, A = cv2.findTransformECC(gray[i + 1], gray[i], A, cv2.MOTION_AFFINE, crit, mask, 5)
        except cv2.error:
            break
        qn = apply_affine(A, q).astype(np.float32)
        # (3) sanity
        inside = qn[:, 0].min() >= 1 and qn[:, 1].min() >= 1 and qn[:, 0].max() < w - 1 and qn[:, 1].max() < h - 1
        if not inside:
            break
        ar = cv2.contourArea(qn) / area0
        asp = (qn[:, 0].max() - qn[:, 0].min()) / max(1.0, qn[:, 1].max() - qn[:, 1].min())
        con = plate_contrast(lum[i], qn)
        ok = cc >= min_cc and 0.25 <= ar <= 1.8 and 0.6 * asp0 <= asp <= 1.6 * asp0 and con <= max_contrast
        log(f"  f{i}: shift {shift.round(1).tolist()} cc {cc:.3f} area {ar:.2f} aspect {asp:.2f} contrast {con:.2f}"
            + ("" if ok else f" -> FAIL {fails + 1}"))
        if not ok:
            fails += 1
            if fails >= 2:
                # the previous frame was the first failure too: drop it
                quads.pop(i + 1, None)
                ccs.pop(i + 1, None)
                break
        else:
            fails = 0
        vel = (qn.mean(0) - q.mean(0)).astype(np.float32)
        q = qn
        quads[i], ccs[i] = q.copy(), float(cc)
    return quads, ccs


def motion_kernel(vx, vy, length):
    """Normalised line kernel of `length` px along (vx, vy); length < 1 -> identity."""
    if length < 1.0:
        k = np.zeros((1, 1), np.float32)
        k[0, 0] = 1
        return k
    n = int(math.ceil(length)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    ang = math.atan2(vy, vx)
    for t in np.linspace(-length / 2, length / 2, max(3, int(length * 3))):
        x = int(round(c + t * math.cos(ang)))
        y = int(round(c + t * math.sin(ang)))
        k[y, x] += 1
    return k / k.sum()


def composite(frames, anchor, name, quads, opts):
    """Warp the anchor-frame plate into every tracked frame < anchor, blurred and gain-matched."""
    h, w = frames[0].shape[:2]
    qa = quads[anchor]
    src = frames[anchor].astype(np.float32)
    plate_mask_a = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(plate_mask_a, qa.astype(np.int32), 1.0)
    # plate tone in the anchor frame (the clean paste): dark background level = 35th percentile luminance
    lum_a = cv2.cvtColor(frames[anchor], cv2.COLOR_BGR2GRAY)
    tone_a = np.percentile(lum_a[plate_mask_a > 0], 35)
    report = []
    idx = sorted(i for i in quads if i < anchor)
    first = idx[0] if idx else anchor
    for i in idx:
        fade = min(1.0, (i - first + 1) / float(max(1, opts.get("fade_frames", 4))))
        q = quads[i]
        # affine anchor -> frame i from the two quads (least squares over 4 corners)
        A, _ = cv2.estimateAffine2D(qa, q, method=cv2.LMEDS)
        if A is None:
            continue
        layer = cv2.warpAffine(src, A, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        # mask = tracked quad, grown 1.5 px to cover the garbled plate's edge, feathered
        m = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(m, cv2.convexHull(q.astype(np.int32)), 1.0)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        m = cv2.GaussianBlur(m, (0, 0), 0.9)
        # gain-match the plate's dark background to what Seedance lit in this frame
        lum_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        inside = m > 0.5
        tone_i = np.percentile(lum_i[inside], 35) if inside.any() else tone_a
        gain = float(np.clip((tone_i + 4) / (tone_a + 4), 0.75, 1.35))
        layer *= gain
        # motion blur along this plate's velocity (centred difference), base softness easing out at anchor
        prv = quads.get(i - 1, q)
        nxt = quads.get(i + 1, q)
        v = (nxt.mean(0) - prv.mean(0)) / (2.0 if (i - 1) in quads and (i + 1) in quads else 1.0)
        speed = float(np.hypot(*v))
        length = opts["shutter"] * speed
        ease = min(1.0, (anchor - i) / float(max(1, opts["ease_frames"])))
        sigma = opts["base_sigma"] * ease
        x0, y0, x1, y1 = quad_bbox(q, 12 + length, 12 + length, w, h)
        crop = layer[y0:y1, x0:x1]
        mc = m[y0:y1, x0:x1]
        k = motion_kernel(v[0], v[1], length)
        if k.shape[0] > 1:
            crop = cv2.filter2D(crop, -1, k, borderType=cv2.BORDER_REFLECT)
            mc = cv2.filter2D(mc, -1, k, borderType=cv2.BORDER_CONSTANT)
        if sigma > 0.05:
            crop = cv2.GaussianBlur(crop, (0, 0), sigma)
        mc = mc * fade
        base = frames[i][y0:y1, x0:x1].astype(np.float32)
        out = base * (1 - mc[..., None]) + crop * mc[..., None]
        frames[i][y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        report.append({"frame": i, "speed_px": round(speed, 2), "blur_px": round(length, 2), "fade": round(fade, 2),
                       "sigma": round(sigma, 2), "gain": round(gain, 3)})
    return report


def sheet(tiles, cols):
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles[r:r + cols]]
        row += [np.zeros((h, w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    for n in os.listdir(out):
        os.remove(os.path.join(out, n))
    opts = {"shutter": float(job.get("shutter", 0.5)), "base_sigma": float(job.get("base_sigma", 0.6)),
            "ease_frames": int(job.get("ease_frames", 12)), "fade_frames": int(job.get("fade_frames", 4))}
    anchor = int(job["anchor"])
    frames, fps = read_frames(job["src"])
    orig = [f.copy() for f in frames]
    print(f"read {len(frames)} frames at {fps} fps", flush=True)
    track = {}
    logf = open(os.path.join(out, "track_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    for k, p in enumerate(job["plates"]):
        quad0 = np.array(p["quad"], np.float32)
        log(f"== {p['name']}")
        quads, ccs = track_back(orig, anchor, quad0, float(job.get("min_cc", 0.6)), log=log)
        first = min(quads)
        print(f"{p['name']}: tracked back to frame {first} (min cc {min(ccs.values()):.3f})", flush=True)
        rep = composite(frames, anchor, p["name"], quads, opts)
        track[p["name"]] = {"tracked_from": first, "anchor": anchor,
                            "quads": {str(i): quads[i].round(2).tolist() for i in sorted(quads)},
                            "cc": {str(i): round(ccs[i], 4) for i in sorted(ccs)}, "frames": rep}
        print(f"PROGRESS {int(30 + 50 * (k + 1) / len(job['plates']))}", flush=True)
    logf.close()
    write_mp4(frames, fps, os.path.join(out, "plate_soften.mp4"), int(job.get("crf", 12)))
    with open(os.path.join(out, "track.json"), "w") as f:
        json.dump(track, f)
    # track sheet: tracked quads drawn on the original, every 8 frames
    tiles = []
    for i in range(0, anchor + 1, 8):
        t = orig[i].copy()
        for nm, tr in track.items():
            q = tr["quads"].get(str(i))
            if q:
                cv2.polylines(t, [np.array(q, np.int32)], True, (0, 255, 255), 1)
        cv2.putText(t, f"f{i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(t, (320, 320), interpolation=cv2.INTER_AREA))
    cv2.imwrite(os.path.join(out, "track_sheet.png"), sheet(tiles, 5))
    # compare sheet: plate strip before/after at 2x around both plates
    ys = [int(np.array(q, np.float32)[:, 1].mean()) for tr in track.values() for q in tr["quads"].values()]
    yc = int(np.median(ys)) if ys else frames[0].shape[0] // 2
    rows = []
    for i in list(range(36, anchor, 8)) + [anchor - 1, anchor]:
        for lab, src in (("before", orig), ("after", frames)):
            s = src[i][max(0, yc - 30):yc + 30, 60:760]
            s = cv2.resize(s, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LANCZOS4)
            cv2.putText(s, f"{lab} f{i} ({i / fps:.2f}s)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rows.append(s)
    cv2.imwrite(os.path.join(out, "compare_sheet.png"), np.vstack(rows))
    print("PROGRESS 100", flush=True)


if __name__ == "__main__":
    main()
