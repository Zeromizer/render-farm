"""Planar patch: tracked replacement of flat elements (number plates, badges, labels) in a clip.

    python planar.py spec.json

spec.json
  {
    "clip": "in.mp4", "out": "out.mp4", "proof": "out-proof.png" (optional), "debug_dir": null,
    "max_frames": 720 (optional; the clip is held in RAM, ~6 MB per 1080p frame),
    "patches": [
      {"name": "plate", "artwork": "plate.png", "alpha": null,
       "key_frame": -1 (middle), "key_box": null | [x0, y0, x1, y1],
       "template": "gray" | "edges" | "none", "min_score": 0.35,
       "refine": true, "smooth": 2, "win": 1.0,
       "clear": 0.0, "clear_grow": 2.0, "clear_radius": 5, "clear_shape": "auto" | "alpha" | "rect",
       "blur": 1.2, "feather": 2.0, "match": true,
       "frames": null | [first, last] (inclusive; the element is only tracked and pasted inside this
       range, for a turntable where a face is toward the camera for part of the clip; list the same
       artwork twice with two ranges when it comes round twice), "fade": 0 (frames of linear blend
       from the generated element at each end of the range)}
    ]
  }

Every patch is a corner pin driven by a planar track:
  1. key frame: the element is located either from `key_box` or by multi-scale template match of
     the artwork (`gray` for high-contrast artwork such as a plate, `edges` = Canny maps for chrome
     lettering whose brightness depends on the lighting). Optionally refined to the dark rounded
     rectangle found inside that window (`refine`, plates only).
  2. track: both directions from the key frame. Per frame the dark quad is re-detected inside a
     window around the previous corners; when that fails (always, for letters-only elements), LK
     optical flow on features in the window gives a RANSAC homography that carries the corners.
     Corner tracks are smoothed with a short moving average.
  3. composite: `clear` > 0 first inpaints the footage under the element so generated lettering does
     not ghost under a letters-only alpha: with an alpha the cleared area is the alpha's own shape
     grown by `clear_grow` px (frame scale), without one it is the rectangle shrunk by `clear`
     (`clear_shape` forces either); `clear_radius` is the Telea inpaint radius. The
     artwork is warped onto the quad, alpha = supplied mask or a rounded rectangle, feathered,
     brightness-matched to the footage inside the quad (`match`), and blurred slightly so it takes
     the footage's softness. Audio is copied from the source.
Deterministic: the text is never generated, so it reads correctly in every frame.

Prints PHASE <text>, PROGRESS <0-100> and MEASURED ... lines for runners/planar_patch.py.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract"))
from common import find_ffmpeg_tool  # noqa: E402

DEFAULTS = {"alpha": None, "key_frame": -1, "key_box": None, "template": "gray", "min_score": 0.35,
            "refine": True, "smooth": 2, "win": 1.0, "clear": 0.0, "clear_grow": 2.0, "clear_radius": 5,
            "clear_shape": "auto", "blur": 1.2, "feather": 2.0, "match": True,
            "frames": None, "fade": 0}


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def dark_quad(gray, x0, y0, x1, y1, min_area, aspect=None, area_ref=None):
    """Dark quadrilateral inside the window that looks like the element (aspect within 25% of
    `aspect`, area 0.5-2x `area_ref`), in full-frame coords, or None. A plate's black is no darker
    than the bumper recess around it, so several thresholds are tried and the best-fitting
    rectangle wins rather than the largest."""
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return None
    best = None
    blur = cv2.GaussianBlur(win, (3, 3), 0)
    for t in (28, 40, 55, 70):
        thr = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)[1]
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            if min(w, h) < 4 or a / max(w * h, 1) < 0.8:
                continue
            asp = max(w, h) / max(min(w, h), 1)
            score = 0.0
            if aspect:
                if abs(asp - aspect) / aspect > 0.25:
                    continue
                score += abs(asp - aspect) / aspect
            if area_ref:
                if not (0.5 * area_ref <= w * h <= 2.0 * area_ref):
                    continue
                score += abs(w * h - area_ref) / area_ref
            if best is None or score < best[0]:
                best = (score, cv2.boxPoints(rect))
    if best is None:
        return None
    q = order_quad(best[1])
    q[:, 0] += x0
    q[:, 1] += y0
    return q


def _edges(gray):
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160).astype(np.float32) / 255.0


def find_element(gray, tmpl_gray, mode):
    """Multi-scale template match; returns ((x0, y0, x1, y1), score). `edges` matches Canny maps,
    which is what works for chrome lettering: the artwork's brightness never matches the footage,
    its outlines do."""
    scene_edges = cv2.GaussianBlur(_edges(gray), (0, 0), 1.0) if mode == "edges" else None
    best = (None, -1.0)
    th, tw = tmpl_gray.shape
    # Edge maps of a tiny template are almost empty and correlate "perfectly" with any empty patch
    # of sky, so edges mode needs a real minimum size and a density check: the matched window must
    # carry a comparable amount of edge to the template, or the score is discounted.
    min_w = 48 if mode == "edges" else 12
    for s in np.linspace(0.06, 0.6, 40):
        w, h = int(tw * s), int(th * s)
        if w < min_w or h < 6 or w >= gray.shape[1] or h >= gray.shape[0]:
            continue
        t = cv2.resize(tmpl_gray, (w, h), interpolation=cv2.INTER_AREA)
        if mode == "edges":
            t = cv2.GaussianBlur(_edges(t), (0, 0), 1.0)
            sc = scene_edges
            if t.mean() < 0.02:
                continue
        else:
            sc = gray
        r = cv2.matchTemplate(sc, t, cv2.TM_CCOEFF_NORMED)
        if mode == "edges":
            # A near-constant window makes the normalised correlation degenerate (it reports 1.0
            # on empty sky), so windows carrying under 30% of the template's edge density are out.
            dens = cv2.boxFilter(sc, -1, (w, h), anchor=(0, 0), normalize=True, borderType=cv2.BORDER_CONSTANT)
            r[dens[:r.shape[0], :r.shape[1]] < 0.3 * t.mean()] = -1.0
        _, mx, _, loc = cv2.minMaxLoc(r)
        score = float(mx)
        if score > best[1]:
            best = ((loc[0], loc[1], loc[0] + w, loc[1] + h), score)
    return best


def track(grays, key, q, aspect, refine, win_scale):
    """Corner quads for every frame, tracked both ways from the key frame."""
    n = len(grays)
    H, W = grays[0].shape
    quads = [None] * n
    quads[key] = q
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    misses = 0
    lost = 0
    for direction in (1, -1):
        i = key
        while 0 <= i + direction < n:
            j = i + direction
            prev = quads[i]
            cx, cy = prev.mean(0)
            pw = float(np.ptp(prev[:, 0]))
            ph = float(np.ptp(prev[:, 1]))
            half_h = max(ph * 1.2, pw * 0.5 * win_scale)
            win = (int(max(0, cx - pw * win_scale)), int(max(0, cy - half_h)),
                   int(min(W, cx + pw * win_scale)), int(min(H, cy + half_h)))
            q2 = dark_quad(grays[j], *win, 0.4 * pw * ph, aspect=aspect, area_ref=pw * ph) if refine else None
            if q2 is not None and np.abs(q2 - prev).max() < 0.35 * pw:
                quads[j] = q2
            else:
                mask = np.zeros_like(grays[i])
                cv2.rectangle(mask, (win[0], win[1]), (win[2], win[3]), 255, -1)
                p0 = cv2.goodFeaturesToTrack(grays[i], 200, 0.01, 4, mask=mask)
                quads[j] = prev.copy()
                misses += 1
                moved = False
                if p0 is not None and len(p0) >= 8:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[i], grays[j], p0, None, **lk)
                    good = st.ravel() == 1
                    if good.sum() >= 8:
                        Hm, _ = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 3.0)
                        if Hm is not None:
                            quads[j] = cv2.perspectiveTransform(prev.reshape(1, 4, 2), Hm).reshape(4, 2).astype(np.float32)
                            moved = True
                if not moved:
                    lost += 1
            i = j
    return quads, misses, lost


def smooth_quads(quads, k):
    arr = np.stack(quads)
    return np.stack([arr[max(0, i - k):i + k + 1].mean(0) for i in range(len(quads))])


def range_weights(n, lo, hi, fade):
    """Per-frame blend weight of a patch: 0 outside [lo, hi], a linear ramp of `fade` frames inside
    each end, 1 elsewhere. No ramp at the clip's own ends: a loop that opens and closes on the
    element wants it fully patched on frame 0 and on the last frame."""
    fade = max(0, int(fade))
    weight = np.zeros(n, np.float32)
    for i in range(lo, hi + 1):
        w_in = (i - lo + 1) / (fade + 1) if (fade > 0 and lo > 0) else 1.0
        w_out = (hi - i + 1) / (fade + 1) if (fade > 0 and hi < n - 1) else 1.0
        weight[i] = min(1.0, w_in, w_out)
    return weight


def rounded_alpha(h, w):
    alpha = np.zeros((h, w), np.float32)
    r = int(0.06 * h)
    cv2.rectangle(alpha, (r, 0), (w - r, h), 1.0, -1)
    cv2.rectangle(alpha, (0, r), (w, h - r), 1.0, -1)
    for c in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(alpha, c, r, 1.0, -1)
    return alpha


def clear_mask(alpha, has_alpha, Hm, W, H, p):
    """Footage area to inpaint before pasting, as a uint8 mask in frame coords. With a supplied
    alpha the mask is the alpha's own shape grown by `clear_grow` px, so a round badge does not
    take a rectangle's corners (and the crease above it) with it; without one, the artwork
    rectangle shrunk by `clear` on each side. `clear_shape` picks explicitly: "alpha" | "rect" |
    "auto" (alpha when one is supplied). Lettering whose generated glyphs are bigger or differently
    shaped than the artwork needs "rect": a tight alpha-shaped clear leaves their edges to ghost."""
    ph_, pw_ = alpha.shape[:2]
    mode = p.get("clear_shape", "auto")
    if mode == "alpha" or (mode == "auto" and has_alpha):
        shape = (alpha > 0.5).astype(np.uint8) * 255
        cm = cv2.warpPerspective(shape, Hm, (W, H), flags=cv2.INTER_NEAREST)
        grow = int(round(float(p.get("clear_grow", 2.0))))
        if grow > 0:
            cm = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        return cm
    cr = np.zeros((ph_, pw_), np.uint8)
    mx, my = int(pw_ * p["clear"]), int(ph_ * p["clear"])
    cv2.rectangle(cr, (mx, my), (pw_ - mx, ph_ - my), 255, -1)
    return cv2.dilate(cv2.warpPerspective(cr, Hm, (W, H), flags=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))


def composite(frame, art, alpha, quad, p, has_alpha=False):
    """Paste `art` (with `alpha`) onto `quad` of `frame` in place; returns the new frame."""
    H, W = frame.shape[:2]
    ph_, pw_ = art.shape[:2]
    src = np.array([[0, 0], [pw_, 0], [pw_, ph_], [0, ph_]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    f = frame
    if p["clear"] > 0:
        cm = clear_mask(alpha, has_alpha, Hm, W, H, p)
        f = cv2.inpaint(f, cm, max(1, int(p.get("clear_radius", 5))), cv2.INPAINT_TELEA)
    warped = cv2.warpPerspective(art, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wa = cv2.warpPerspective(alpha, Hm, (W, H), flags=cv2.INTER_LINEAR)
    m = wa > 0.99
    if p["match"] and m.sum() > 50:
        fv = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[m].astype(np.float32)
        av = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)[m].astype(np.float32)
        fb, ab = np.percentile(fv, 30), max(np.percentile(av, 30), 1.0)
        fw, aw = np.percentile(fv, 97), max(np.percentile(av, 97), 1.0)
        gain = float(np.clip((fw - fb) / max(aw - ab, 1.0), 0.5, 1.5))
        off = fb - ab * gain
        warped = np.clip(warped.astype(np.float32) * gain + off, 0, 255).astype(np.uint8)
    if p["blur"] > 0:
        warped = cv2.GaussianBlur(warped, (0, 0), p["blur"])
    if p["feather"] > 0:
        wa = cv2.GaussianBlur(wa, (0, 0), p["feather"])
    wa3 = wa[..., None]
    return (f.astype(np.float32) * (1 - wa3) + warped.astype(np.float32) * wa3).astype(np.uint8)


def read_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, fps


PROOF_SAMPLES = 5


def proof_indices(n, samples=PROOF_SAMPLES):
    return [int(round(i * (n - 1) / (samples - 1))) for i in range(samples)] if n > 1 else [0]


def proof_sheet(before, after, quads_by_patch, path):
    """Zoom strip per patch: original row over patched row at the sampled frames, with the patch's
    MEASURED lines burned in so track quality can be judged without the log. `before` and `after`
    are dicts frame index -> image (only the sampled frames are kept, not the whole clip)."""
    blocks = []
    for name, sm, stats, idx in quads_by_patch:
        rows = []
        banner = np.full((22, 1500, 3), 32, np.uint8)
        cv2.putText(banner, " | ".join(stats), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows.append(banner)
        for src in (before, after):
            tiles = []
            for i in idx:
                q = sm[i]
                cx, cy = q.mean(0)
                w = max(float(np.ptp(q[:, 0])) * 1.6, 64)
                h = max(w * 0.4, float(np.ptp(q[:, 1])) * 2.5)
                H, W = src[i].shape[:2]
                x0, x1 = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
                y0, y1 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
                c = cv2.resize(src[i][y0:y1, x0:x1], (300, int(300 * (y1 - y0) / max(x1 - x0, 1))),
                               interpolation=cv2.INTER_CUBIC)
                cv2.putText(c, f"{name} f{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                tiles.append(c)
            hh = min(t.shape[0] for t in tiles)
            rows.append(np.hstack([t[:hh] for t in tiles]))
        blocks.append(np.vstack(rows))
    ww = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 4), (0, ww - b.shape[1]), (0, 0))) for b in blocks]
    cv2.imwrite(path, np.vstack(blocks))


def main(spec_path):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    emit("PHASE", "decoding")
    frames, fps = read_clip(spec["clip"])
    n = len(frames)
    H, W = frames[0].shape[:2]
    print(f"{n} frames {W}x{H} @ {fps:.3f}", flush=True)
    max_frames = int(spec.get("max_frames") or 720)
    if n > max_frames:
        raise RuntimeError(f"clip has {n} frames, over the {max_frames}-frame cap (whole clip is held in RAM); "
                           f"split it or raise max_frames")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    before = {} if spec.get("proof") else None   # sampled originals, filled per patch range
    debug = spec.get("debug_dir")
    if debug:
        os.makedirs(debug, exist_ok=True)

    tracks = []
    patches = spec["patches"]
    for pi, raw in enumerate(patches):
        p = dict(DEFAULTS, **raw)
        name = p.get("name") or f"patch{pi}"
        emit("PHASE", f"locating {name}")
        art = cv2.imread(p["artwork"], cv2.IMREAD_COLOR)
        if art is None:
            raise RuntimeError(f"{name}: cannot read artwork {p['artwork']}")
        tmpl_gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
        ah, aw = art.shape[:2]
        if p["alpha"]:
            alpha = cv2.imread(p["alpha"], cv2.IMREAD_GRAYSCALE)
            if alpha is None or alpha.shape != (ah, aw):
                raise RuntimeError(f"{name}: alpha must be a grayscale image of the artwork's size")
            alpha = alpha.astype(np.float32) / 255.0
        else:
            alpha = rounded_alpha(ah, aw)
        lo, hi = (0, n - 1) if not p["frames"] else (int(p["frames"][0]), int(p["frames"][1]))
        lo, hi = max(0, lo), min(n - 1, hi)
        if hi < lo:
            raise RuntimeError(f"{name}: frames {p['frames']} is outside the clip ({n} frames)")
        key = int(p["key_frame"]) if int(p["key_frame"]) >= 0 else (lo + hi) // 2
        key = min(max(key, lo), hi)
        if p["key_box"]:
            box, score = tuple(int(v) for v in p["key_box"]), 1.0
        elif p["template"] in ("gray", "edges"):
            box, score = find_element(grays[key], tmpl_gray, p["template"])
            if box is None or score < float(p["min_score"]):
                raise RuntimeError(f"{name}: template match too weak ({score:.2f} < {p['min_score']}); "
                                   f"pass key_box for this clip")
        else:
            raise RuntimeError(f"{name}: needs key_box when template is 'none'")
        x0, y0, x1, y1 = box
        pad = int(0.35 * (x1 - x0))
        aspect = aw / ah
        box_area = (x1 - x0) * (y1 - y0)
        q = None
        if p["refine"]:
            q = dark_quad(grays[key], max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad),
                          0.3 * box_area, aspect=aspect, area_ref=box_area)
        refined = q is not None
        if q is None:
            q = order_quad([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        stats = [f"{name} key_frame={key} score={score:.2f} box={list(box)} refined={refined}"]
        emit("MEASURED", stats[0])

        emit("PHASE", f"tracking {name}")
        span = hi - lo + 1
        quads, misses, lost = track(grays[lo:hi + 1], key - lo, q, aspect, bool(p["refine"]), float(p["win"]))
        sm_span = smooth_quads(quads, int(p["smooth"]))
        sm = [None] * n
        for k in range(span):
            sm[lo + k] = sm_span[k]
        rng = f" frames={lo}-{hi}" if p["frames"] else ""
        stats.append(f"detections={span - misses} flow={misses - lost} held={lost}{rng}")
        emit("MEASURED", f"{name} {stats[1]}")
        if lost > 0.1 * span:
            raise RuntimeError(f"{name}: track lost on {lost} of {span} frames (element leaves frame or too few "
                               f"features); crop the clip, narrow frames, or give a larger win")
        weight = range_weights(n, lo, hi, int(p["fade"]))
        idx = [lo + k for k in proof_indices(span)]
        if before is not None:
            for i in idx:
                before.setdefault(i, frames[i].copy())
        tracks.append((name, p, art, alpha, sm, bool(p["alpha"]), stats, weight, idx))

    emit("PHASE", "compositing")
    for i in range(n):
        f = frames[i]
        for name, p, art, alpha, sm, has_alpha, _, weight, _ in tracks:
            w = float(weight[i])
            if w <= 0.0:
                continue
            g = composite(f, art, alpha, sm[i], p, has_alpha)
            f = g if w >= 1.0 else cv2.addWeighted(g, w, f, 1.0 - w, 0.0)
        frames[i] = f
        if debug and i % 12 == 0:
            dbg = frames[i].copy()
            for name, p, art, alpha, sm, _, _, weight, _ in tracks:
                if weight[i] > 0:
                    cv2.polylines(dbg, [sm[i].astype(np.int32)], True, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(debug, f"track_{i:06d}.png"), dbg)
        if i % 10 == 0:
            emit("PROGRESS", str(int(100 * (i + 1) / n)))

    emit("PHASE", "encoding")
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    out = spec["out"]
    tmp = tempfile.mkdtemp(prefix="planar_", dir=os.path.dirname(os.path.abspath(out)) or None)
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:06d}.png"), f)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:.3f}",
               "-i", os.path.join(tmp, "%06d.png"), "-i", spec["clip"], "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if spec.get("proof"):
        after = {i: frames[i] for i in before}
        proof_sheet(before, after, [(t[0], t[4], t[6], t[8]) for t in tracks], spec["proof"])
    emit("PROGRESS", "100")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
