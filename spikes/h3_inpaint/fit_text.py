"""Robust plate/badge lettering replacement on an H3 clip (spike).

Tracking: per-frame homography by ECC against a FIXED anchor frame (no frame-to-frame
accumulation), warm-started from the neighbouring frame, run on the ORIGINAL footage
(the generated letters + bumper give texture). Falls back to a sequential step only
when the anchor match is weak. Corners are then smoothed in time.

Look: the H3-blanked plate stays as the base (its shading, grain and blur are already
right); only the letters are added: letter mask from the press artwork, colour sampled
from the footage's own lettering, softened to the footage's sharpness, motion-blurred
along each frame's measured velocity, plus matching grain.
"""
import json
import subprocess
import sys

import cv2
import numpy as np


def read(p, n=None):
    cap = cv2.VideoCapture(p)
    out = []
    while True:
        ok, f = cap.read()
        if not ok or (n and len(out) >= n):
            break
        out.append(f)
    return out


def ecc(tmpl, img, mask, warp, iters=120):
    try:
        cc, w = cv2.findTransformECC(tmpl, img, warp.copy(), cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5),
                                     mask, 5)
        return cc, w
    except cv2.error:
        return -1.0, warp


def track(grays, anchor, region, lo, hi, log):
    """Homographies H[i] mapping anchor-frame coords -> frame i, for lo..hi."""
    x0, y0, x1, y1 = region
    mask = np.zeros_like(grays[anchor], np.uint8)
    mask[y0:y1, x0:x1] = 255
    prep = [cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32) for g in grays]
    H = {anchor: np.eye(3, dtype=np.float32)}
    score = {anchor: 1.0}
    for rng in (range(anchor - 1, lo - 1, -1), range(anchor + 1, hi + 1)):
        prev = anchor
        for i in rng:
            cc, w = ecc(prep[anchor], prep[i], mask, H[prev])
            if cc < 0.6:
                # weak anchor match (blur / appearance change): step from the neighbour instead
                m2 = cv2.warpPerspective(mask, H[prev], mask.shape[::-1], flags=cv2.INTER_NEAREST)
                cc2, step = ecc(prep[prev], prep[i], m2, np.eye(3, dtype=np.float32))
                if cc2 > 0.6:
                    w = (step @ H[prev]).astype(np.float32)
                    cc = cc2 * 0.99
                else:
                    w = H[prev]
            H[i] = (w / w[2, 2]).astype(np.float32)
            score[i] = cc
            prev = i
    weak = [i for i, s in score.items() if s < 0.6]
    log(f"tracked {len(H)} frames, weak {len(weak)} {weak[:10]}")
    return H, score


def track_affine(grays, anchor, region, lo, hi, centres, log, shutter=0.5):
    """Affine (6-dof) ECC per frame against the anchor, warm-started from the previous frame
    and shifted by an independent template-match centre track; the anchor template is
    motion-blurred with each frame's own velocity before matching. Returns 3x3 H dict."""
    x0, y0, x1, y1 = region
    mask = np.zeros_like(grays[anchor], np.uint8)
    mask[y0:y1, x0:x1] = 255
    base = cv2.GaussianBlur(grays[anchor], (0, 0), 0.8).astype(np.float32)
    prep = [cv2.GaussianBlur(g, (0, 0), 0.8).astype(np.float32) for g in grays]
    A = {anchor: np.eye(2, 3, dtype=np.float32)}
    score = {anchor: 1.0}
    for rng in (range(anchor - 1, lo - 1, -1), range(anchor + 1, hi + 1)):
        prev = anchor
        for i in rng:
            w = A[prev].copy()
            if i in centres and prev in centres:
                w[:, 2] += np.float32(centres[i]) - np.float32(centres[prev])
            j0, j1 = centres.get(i - 1, centres.get(i)), centres.get(i + 1, centres.get(i))
            v = (np.float32(j1) - np.float32(j0)) / 2 if j0 is not None and j1 is not None else np.zeros(2)
            k = motion_kernel(v[0], v[1], shutter)
            tmpl = cv2.filter2D(base, -1, k) if k is not None else base
            try:
                cc, w2 = cv2.findTransformECC(tmpl, prep[i], w, cv2.MOTION_AFFINE,
                                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 3)
            except cv2.error:
                cc, w2 = -1.0, w
            # reject implausible jumps in scale/shear vs the previous frame
            sc_prev = np.sqrt(abs(np.linalg.det(A[prev][:, :2])))
            sc_new = np.sqrt(abs(np.linalg.det(w2[:, :2])))
            if cc < 0.5 or abs(sc_new / sc_prev - 1) > 0.06:
                w2, cc = w, min(cc, 0.49)
            A[i], score[i] = w2, float(cc)
            prev = i
    H = {i: np.vstack([a, [0, 0, 1]]).astype(np.float32) for i, a in A.items()}
    weak = sorted(i for i, s in score.items() if s < 0.5)
    log(f"affine tracked {len(H)} frames, weak {len(weak)} {weak[:15]}")
    return H, score


def track_local(grays, anchor, region, lo, hi, centres, log, shutter=0.5, search=24, motion="affine"):
    """Affine/homography ECC in plate-centred windows: template = anchor crop of `region`,
    input = a window around the predicted position in frame i. Conditioning is good because
    the warp acts about the window origin, not the image corner. The anchor template is
    motion-blurred with frame i's velocity. Returns H dict mapping anchor -> frame coords."""
    x0, y0, x1, y1 = region
    tw, th = x1 - x0, y1 - y0
    tmpl0 = cv2.GaussianBlur(grays[anchor], (0, 0), 0.7)[y0:y1, x0:x1].astype(np.float32)
    mt = cv2.MOTION_HOMOGRAPHY if motion == "homography" else cv2.MOTION_AFFINE
    def tr(dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32)
    H = {anchor: np.eye(3, dtype=np.float32)}
    score = {anchor: 1.0}
    Hh, Wh = grays[0].shape
    for rng in (range(anchor - 1, lo - 1, -1), range(anchor + 1, hi + 1)):
        prev = anchor
        for i in rng:
            # predicted top-left of the template in frame i: previous solution + centre-track delta
            Hp = H[prev].copy()
            d = (np.float32(centres[i]) - np.float32(centres[prev])) if (i in centres and prev in centres) else np.zeros(2)
            Hp = tr(d[0], d[1]) @ Hp
            # window origin in frame i
            tl = cv2.perspectiveTransform(np.float32([[[x0, y0]]]), Hp).reshape(2)
            ox = int(round(tl[0])) - search
            oy = int(round(tl[1])) - search
            ox = min(max(0, ox), Wh - tw - 2 * search); oy = min(max(0, oy), Hh - th - 2 * search)
            inp = cv2.GaussianBlur(grays[i], (0, 0), 0.7)[oy:oy + th + 2 * search, ox:ox + tw + 2 * search].astype(np.float32)
            # local warp init: template coords -> window coords
            W0 = tr(-ox, -oy) @ Hp @ tr(x0, y0)
            W0 = W0 / W0[2, 2]
            j0, j1 = centres.get(i - 1, centres.get(i)), centres.get(i + 1, centres.get(i))
            v = (np.float32(j1) - np.float32(j0)) / 2 if (j0 is not None and j1 is not None) else np.zeros(2)
            k = motion_kernel(v[0], v[1], shutter)
            tmpl = cv2.filter2D(tmpl0, -1, k) if k is not None else tmpl0
            init = W0.astype(np.float32) if mt == cv2.MOTION_HOMOGRAPHY else W0[:2].astype(np.float32)
            try:
                cc, w = cv2.findTransformECC(tmpl, inp, init, mt,
                                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 3)
            except cv2.error:
                cc, w = -1.0, init
            W = w if w.shape[0] == 3 else np.vstack([w, [0, 0, 1]])
            Hn = (tr(ox, oy) @ W @ tr(-x0, -y0)).astype(np.float32)
            Hn /= Hn[2, 2]
            sc_p = np.sqrt(abs(np.linalg.det(H[prev][:2, :2])))
            sc_n = np.sqrt(abs(np.linalg.det(Hn[:2, :2])))
            if cc < 0.6 or abs(sc_n / sc_p - 1) > 0.05:
                Hn, cc = Hp, min(cc, 0.59)
            H[i], score[i] = Hn, float(cc)
            prev = i
    weak = sorted(i for i, s in score.items() if s < 0.6)
    log(f"local {motion} tracked {len(H)} frames, weak {len(weak)} {weak[:15]}")
    return H, score


def corners_track(H, quad, lo, hi, smooth=2, log=print, centres=None):
    q = np.float32(quad).reshape(-1, 1, 2)
    C = np.array([cv2.perspectiveTransform(q, H[i]).reshape(4, 2) for i in range(lo, hi + 1)])
    # sanity: area within 0.5-1.6x of the anchor quad and convex; reject + interpolate the rest
    a0 = cv2.contourArea(np.float32(quad))
    ok = np.array([cv2.isContourConvex(c.astype(np.float32)) and 0.5 < cv2.contourArea(c.astype(np.float32)) / a0 < 1.6
                   for c in C])
    # and no centre jump > 3x the median step
    ctr = C.mean(1)
    step = np.r_[0, np.linalg.norm(np.diff(ctr, axis=0), axis=1)]
    med = np.median(step[step > 0]) if (step > 0).any() else 1
    ok &= step < max(25.0, 4 * med)
    bad = [lo + k for k in np.where(~ok)[0]]
    if bad:
        log(f"rejected {len(bad)} frames: {bad[:20]}")
        idx = np.arange(len(C))
        good = idx[ok]
        for k in idx[~ok]:
            ref = good[np.argmin(np.abs(good - k))]
            if centres and (lo + k) in centres and (lo + ref) in centres:
                C[k] = C[ref] + (np.float32(centres[lo + k]) - np.float32(centres[lo + ref]))
            else:
                C[k] = C[ref]
    if smooth > 0:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        pad = np.pad(C, ((smooth, smooth), (0, 0), (0, 0)), mode="edge")
        C = np.stack([np.stack([np.convolve(pad[:, c, d], k, mode="valid") for d in range(2)], 1)
                      for c in range(4)], 1)
    return {lo + k: C[k] for k in range(len(C))}


def motion_kernel(vx, vy, shutter=0.5):
    L = float(np.hypot(vx, vy)) * shutter
    if L < 1.0:
        return None
    n = int(np.ceil(L)) | 1
    k = np.zeros((n * 2 + 1, n * 2 + 1), np.float32)
    c = n
    for t in np.linspace(-0.5, 0.5, 4 * n + 1):
        k[int(round(c + t * vy * shutter)), int(round(c + t * vx * shutter))] += 1
    return k / k.sum()


def add_letters(frame, quad, letter_mask, colour, soften, kern, grain, rng, ss=4):
    """Warp a letters-only mask into the quad (supersampled), soften + motion-blur it,
    and lay `colour` over the frame through it."""
    h, w = letter_mask.shape
    x0, y0 = np.floor(quad.min(0)).astype(int) - 12
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 12
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return
    dst = (quad - [x0, y0]) * ss
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(dst))
    big = cv2.warpPerspective(letter_mask.astype(np.float32) / 255.0, M, (bw * ss, bh * ss),
                              flags=cv2.INTER_LINEAR)
    a = cv2.resize(big, (bw, bh), interpolation=cv2.INTER_AREA)
    if soften > 0:
        a = cv2.GaussianBlur(a, (0, 0), soften)
    if kern is not None:
        a = cv2.filter2D(a, -1, kern)
    a = np.clip(a, 0, 1)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    col = np.ones_like(roi) * colour
    if grain > 0:
        col += rng.normal(0, grain, roi.shape[:2])[..., None]
    frame[y0:y1, x0:x1] = np.clip(roi * (1 - a) + col * a, 0, 255).astype(np.uint8)


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    log = lambda *a: print(*a, flush=True)
    orig = read(cfg["original"], cfg["frames"])
    base = read(cfg["base"], cfg["frames"])
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in orig]
    out = [f.copy() for f in base]
    rng = np.random.default_rng(0)
    report = {}
    for el in cfg["elements"]:
        lo, hi = el["range"]
        cache = f"{cfg['out']}.{el['name']}.track.npz"
        try:
            z = np.load(cache)
            H = {int(k): z[k] for k in z.files if k.isdigit()}
            score = {int(k): float(v) for k, v in zip(z["si"], z["sv"])}
            log(el["name"], "track loaded from cache")
        except (FileNotFoundError, OSError):
            if el.get("model") in ("local", "local_h"):
                cb = json.load(open(el["centres"]))
                centres = {int(k): ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2) for k, v in cb.items()}
                H, score = track_local(grays, el["anchor"], el["region"], lo, hi, centres, log,
                                       el.get("shutter", 0.5), el.get("search", 24),
                                       "homography" if el["model"] == "local_h" else "affine")
            elif el.get("model") == "affine":
                cb = json.load(open(el["centres"]))
                centres = {int(k): ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2) for k, v in cb.items()}
                H, score = track_affine(grays, el["anchor"], el["region"], lo, hi, centres, log,
                                        el.get("shutter", 0.5))
            else:
                H, score = track(grays, el["anchor"], el["region"], lo, hi, log)
            np.savez(cache, si=np.array(list(score.keys())), sv=np.array(list(score.values())),
                     **{str(k): v for k, v in H.items()})
        cen = None
        if el.get("centres"):
            cb = json.load(open(el["centres"]))
            cen = {int(k): ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2) for k, v in cb.items()}
        C = corners_track(H, el["quad"], lo, hi, el.get("smooth", 2), log, cen)
        lm = cv2.imread(el["letters"], cv2.IMREAD_GRAYSCALE)
        colour = np.array(el["colour"], np.float32)
        quads = {}
        for i in range(lo, hi + 1):
            q = C[i]
            prv, nxt = C.get(i - 1, q), C.get(i + 1, q)
            v = (nxt.mean(0) - prv.mean(0)) / (2 if (i - 1 in C and i + 1 in C) else 1)
            kern = motion_kernel(v[0], v[1], el.get("shutter", 0.5))
            if el.get("clear_mask_from_original"):
                # inpaint the generated letters out of the base (badge: not H3-blanked)
                m = np.zeros(grays[i].shape, np.uint8)
                cv2.fillConvexPoly(m, np.int32(np.round(q)), 255)
                m = cv2.dilate(m, np.ones((5, 5), np.uint8))
                out[i] = cv2.inpaint(out[i], m, 4, cv2.INPAINT_TELEA)
            add_letters(out[i], q, lm, colour, el.get("soften", 0.8), kern, el.get("grain", 3.0), rng)
            quads[i] = q.round(2).tolist()
        report[el["name"]] = {"score_min": round(min(score.values()), 3), "quads": quads}
        log(el["name"], "min ECC", round(min(score.values()), 3))
    json.dump(report, open(cfg["report"], "w"))
    h, w = out[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", cfg["out"]],
                         stdin=subprocess.PIPE)
    for f in out:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    log("wrote", cfg["out"])


if __name__ == "__main__":
    main(sys.argv[1])
