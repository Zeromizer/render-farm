"""Kinetic-type measurement: HOW each on-screen text event enters and leaves.

The text stage (2 fps OCR) says which words were on screen and roughly when.
Word animations live between those samples (0.1-0.5 s), and OCR cannot read
text while it moves or blurs, so this stage goes back to the video at its
native frame rate around each event's entrance and exit and follows the
SETTLED text block through them with three measures, all taken against the
text's own pixels so a moving background does not read as motion:

  presence  the text's local contrast against a thin ring of background just
            around its letters, as a fraction of the settled contrast
            (0 = not there, 1 = fully there). Per half and per word too.
  place     where the text's pattern is, and at what scale (masked template
            match over the text row, multi-scale).
  sharpness Laplacian energy on the letters vs settled (blur-in, motion blur).

Those curves become labels — cut, fade, slide, whip, scale_up (pop),
scale_down (slam), wipe, stagger, blur_in — with duration, direction, easing
and overshoot. The extractor's rule applies: a wrong label is worse than a
missing one, so anything ambiguous is "unknown", and a strip image of the
transition lets a person or a model judge it by eye.

What to trust: on synthetic clips the labels score 44/46. On real footage
(handheld captions, text riding whips and cuts) the TIMINGS hold — onset,
settle, carried by a cut or a scene move — but the mechanism labels are
noisy, so they ship as hints with a confidence that drops over a busy
background, and the annotate pass names the mechanism from the strips.
"Pixels for facts, model for judgement."

Pure compute: numpy + cv2 only. Tune against kinetic_bench/ (synthetic clips
with known entrances and exits, and their scorer) before shipping a change.
"""
import math
import re

import cv2
import numpy as np

WORK_W = 360           # analysis width; height follows the aspect
ENTRY_BEFORE = 1.2     # seconds before the OCR onset to look for the real start
ENTRY_AFTER = 0.75     # ...and after it, to see the text settle
EXIT_BEFORE = 0.75
EXIT_AFTER = 1.2
PRESENT = 0.85         # presence at/above this = fully there
ABSENT = 0.15          # presence at/below this = not there
MATCH_OK = 0.72        # template score that counts as "found"
SCALES = (0.3, 0.45, 0.6, 0.75, 0.88, 1.0, 1.12, 1.3, 1.6, 2.0, 2.6)


# ----------------------------------------------------------------- frames

def read_window(video, t0, t1, native_fps, width=WORK_W):
    """Colour frames over [t0, t1) at the native rate, resized to `width`.
    Seeks by frame INDEX and derives times from it: CAP_PROP_POS_MSEC after a
    seek is unreliable enough to shift every timestamp by ~0.3 s."""
    cap = cv2.VideoCapture(video)
    frames, times = [], []
    try:
        i0 = max(0, int(round(t0 * native_fps)))
        i1 = int(round(t1 * native_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        for i in range(i0, i1):
            ok, frame = cap.read()
            if not ok:
                break
            h = round(frame.shape[0] * width / frame.shape[1])
            frames.append(cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA))
            times.append(i / native_fps)
    finally:
        cap.release()
    return frames, times


def _gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _grad(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


# ----------------------------------------------------------------- the text itself

def _box_px(bbox, w, h, pad=0.15):
    x0, y0, x1, y1 = bbox
    bw, bh = (x1 - x0), (y1 - y0)
    x0, x1 = x0 - bw * pad, x1 + bw * pad
    y0, y1 = y0 - bh * pad, y1 + bh * pad
    return (max(0, int(x0 * w)), max(0, int(y0 * h)), min(w, int(math.ceil(x1 * w))), min(h, int(math.ceil(y1 * h))))


class Text:
    """The settled text block: its pixels (core), the background ring around
    them, its polarity and its settled contrast."""

    def __init__(self, settled_frames, empty_frames, box, n_words=None):
        x0, y0, x1, y1 = box
        self.box = box
        crops = np.stack([_gray(f)[y0:y1, x0:x1] for f in settled_frames])
        self.settled = np.median(crops, axis=0)
        steady = crops.std(axis=0) < 14
        empties = [_gray(f)[y0:y1, x0:x1] for f in empty_frames]
        # a text pixel differs from EVERY text-free frame (min over them: a
        # moving background differs from one or another, the text from all).
        # Text on screen from frame 0 is given only exit-side empties.
        differs = np.min([np.abs(self.settled - e) for e in empties], axis=0) > 30
        cand = (steady & differs).astype(np.uint8)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        self.core = cand > 0
        d3 = cv2.dilate(cand, np.ones((3, 3), np.uint8))
        d6 = cv2.dilate(cand, np.ones((7, 7), np.uint8))
        self.ring = (d6 > 0) & ~(d3 > 0)
        self.shape = (d6 > 0)  # core + ring: the pattern to match
        self.ok = self.core.sum() >= 20 and self.ring.sum() >= 20
        if not self.ok:
            return
        self.pol = 1.0 if self.settled[self.core].mean() >= self.settled[self.ring].mean() else -1.0
        self.c_settled = self._contrast(self.settled, self.core, self.ring)
        self.ok = self.c_settled > 15
        self.sharp = sharpness(self.settled, self.core)
        self.words = merge_to_count(word_columns(self.core), n_words)
        # the scene around the text: a box one text-height above and below and
        # a quarter-width either side, the text itself masked out. If THIS
        # arrives with the text, the text is riding a scene transition.
        H, W = settled_frames[0].shape[:2]
        bh, bw = y1 - y0, x1 - x0
        self.ctx_box = (max(0, x0 - bw // 4), max(0, y0 - bh), min(W, x1 + bw // 4), min(H, y1 + bh))
        cx0, cy0, cx1, cy1 = self.ctx_box
        ctx = np.stack([_gray(f)[cy0:cy1, cx0:cx1] for f in settled_frames])
        self.ctx_settled = np.median(ctx, axis=0)
        cm = np.ones(self.ctx_settled.shape, bool)
        core_d = cv2.dilate(self.core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        cm[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] &= ~core_d
        self.ctx_mask = cm
        self.ctx_steady = min(self.ctx_similarity_crop(c) for c in ctx) if len(ctx) else 0.0

    def ctx_similarity_crop(self, crop):
        return self.ctx_pair(crop, self.ctx_settled)

    def ctx_pair(self, a, b):
        """Similarity of two context crops over the context mask."""
        a, b = a[self.ctx_mask], b[self.ctx_mask]
        if a.size < 30 or a.std() < 3 or b.std() < 3:
            return 1.0 if a.size >= 30 and abs(a.mean() - b.mean()) < 12 else 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def ctx_similarity(self, frame_gray):
        cx0, cy0, cx1, cy1 = self.ctx_box
        return self.ctx_similarity_crop(frame_gray[cy0:cy1, cx0:cx1])

    def _contrast(self, g, core, ring):
        if core.sum() < 5 or ring.sum() < 5:
            return 0.0
        return float(self.pol * (g[core].mean() - np.median(g[ring])))

    def presence(self, crop, part=None):
        core, ring = self.core, self.ring
        if part is not None:
            core, ring = core & part, ring & part
        if self.c_settled <= 0:
            return 0.0
        ref = self._contrast(self.settled, core, ring) or self.c_settled
        return max(0.0, min(1.3, self._contrast(crop, core, ring) / ref))

    def halves(self):
        h, w = self.core.shape
        parts = {}
        for name, sl in (("left", (slice(None), slice(0, w // 2))), ("right", (slice(None), slice(w // 2, None))),
                         ("top", (slice(0, h // 2), slice(None))), ("bottom", (slice(h // 2, None), slice(None)))):
            m = np.zeros_like(self.core)
            m[sl] = True
            parts[name] = m
        return parts


def word_columns(mask):
    """Split the mask into word-ish column spans by vertical gaps."""
    cols = mask.any(axis=0)
    h = mask.shape[0]
    gap_needed = max(3, int(h * 0.12))
    spans, start, gap = [], None, 0
    for x, on in enumerate(cols):
        if on:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= gap_needed:
                spans.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None:
        spans.append((start, len(cols)))
    return [s for s in spans if s[1] - s[0] >= 3]


def merge_to_count(spans, n):
    """Column spans split on pixel gaps can cut one word in two (wide
    letter-spacing, a thin glyph, a stroke lost in the background). When the
    OCR says the line has n words, close the narrowest gaps until there are n."""
    spans = list(spans)
    if not n or n < 1:
        return spans
    while len(spans) > n:
        k = min(range(len(spans) - 1), key=lambda i: spans[i + 1][0] - spans[i][1])
        spans[k:k + 2] = [(spans[k][0], spans[k + 1][1])]
    return spans


def sharpness(gray_crop, mask):
    lap = np.abs(cv2.Laplacian(gray_crop, cv2.CV_32F))
    return float(lap[mask].mean()) if mask.any() else 0.0


def locate(frame_gray, text, band_pad=0.2, streak=False):
    """Where the settled pattern is in this frame, and at what scale: masked
    TM_CCOEFF_NORMED over the text's row (whole width — whips come from
    off-screen). With streak=True the template is also tried motion-blurred
    along x and y at scale 1, for a whip too smeared to match sharp.
    Returns (score, cx, cy, scale, streak_axis or None)."""
    H, W = frame_gray.shape
    x0, y0, x1, y1 = text.box
    tmpl = text.settled
    m = text.shape.astype(np.uint8)
    cy_anchor = (y0 + y1) / 2
    best = (-1.0, (x0 + x1) / 2, cy_anchor, 1.0, None)
    variants = [(s, None, 0) for s in SCALES]
    if streak:
        variants += [(1.0, "x", k) for k in (9, 17, 31)] + [(1.0, "y", k) for k in (7, 13, 23)]
    for s, axis, k in variants:
        tw, th = int(tmpl.shape[1] * s), int(tmpl.shape[0] * s)
        if tw < 8 or th < 6 or tw >= W or th >= H:
            continue
        t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
        if axis:
            kern = np.ones((1, k), np.float32) / k if axis == "x" else np.ones((k, 1), np.float32) / k
            t = cv2.filter2D(t, -1, kern)
            mm = cv2.dilate(mm, np.ones((1, k) if axis == "x" else (k, 1), np.uint8))
        if mm.sum() < 30 or t[mm > 0].std() < 6:
            continue
        ry0 = max(0, int(cy_anchor - th / 2 - band_pad * H))
        ry1 = min(H, int(cy_anchor + th / 2 + band_pad * H))
        region = frame_gray[ry0:ry1]
        if region.shape[0] < th:
            continue
        res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED, mask=mm)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        res[res > 1.0001] = -1.0   # degenerate windows (flat patches) report nonsense
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), loc[0] + tw / 2, ry0 + loc[1] + th / 2, s, axis)
    if streak:
        e = locate_edge(frame_gray, text, band_pad)
        if e[0] > best[0]:
            best = e
    return best


def locate_edge(frame_gray, text, band_pad=0.2):
    """The text hanging off a frame edge, part-way through a slide/whip in or
    out: match only the part that would be visible (the leading 70% or 45% of
    the text) against the band, and keep the hit only if the implied full box
    really does cross that edge. Returns (score, cx, cy, 1.0, "edge")."""
    H, W = frame_gray.shape
    x0, y0, x1, y1 = text.box
    tmpl, m = text.settled, text.shape.astype(np.uint8)
    th, tw = tmpl.shape
    cy_anchor = (y0 + y1) / 2
    best = (-1.0, (x0 + x1) / 2, cy_anchor, 1.0, "edge")
    ry0 = max(0, int(cy_anchor - th / 2 - band_pad * H))
    ry1 = min(H, int(cy_anchor + th / 2 + band_pad * H))
    region = frame_gray[ry0:ry1]
    if region.shape[0] < th:
        return best
    for frac in (0.7, 0.45):
        k = max(8, int(tw * frac))
        for part in ("left", "right"):
            keep = (slice(None), slice(0, k)) if part == "left" else (slice(None), slice(tw - k, tw))
            t, mm = tmpl[keep], m[keep]
            if mm.sum() < 30 or t[mm > 0].std() < 8:
                continue
            res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED, mask=mm)
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            res[res > 1.0001] = -1.0
            _, mx, _, loc = cv2.minMaxLoc(res)
            if part == "left":
                cx = loc[0] + tw / 2            # its right part is off the right edge
                crosses = cx + tw / 2 > W + 2
            else:
                cx = loc[0] + k - tw / 2        # its left part is off the left edge
                crosses = cx - tw / 2 < -2
            if crosses and mx > best[0]:
                best = (float(mx), cx, ry0 + loc[1] + th / 2, 1.0, "edge")
    return best


# ----------------------------------------------------------------- one transition

def _first(xs, pred):
    for i, x in enumerate(xs):
        if pred(x):
            return i
    return None


def _last(xs, pred):
    for i in range(len(xs) - 1, -1, -1):
        if pred(xs[i]):
            return i
    return None


def _easing(progress, times):
    """From how much of the change is done a third of the way through the
    time: ~0.70 for a cubic ease-out, 0.33 linear, ~0.04 cubic ease-in, and an
    ease-in-out sits low at 1/3 but high at 2/3. Robust to the measured end
    being a little early, which the half-way-time test is not."""
    if len(progress) < 3 or times[-1] - times[0] <= 0:
        return "unknown"
    t0, t1 = times[0], times[-1]

    def at(frac):
        tt = t0 + (t1 - t0) * frac
        return float(np.interp(tt, times, progress))

    p3, p6 = at(1 / 3), at(2 / 3)
    if p3 >= 0.48:
        return "ease_out"
    if p3 <= 0.2:
        return "ease_in_out" if p6 >= 0.6 else "ease_in"
    return "linear"


CURVE_MIN_POINTS = 5   # a move needs this many samples before its curve is worth fitting
CURVE_MAX_RMSE = 0.08  # ...and must fit at least this well to ship
CURVE_MAX_STEP = 0.75  # ...and no single frame step may carry more of it than this
# standard curves, preferred when they fit nearly as well as the free fit
NAMED_CURVES = {
    "linear": [0.0, 0.0, 1.0, 1.0],
    "ease": [0.25, 0.1, 0.25, 1.0],
    "ease_in": [0.42, 0.0, 1.0, 1.0],
    "ease_out": [0.0, 0.0, 0.58, 1.0],
    "ease_in_out": [0.42, 0.0, 0.58, 1.0],
    "ease_in_cubic": [0.32, 0.0, 0.67, 0.0],
    "ease_out_cubic": [0.33, 1.0, 0.68, 1.0],
    "ease_in_out_cubic": [0.65, 0.0, 0.35, 1.0],
    "ease_out_quint": [0.22, 1.0, 0.36, 1.0],
    "ease_out_expo": [0.16, 1.0, 0.3, 1.0],
    "ease_out_back": [0.34, 1.56, 0.64, 1.0],
}


def _bezier_y(params, t, iters=22):
    """CSS cubic-bezier(x1, y1, x2, y2) evaluated at times t (0..1), for many
    parameter sets at once: params (P, 4), t (n,) -> (P, n). x(u) is
    monotonic when x1, x2 are in [0, 1], so u is found by bisection."""
    x1, y1, x2, y2 = (params[:, k:k + 1] for k in range(4))
    lo = np.zeros((params.shape[0], t.size))
    hi = np.ones_like(lo)
    for _ in range(iters):
        u = (lo + hi) / 2
        x = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u ** 2 * x2 + u ** 3
        below = x < t[None, :]
        lo = np.where(below, u, lo)
        hi = np.where(below, hi, u)
    u = (lo + hi) / 2
    return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u ** 2 * y2 + u ** 3


def fit_bezier(times, progress, reverse=False):
    """The CSS cubic-bezier that best follows a measured progress curve
    (0 = not started, 1 = settled; overshoot allowed). Coarse grid, then a
    finer one around the best. `reverse` reads an exit: the frames were
    walked backwards, so real time is 1 - t and 'how far it has left' is
    1 - progress. Returns ([x1, y1, x2, y2], rmse, standard name or None),
    or None when there are too few samples or no curve fits well: a wrong
    curve is worse than none."""
    by_t = {}
    for tt, pp in zip(times, progress):   # one sample per frame time, the later one wins
        if math.isfinite(tt) and math.isfinite(pp):
            by_t[round(float(tt), 4)] = float(pp)
    t = np.array(list(by_t.keys()))
    p = np.array(list(by_t.values()))
    if t.size < CURVE_MIN_POINTS or t.max() - t.min() <= 0:
        return None
    tn = (t - t.min()) / (t.max() - t.min())
    if reverse:
        tn, p = 1 - tn, 1 - p
    order = np.argsort(tn)
    tn, p = tn[order], p[order]
    # the move over the MEASURED window, 0 at its first frame and 1 at its
    # last: the first frame is usually already part-way (a fade is ~17%
    # visible by the frame it is first seen) and forcing 0 there bends a
    # straight line into an S. The curve is replayed over the measured duration.
    lo, hi = float(p[0]), float(p[-1])
    if hi - lo < 0.3:
        return None
    p = np.clip((p - lo) / (hi - lo), -0.3, 1.6)
    # nearly all of it in one frame step is a jump, not a curve — typically a
    # window that opened early over footage (flat, then the text lands); a
    # curve fitted to that would teach "hold, then snap" (real pop overshoot
    # takes ~70% in its first frame, so the bar sits above that)
    if np.max(np.abs(np.diff(p))) > CURVE_MAX_STEP:
        return None

    def best_of(xs1, ys1, xs2, ys2):
        grid = np.array(np.meshgrid(xs1, ys1, xs2, ys2, indexing="ij")).reshape(4, -1).T
        err = np.sqrt(np.mean((_bezier_y(grid, tn) - p[None, :]) ** 2, axis=1))
        k = int(np.argmin(err))
        return grid[k], float(err[k])

    xs, ys = np.linspace(0, 1, 11), np.linspace(-0.4, 1.6, 11)
    (a, b, c, d), _ = best_of(xs, ys, xs, ys)
    fine = lambda v, lo, hi, step: np.clip(np.linspace(v - step, v + step, 5), lo, hi)
    (a, b, c, d), err = best_of(fine(a, 0, 1, 0.1), fine(b, -0.4, 1.6, 0.2), fine(c, 0, 1, 0.1), fine(d, -0.4, 1.6, 0.2))
    if err > CURVE_MAX_RMSE:
        return None
    # bezier parameters are not unique (a straight line fits as [0.85, 0.9,
    # 0.95, 0.9] too): prefer a standard curve that fits nearly as well, so the
    # number a builder reads is one it recognises
    named = np.array(list(NAMED_CURVES.values()), dtype=float)
    n_err = np.sqrt(np.mean((_bezier_y(named, tn) - p[None, :]) ** 2, axis=1))
    k = int(np.argmin(n_err))
    if n_err[k] <= min(CURVE_MAX_RMSE, err + 0.015):
        return list(NAMED_CURVES.values())[k], round(float(n_err[k]), 3), list(NAMED_CURVES)[k]
    return [round(float(v), 2) for v in (a, b, c, d)], round(err, 3), None


SUDDEN_STEP = 0.3     # frame-to-frame context dissimilarity (1 - corr) that counts as a jump


def _sudden_context(grays, text, onset_i, settled_i):
    """Did the scene around the text JUMP during the move? A whip, push, flash
    or cut changes it in one or two big frame-to-frame steps; a walking or
    handheld camera changes it as much in total but in small, even steps, and
    that must not read as 'carried by the scene' (found on 'Great text
    animation', a presenter filmed walking: 7 false scene flags)."""
    cx0, cy0, cx1, cy1 = text.ctx_box
    crops = [g[cy0:cy1, cx0:cx1] for g in grays]
    steps = [1.0 - text.ctx_pair(crops[i - 1], crops[i]) for i in range(1, len(crops))]
    # steps[j - 1] is the change INTO frame j
    inside = steps[max(0, onset_i - 1):max(onset_i, settled_i)]
    outside = [v for j, v in enumerate(steps, start=1) if j < onset_i - 1 or j > settled_i + 1]
    base = float(np.median(outside)) if outside else 0.0
    return bool(inside) and max(inside) >= max(SUDDEN_STEP, 4 * base)


def analyse_transition(frames, times, text, fps, entering, cuts=()):
    """One entrance (entering=True) or exit. For an exit the frames are
    reversed so both read 'absent -> settled' and share the logic."""
    if not entering:
        frames, times = frames[::-1], times[::-1]
    n = len(frames)
    x0, y0, x1, y1 = text.box
    grays = [_gray(f) for f in frames]
    crops = [g[y0:y1, x0:x1] for g in grays]
    pres = [text.presence(c) for c in crops]
    raw_sharp = [sharpness(c, text.core) / (text.sharp or 1.0) for c in crops]
    # sharpness relative to how much of the text is there: a fade lowers both
    # together (ratio ~1), a blur lowers sharpness far more than contrast
    rel_sharp = [s / max(p, 0.25) for s, p in zip(raw_sharp, pres)]
    final = float(np.median(pres[-4:]))
    if final < 0.6:
        return {"type": "unknown", "why": "never settles in the window"}

    Hf, Wf = grays[0].shape
    ax, ay = (x0 + x1) / 2, (y0 + y1) / 2
    first_seen = _first(pres, lambda v: v > 0.05)
    start = max(0, (first_seen if first_seen is not None else n - 1) - int(0.5 * fps))
    track = {}
    for i in range(start, n):
        sc, cx, cy, s, axis = locate(grays[i], text)
        if sc < MATCH_OK and pres[i] < PRESENT:
            sc2, cx2, cy2, s2, ax2 = locate(grays[i], text, streak=True)
            if sc2 > max(sc, 0.62):
                sc, cx, cy, s, axis = sc2, cx2, cy2, s2, ax2
        track[i] = (sc, cx, cy, s, axis)

    def found(i):
        sc, _, _, _, axis = track.get(i, (-1, 0, 0, 1, None))
        return sc >= (0.62 if axis else MATCH_OK)

    def off(i):
        _, cx, cy, s, _ = track[i]
        return math.hypot((cx - ax) / Wf, (cy - ay) / Hf), s

    # presence noise over this background: footage behind the text makes the
    # contrast measure wobble, so "absent" is judged against the wobble seen
    # before the text arrives rather than a fixed 0.15
    pre = pres[:max(3, n // 4)]
    absent = max(ABSENT, float(np.percentile(pre, 90)) + 0.1) if len(pre) >= 3 else ABSENT
    absent = min(absent, 0.45 * final)
    # a busy background (footage wobbling the contrast measure, or a scene
    # around the text that is itself moving) is where the labels go wrong
    busy = absent > 0.3 or text.ctx_steady < 0.8
    sharp_ref = float(np.median(raw_sharp[-4:]))

    def bad(i):
        if pres[i] < 0.93 * final or raw_sharp[i] < 0.88 * sharp_ref:
            return True
        if found(i):
            d, s = off(i)
            return d > 0.011 or abs(s - 1) > 0.05
        return False

    last_bad = _last(list(range(n)), bad)
    settled_i = (last_bad + 1) if last_bad is not None else 0
    if settled_i >= n:
        return {"type": "unknown", "why": "never settles in the window"}

    # onset: walk back while the text is somewhere — in place or found elsewhere
    onset_i = settled_i
    while onset_i - 1 >= start and (pres[onset_i - 1] > absent or found(onset_i - 1)):
        onset_i -= 1
    if onset_i == settled_i and settled_i > 0 and pres[settled_i - 1] > absent:
        onset_i = settled_i - 1
    # the walk ran out of window: the text (or a look-alike the OCR split it
    # from) was already up when the window opened. Not a cut at the window edge.
    if onset_i <= 0 and pres[0] > absent:
        return {"type": "unknown", "why": "already on screen when the window opens"}

    # captions that drift or zoom slowly all the while they are up never meet
    # the strict "holds still" test until late: when the text first reaches
    # (and keeps) nearly full presence well before that, the move is over there
    k = _first(list(range(onset_i, settled_i + 1)),
               lambda i: all(pres[j] >= 0.9 * final for j in range(i, min(n, i + 3))))
    reach = None if k is None else onset_i + k   # _first returns a position, not a frame index
    if reach is not None and (settled_i - reach) > 0.25 * fps:
        settled_i = reach

    frames_n = settled_i - onset_i
    dur = abs(times[settled_i] - times[onset_i])

    # carried by the scene? The context around the text is steady once the
    # text is settled, different just before the text starts, and settles
    # when the text does — a whip/push/flash of the whole shot, not a text
    # animation. Or a shot cut lands on the text's first frame.
    kind = "text"
    ctx = [text.ctx_similarity(g) for g in grays]
    if text.ctx_steady >= 0.8:
        before = ctx[max(0, onset_i - 1)]
        c_low = _last(ctx, lambda v: v < 0.8)
        ctx_settle = (c_low + 1) if c_low is not None else 0
        if (before < 0.55 and abs(ctx_settle - settled_i) <= max(3, frames_n // 2 + 2)
                and _sudden_context(grays, text, onset_i, settled_i)):
            kind = "scene"
    # ...or the WHOLE picture changes suddenly as the text arrives (a flash,
    # push or whip of the shot), which the local context can miss on a flat
    # card. Frame-to-frame change outside the text box, 64 px wide: a spike
    # inside the transition well above the window's usual change, and the
    # frame before the move unlike the settled one.
    if kind == "text":
        small = []
        for g in grays:
            m = g.copy()
            m[y0:y1, x0:x1] = np.nan
            h = max(8, int(64 * g.shape[0] / g.shape[1]))
            small.append(cv2.resize(np.nan_to_num(m, nan=float(np.nanmean(m))), (64, h), interpolation=cv2.INTER_AREA))
        step = [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, n)]
        inside = [step[i - 1] for i in range(max(1, onset_i), settled_i + 1)]
        outside = [v for j, v in enumerate(step, start=1) if j < onset_i - 1 or j > settled_i + 1]
        base = float(np.median(outside)) if outside else 0.0
        a, b = small[max(0, onset_i - 1)].ravel(), small[settled_i].ravel()
        unlike = (np.corrcoef(a, b)[0, 1] < 0.6) if a.std() > 2 and b.std() > 2 else abs(a.mean() - b.mean()) > 15
        # only on a steady shot: a handheld or moving camera changes the whole
        # frame all the time, and its real cuts are in the shot list anyway
        if inside and base < 2.5 and max(inside) > 4 * base + 3 and unlike:
            kind = "scene"
    t_on = times[onset_i]
    if any(abs(t_on - c) <= 1.5 / fps for c in cuts):
        kind = "cut"
    out = {"t_start": round(float(times[onset_i] if entering else times[settled_i]), 3),
           "t_end": round(float(times[settled_i] if entering else times[onset_i]), 3),
           "duration_s": round(dur, 3), "frames": int(frames_n), "carried_by": None if kind == "text" else kind,
           "background": "busy" if busy else "steady"}
    if frames_n <= 1:
        return {**out, "type": "cut", "confidence": 0.9}

    span = list(range(onset_i, settled_i + 1))
    # elapsed time from the first frame of the move, increasing either way
    # (an exit is read backwards, so raw times run down)
    el = {i: abs(times[i] - times[onset_i]) for i in span}
    t_span = [el[i] for i in span]
    moving = [i for i in span[:-1] if found(i)]
    dx = dy = 0.0
    scale0 = 1.0
    streaked = any(track[i][4] in ("x", "y") for i in moving)
    if moving:
        _, cx, cy, s, _ = track[moving[0]]
        dx, dy, scale0 = (cx - ax) / Wf, (cy - ay) / Hf, s
    scales = [track[i][3] for i in span if found(i) and not track[i][4]]
    overshoot = bool(scales) and max(scales) >= 1.08 and scale0 < 0.95
    visible = [i for i in span if pres[i] >= 0.25] or span
    blurred = streaked or min(rel_sharp[i] for i in visible[:max(2, len(visible) // 2)]) < 0.55
    moved = math.hypot(dx, dy)

    def fill_at(part):
        """Frames after the move starts at which this part is 70% there
        (_first returns a position in span, which starts at the onset)."""
        k = _first(span, lambda i: text.presence(crops[i], part) >= 0.7 * final)
        return k if k is not None else frames_n + 1

    parts = text.halves()
    lr = fill_at(parts["right"]) - fill_at(parts["left"])   # >0: left fills first
    tb = fill_at(parts["bottom"]) - fill_at(parts["top"])   # >0: top fills first

    word_on, within = [], []
    for (c0, c1) in text.words:
        m = np.zeros_like(text.core)
        m[:, c0:c1] = True
        word_on.append(fill_at(m))
        mid = (c0 + c1) // 2
        ml, mr = np.zeros_like(m), np.zeros_like(m)
        ml[:, c0:mid] = True
        mr[:, mid:c1] = True
        within.append(fill_at(mr) - fill_at(ml))

    measures = {"from_dx": round(dx, 3), "from_dy": round(dy, 3), "scale_from": round(scale0, 2),
                "blurred": bool(blurred), "streak": bool(streaked), "fill_lr": lr, "fill_tb": tb,
                "word_fill": word_on, "word_inner_lr": within}
    typ, conf, extra = "unknown", 0.3, {}
    direction = None
    if moved > 0.025:
        direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("bottom" if dy > 0 else "top")

    staggered = (len(word_on) >= 2 and word_on == sorted(word_on)
                 and (word_on[-1] - word_on[0]) >= 3 and all(k <= frames_n for k in word_on))
    typewriter = staggered and np.mean(within) >= 1.0
    if staggered and not typewriter and moved < 0.08:
        step = (word_on[-1] - word_on[0]) / (len(word_on) - 1) / fps
        typ, conf, extra = "stagger", 0.7, {"unit": "word", "stagger_s": round(step, 3)}
    elif moved > 0.025 and abs(scale0 - 1) < 0.2:
        prog = [1 - min(1.0, off(i)[0] / moved) for i in span if found(i)]
        tp = [el[i] for i in span if found(i)]
        fast = dur <= 0.2 or moved / max(dur, 1e-3) > 2.5
        typ, conf = ("whip", 0.75) if (fast and blurred) else ("slide", 0.75)
        extra = {"from": direction, "distance": round(moved, 3), "easing": _easing(prog, tp),
                 "fade": pres[span[0]] < 0.5 and not blurred}
    elif scale0 < 0.85:
        typ, conf, extra = "scale_up", 0.75, {"overshoot": overshoot, "scale_from": round(scale0, 2)}
    elif scale0 > 1.2:
        typ, conf, extra = "scale_down", 0.7, {"scale_from": round(scale0, 2)}
    elif typewriter or max(abs(lr), abs(tb)) >= 2:
        if typewriter or abs(lr) >= abs(tb):
            frm = "left" if lr >= 0 else "right"
        else:
            frm = "top" if tb > 0 else "bottom"
        typ, conf, extra = "wipe", 0.65, {"from": frm, **({"unit": "character"} if typewriter else {})}
    elif blurred:
        typ, conf, extra = "blur_in", 0.6, {}
    elif frames_n >= 3:
        typ, conf, extra = "fade", 0.7, {"easing": _easing([min(1, pres[i] / final) for i in span], t_span)}

    # the measured curve, as a CSS cubic-bezier over the measured duration:
    # position for a slide/whip, scale for a pop/slam, presence for the rest
    curve = None
    if typ in ("slide", "whip"):
        curve = ([el[i] for i in span if found(i)] + [el[span[-1]]],
                 [1 - min(1.0, off(i)[0] / moved) for i in span if found(i)] + [1.0])
    elif typ in ("scale_up", "scale_down") and abs(1 - scale0) > 0.1:
        pts = [(el[i], (track[i][3] - scale0) / (1 - scale0)) for i in span if found(i) and not track[i][4]]
        curve = ([q[0] for q in pts] + [el[span[-1]]], [q[1] for q in pts] + [1.0])
    elif typ in ("fade", "wipe", "stagger", "blur_in"):
        curve = (t_span, [pres[i] / final for i in span])
    # only the words' own move, over a steady background: a move riding the
    # edit is the edit's curve, and busy footage makes the signal wobble
    # enough to fit a plausible-looking wrong curve
    if curve is not None and not busy and kind == "text":
        fit = fit_bezier(*curve, reverse=not entering)
        if fit:
            extra["bezier"], extra["curve_rmse"], name = fit
            if name:
                extra["curve"] = name
    if busy:
        conf *= 0.7
    # per-word arrival, seconds after the move starts: a fact whatever the label
    # — only when the words really do arrive in turn (two frames or more apart)
    if len(word_on) >= 2 and all(k <= frames_n for k in word_on) and max(word_on) - min(word_on) >= 2:
        extra["word_times_s"] = [round(k / fps, 3) for k in word_on]
    if not entering:
        if "from" in extra:
            extra["to"] = extra.pop("from")
        typ = {"scale_up": "scale_down", "scale_down": "scale_up", "blur_in": "blur_out"}.get(typ, typ)
        if "easing" in extra:
            extra["easing"] = {"ease_out": "ease_in", "ease_in": "ease_out"}.get(extra["easing"], extra["easing"])
    return {**out, "type": typ, "confidence": round(conf, 2), **extra, "_measures": measures}


# ----------------------------------------------------------------- strips

STRIP_MOVE_TILES = 6   # the move itself: every frame when it is this short, else this many spread over it


def strip_times(times, t_from, t_to):
    """Which frames to show for a move over [t_from, t_to]: two before it,
    the move at the native rate (every frame of a short move — a 130 ms
    blur-fade is 4 frames at 30 fps — or STRIP_MOVE_TILES spread over a
    longer one), then one just after and two later (+0.25 s, +0.5 s) so a
    move whose measured end is early is still in view. Returns frame indices,
    in time order, without repeats."""
    if not times:
        return []
    step = abs(times[1] - times[0]) if len(times) > 1 else 1 / 30
    span = max(0.0, t_to - t_from)
    k = int(round(span / step))
    move = ([t_from + j * step for j in range(k + 1)] if k + 1 <= STRIP_MOVE_TILES
            else [t_from + span * j / (STRIP_MOVE_TILES - 1) for j in range(STRIP_MOVE_TILES)])
    targets = [t_from - 2 * step, t_from - step] + move + [t_to + step, t_to + 0.25, t_to + 0.5]
    picked = []
    for tt in targets:
        i = min(range(len(times)), key=lambda j: abs(times[j] - tt))
        if i not in picked:
            picked.append(i)
    return sorted(picked, key=lambda i: times[i])


def strip(frames, times, box, t_from, t_to, tile_w=160):
    """The frames strip_times picks, cropped to the text row with a margin
    either side wide enough to show a move from off the text's spot, tiled
    left to right with their timestamps burned in."""
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    x0, y0, x1, y1 = box
    bh, bw = y1 - y0, x1 - x0
    cy0, cy1 = max(0, y0 - bh), min(H, y1 + bh)
    mx = max(int(0.35 * bw), 2 * bh)
    cx0, cx1 = max(0, x0 - mx), min(W, x1 + mx)
    tiles = []
    for i in strip_times(times, t_from, t_to):
        c = frames[i][cy0:cy1, cx0:cx1]
        h = max(1, min(240, int(c.shape[0] * tile_w / c.shape[1])))
        c = cv2.resize(c, (tile_w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(c, (0, 0), (44, 13), (0, 0, 0), -1)
        cv2.putText(c, f"{times[i]:.2f}", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(c)
    return cv2.hconcat(tiles) if tiles else None


# ----------------------------------------------------------------- one event

JUNK_MIN_UNIQUE = 3


def looks_like_text(s):
    """OCR reads zigzags, stripes and logos as 'YYYYYYYY' or 'IIIII'. A string
    of five or more characters drawn from fewer than three distinct letters is
    a graphic, not words — and so is a run of five of the same letter, which is
    the graphic glued onto a real word ('FINALLYYYYYYYYY'): its box is the
    graphic's, so the word's motion cannot be read from it either."""
    letters = [c for c in s.lower() if c.isalpha()]  # prices like 100,000 are words
    if len(letters) >= 5 and len(set(letters)) < JUNK_MIN_UNIQUE:
        return False
    return re.search(r"([^\W\d_])\1{4}", s.lower()) is None


def work_width(video, bbox):
    """Analyse small text at a higher resolution: pick the width at which the
    text box is at least ~26 px tall (360 px minimum, 900 px or native maximum)."""
    cap = cv2.VideoCapture(video)
    try:
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920
    finally:
        cap.release()
    box_h_at_work = (bbox[3] - bbox[1]) * H * WORK_W / W
    want = WORK_W * max(1.0, 26 / max(box_h_at_work, 1e-3))
    return int(min(W, 900, max(WORK_W, want)))


def settle_pool(frames, box, run=3):
    x0, y0, x1, y1 = box
    crops = [_gray(f)[y0:y1, x0:x1] for f in frames]
    d = [float(np.abs(crops[i] - crops[i - 1]).mean()) for i in range(1, len(crops))]
    best, at = None, 1
    for k in range(1, len(crops) - run + 2):
        cost = sum(d[k - 1:k - 1 + run - 1])
        if best is None or cost < best:
            best, at = cost, k
    return frames[at - 1:at - 1 + run]


def analyse_event(video, block, native_fps, duration, cuts=(), want_strip=True):
    t0, t1 = float(block["t0"]), float(block["t1"])
    width = work_width(video, block["bbox"])
    # the entry window stops where the text is last seen, and the exit window
    # starts where it is first seen: text that leaves soon after it lands must
    # not read as "never settled"
    ent_frames, ent_times = read_window(video, t0 - ENTRY_BEFORE, max(t0 + 0.1, min(t0 + ENTRY_AFTER, t1)), native_fps, width)
    if len(ent_frames) < 4:
        return {"error": "no frames"}
    Hf, Wf = ent_frames[0].shape[:2]
    box = _box_px(block["bbox"], Wf, Hf)
    if box[2] - box[0] < 8 or box[3] - box[1] < 5:
        return {"error": "box too small"}
    # the settled look: the steadiest three consecutive frames inside the hold.
    # Kinetic captions hold for half a second and are often still easing in
    # at the OCR's first sighting, so a fixed sample lands mid-animation.
    hold_end = max(t0 + 0.3, min(t1, t0 + 1.0))
    hold, _ = read_window(video, t0, hold_end, native_fps, width)
    pool = settle_pool(hold, _box_px(block["bbox"], hold[0].shape[1], hold[0].shape[0])) if len(hold) >= 3 else ent_frames[-3:]
    ex_frames, ex_times = [], []
    exit_possible = t1 + 0.3 < duration
    if exit_possible:
        ex_frames, ex_times = read_window(video, min(t1 - 0.1, max(t0, t1 - EXIT_BEFORE)), min(duration, t1 + EXIT_AFTER), native_fps, width)
    at_start = t0 < 0.25
    # text-free references: the earliest frames of the entry window and the
    # last of the exit window (the text is not there in either, normally)
    candidates = ([] if at_start else ent_frames[:2]) + (ex_frames[-2:] if len(ex_frames) >= 4 else [])
    # keep only frames the text is really absent from: an end card that stays
    # to the last frame would otherwise count as "text-free" and erase itself
    x0, y0, x1, y1 = box
    ref = np.median(np.stack([_gray(f)[y0:y1, x0:x1] for f in pool]), axis=0)

    def looks_empty(f):
        c = _gray(f)[y0:y1, x0:x1]
        if c.std() < 2 or ref.std() < 2:
            return abs(c.mean() - ref.mean()) > 20
        return float(np.corrcoef(c.ravel(), ref.ravel())[0, 1]) < 0.7

    empties = [f for f in candidates if looks_empty(f)]
    text = Text(pool, empties, box, n_words=len(block["text"].split())) if empties else None
    error = ("on screen for the whole clip" if not empties
             else None if text.ok else "text not separable from the background")

    if error:
        # nothing to measure against, but the strips still show the move
        entry = exit_ = {"type": "unknown", "why": error}
        out = {"t0": round(t0, 2), "t1": round(t1, 2), "error": error}
    else:
        entry = ({"type": "on_screen_at_start"} if at_start
                 else analyse_transition(ent_frames, ent_times, text, native_fps, True, cuts))
        exit_ = {"type": "unknown", "why": "runs to the end of the clip"}
        if exit_possible and len(ex_frames) >= 4:
            exit_ = analyse_transition(ex_frames, ex_times, text, native_fps, False, cuts)
        out = {"t0": round(t0, 2), "t1": round(t1, 2), "entry": entry, "exit": exit_, "words": len(text.words) or 1}
    if want_strip:
        # an unmeasured transition still gets a strip, over the half second
        # the 2 fps OCR says it happened in, for the annotator to judge by eye
        s_in = (strip(ent_frames, ent_times, box, entry.get("t_start", t0 - 0.5), entry.get("t_end", t0 + 0.1))
                if not at_start else None)
        s_out = (strip(ex_frames, ex_times, box, exit_.get("t_start", t1 - 0.1), exit_.get("t_end", t1 + 0.5))
                 if len(ex_frames) >= 4 else None)
        out["_strip"] = (s_in, s_out)
    return out


def analyse_event_job(args):
    """Process-pool entry: one event, cv2 single-threaded so the pool's
    processes do not fight over cores."""
    cv2.setNumThreads(1)
    video, block, native_fps, duration, cuts = args
    try:
        return analyse_event(video, block, native_fps, duration, cuts)
    except Exception as e:  # one bad event must not sink the stage
        return {"error": f"{type(e).__name__}: {e}"[:200]}
