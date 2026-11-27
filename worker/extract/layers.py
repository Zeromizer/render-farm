"""Text layering: is a piece of display type IN FRONT of the subject, BEHIND it,
or clear of it?

Why (the Swee Seng reference jobs, 2026-11-27): a model-range reference set each
car's name in giant letters BEHIND the car, the roof cutting into the word. The
remake put the name above the car in a box. Nothing measured said the name was
behind anything: the text stage gives a block's size and place, and "behind the
car" lived only in a sentence the annotator may or may not have written. The
same goes for the "title behind the presenter" look on footage.

How: for each display-size text block, on a few frames of its hold,

  subject   instance masks from a COCO detector (torchvision Mask R-CNN: car,
            person, truck, motorcycle, ... ). torchvision already ships in this
            venv with EasyOCR, so there is no new dependency.
  strokes   the text's own pixels, found by its colour: the cluster of colours
            inside the block's box that is unlike the surround and uniform
            (type is a flat colour; footage is not).
  verdict   six numbers decide it:
              cover   how much of the visible letters' box the subject is in
              sure    ...counting only what is subject beyond the outline's error
              span    share of the word's columns where the letters stop short of
                      the word's own line AND at the subject's outline
              cut     how far short they stop there (a straight baseline
                      stays straight unless something covers it)
              through how dense the letters are ON the subject, against how
                      dense they are beside it (1 = they carry on over it)
              touch   how much of the subject's outline, inside that box, has
                      letters right up against it
            clear   the subject does not reach into the letters
            front   the letters carry on over the subject
            behind  no letters lie on the subject, they run right up to its
                    outline, and they are CUT there: short of the line the
                    word's whole letters end on, over a tenth of the word
            on      the whole block lies on the subject: a title laid over it,
                    or lettering that is part of it (a number plate, a badge)

The extractor's rule applies: a wrong label is worse than a missing one. A text
colour that cannot be told from the subject's, a subject the detector does not
see, a box that is mostly subject: all "unknown", with the reason. A sheet
image (kf_layers_NN.jpg) shows every verdict with the subject outlined and the
letters tinted, so a person or the annotate pass can overrule it by eye.

numpy + cv2 here; torch/torchvision are imported inside Subjects so the rest of
the extractor does not pay for them.
"""
import os

import cv2
import numpy as np

WORK_W = 640            # analysis width; height follows the aspect
MIN_SIZE = 0.035        # blocks under this share of the frame height are captions, not display type
MAX_BLOCKS = 24
SAMPLES = (0.3, 0.5, 0.75)   # where in a block's settled hold to look
SUBJECT_SCORE = 0.6
SUBJECT_MIN_AREA = 0.012     # of the frame: smaller is a prop, not the subject
MASK_EDGE = 7           # px at WORK_W: kept clear of the subject when reading the text's colour
CORE_IN = 4             # px inside the detector's outline that is surely subject
RIM_OUT = 6             # px outside it that may still be subject
COVER_MIN = 0.02        # under this share of the letters' box, the subject does not reach the word
SURE_MIN = 0.01         # share of the box that is surely subject: more than the outline's own error
SURE_DEEP = 0.05        # ...and this much is the subject standing well inside the word
CUT_SHORT = 0.05        # a column stops short when it ends this share of the height before the word's line
SPAN_MIN = 0.1          # share of the word's columns that stop short AT the subject's outline: cut by it
SPAN_NONE = 0.05        # ...and under this the letters are whole
CUT_MIN = 0.06          # how far short the cut columns stop, as a share of the letters' height
UNDER_ALL = 0.85        # the subject stands in this share of the word's columns: no whole letter to compare with
THROUGH_FRONT = 0.5     # letters at least half as dense on the subject as beside it
THROUGH_BEHIND = 0.15
TOUCH_MIN = 0.15        # share of the outline inside the box with letters right against it
TOUCH_CLOSE = 0.3
MODEL = "maskrcnn_resnet50_fpn_v2"

# The COCO things that stand as the subject of an ad. Furniture, food and
# kitchenware are detected too, but a title "behind a chair" is not the look.
SUBJECT_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "handbag", "suitcase", "bottle", "cup", "cell phone", "laptop", "tv",
    "skateboard", "surfboard", "sports ball", "teddy bear", "clock", "vase",
    # food and drink ads: the dish is the subject
    "bowl", "wine glass", "banana", "apple", "sandwich", "orange", "pizza", "donut", "cake", "hot dog",
}


class Subjects:
    """Subject masks for a frame, from one detector loaded once."""

    def __init__(self):
        import torch
        from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                                  maskrcnn_resnet50_fpn_v2)

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).eval()
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            self._to_cpu(e)
        self.names = weights.meta["categories"]
        self._cache = {}

    def _to_cpu(self, err):
        """The card is full (a video is generating on it): finish on the CPU,
        a second or two a frame, instead of failing the stage."""
        if self.device.type != "cuda" or "out of memory" not in str(err).lower():
            raise err
        print("layers: GPU out of memory, finishing on the CPU")
        self.torch.cuda.empty_cache()
        self.device = self.torch.device("cpu")
        self.model = self.model.to(self.device)

    def _detect(self, x):
        with self.torch.no_grad():
            return self.model([x.to(self.device)])[0]

    def find(self, key, frame_bgr):
        """[(label, score, bool mask)] for the frame, biggest first. `key`
        caches by frame index: several blocks share a moment."""
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        try:
            out = self._detect(x)
        except RuntimeError as e:
            self._to_cpu(e)
            out = self._detect(x)
        h, w = frame_bgr.shape[:2]
        found = []
        for label, score, mask in zip(out["labels"].tolist(), out["scores"].tolist(), out["masks"]):
            name = self.names[label] if 0 <= label < len(self.names) else "?"
            if score < SUBJECT_SCORE or name not in SUBJECT_LABELS:
                continue
            m = (mask[0] > 0.5).cpu().numpy()
            if m.sum() < SUBJECT_MIN_AREA * h * w:
                continue
            found.append((name, round(float(score), 2), m))
        found.sort(key=lambda s: -int(s[2].sum()))
        self._cache[key] = found
        return found


# ----------------------------------------------------------------- frames

def grab(cap, t, native_fps, width=WORK_W):
    """The frame at t, resized to `width`, and its index (seek by INDEX: see
    kinetic.read_window on why POS_MSEC cannot be trusted)."""
    idx = max(0, int(round(t * native_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, idx
    h, w = frame.shape[:2]
    if w != width:
        frame = cv2.resize(frame, (width, max(2, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return frame, idx


def _box_px(bbox, w, h, pad=0.0):
    x0, y0, x1, y1 = bbox
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    return (max(0, int((x0 - px) * w)), max(0, int((y0 - py) * h)),
            min(w, int(np.ceil((x1 + px) * w))), min(h, int(np.ceil((y1 + py) * h))))


# ----------------------------------------------------------------- strokes

def text_strokes(frame_bgr, bbox, subject):
    """The text's pixels as a bool mask over the frame, and why not when it
    cannot be found. Looks for the colour inside the block's box (off the
    subject) that is unlike the surround and uniform, then takes every pixel of
    that colour in and just around the box, the subject included: whether the
    letters carry on over the subject is the question."""
    h, w = frame_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    x0, y0, x1, y1 = _box_px(bbox, w, h, pad=0.06)
    if x1 - x0 < 12 or y1 - y0 < 6:
        return None, "box too small"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_EDGE + 1, 2 * MASK_EDGE + 1))
    near = cv2.dilate(subject.astype(np.uint8), kernel).astype(bool)

    box = np.zeros((h, w), bool)
    box[y0:y1, x0:x1] = True
    free = box & ~near                       # the box, off the subject
    if free.sum() < 0.15 * box.sum():
        return None, "box is mostly subject"

    # the surround: a band just outside the box, off the subject
    bx, by = max(4, (x1 - x0) // 8), max(4, (y1 - y0) // 3)
    band = np.zeros((h, w), bool)
    band[max(0, y0 - by):min(h, y1 + by), max(0, x0 - bx):min(w, x1 + bx)] = True
    band &= ~box & ~near
    if band.sum() < 40:
        return None, "no surround to compare with"
    surround = np.median(lab[band], axis=0)

    pts = lab[free]
    if len(pts) > 6000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 6000, replace=False)]
    k = 3 if len(pts) >= 300 else 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    best, best_score = None, 0.0
    for c in range(k):
        mine = pts[labels == c]
        share = len(mine) / len(pts)
        if share < 0.04 or share > 0.8:
            continue
        away = float(np.linalg.norm(centres[c] - surround))
        spread = float(np.linalg.norm(mine.std(axis=0)))
        score = away / (1.0 + spread / 6.0)   # unlike the surround, and one flat colour
        if away >= 14 and score > best_score:
            best, best_score = c, score
    if best is None:
        return None, "no text colour stands out from the surround"
    colour = centres[best]
    others = [float(np.linalg.norm(colour - centres[c])) for c in range(k) if c != best]
    tol = float(np.clip(0.45 * min(others + [float(np.linalg.norm(colour - surround))]), 9, 30))

    look = np.zeros((h, w), bool)            # the box and a little around it
    gy = max(3, (y1 - y0) // 5)
    look[max(0, y0 - gy):min(h, y1 + gy), x0:x1] = True
    strokes = (np.linalg.norm(lab - colour, axis=2) < tol) & look
    strokes = cv2.morphologyEx(strokes.astype(np.uint8), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    off = strokes & ~near
    if off.sum() < 0.02 * box.sum():
        return None, "too little of the text colour to be letters"
    if off.sum() > 0.7 * free.sum():
        return None, "the text colour fills the box: not letters"

    # a subject of the text's own colour would read as letters lying on it
    inner = cv2.erode(subject.astype(np.uint8), kernel).astype(bool) & ~look
    if inner.sum() > 200:
        same = float((np.linalg.norm(lab[inner] - colour, axis=1) < tol).mean())
        if same > 0.25:
            return None, "the subject is the text's own colour"
    return strokes, None


# ----------------------------------------------------------------- verdict

def _disc(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _profile(strokes, subject, box):
    """Are the letters CUT by the subject, or do they only stand next to it?

    Reads the letters' edge on the subject's side column by column. A column
    is cut when its letters stop SHORT of the line the word's whole letters end
    on AND stop at the subject's outline. Either alone is ordinary: a T's arm
    or an O's curve ends short of the baseline by its own shape, and a head
    that rises into the gap between two letters has letters at its outline
    that are whole. Returns (span, cut, under), or None when there are too few
    columns to say:
      span   share of the word's columns that are cut
      cut    how far short those columns stop, as a share of the letters' height
      under  share of the word's columns the subject stands in
    """
    x0, x1, y0, y1 = box
    s = strokes[y0:y1, x0:x1]
    sub = subject[y0:y1, x0:x1]
    has = s.any(axis=0)
    cols = np.nonzero(has)[0]
    both = np.nonzero(has & sub.any(axis=0))[0]
    if len(cols) < 8 or len(both) < 3:
        return None
    rows = s.shape[0]
    top = s.argmax(axis=0).astype(np.float32)
    bot = (rows - 1 - s[::-1].argmax(axis=0)).astype(np.float32)
    sub_top = sub.argmax(axis=0).astype(np.float32)
    sub_bot = (rows - 1 - sub[::-1].argmax(axis=0)).astype(np.float32)

    # which edge the subject comes at: from below (a roof under a name) or from above
    if float(np.median(sub_top[both])) > 0.5 * (rows - 1):
        edge, outline = bot, sub_top
        short = float(np.percentile(bot[cols], 90)) - bot
    else:
        edge, outline = top, sub_bot
        short = top - float(np.percentile(top[cols], 10))
    at_outline = np.abs(edge - outline) <= RIM_OUT + 2
    cut_cols = both[at_outline[both] & (short[both] >= max(3.0, CUT_SHORT * rows))]
    span = len(cut_cols) / len(cols)
    cut = float(np.median(short[cut_cols])) / rows if len(cut_cols) else 0.0
    return span, cut, len(both) / len(cols)


def judge(strokes, subject):
    """(layer, numbers) for one frame: the letters against the subject."""
    sub = subject.astype(np.uint8)
    core = cv2.erode(sub, _disc(CORE_IN)).astype(bool)     # surely subject
    reach = cv2.dilate(sub, _disc(RIM_OUT)).astype(bool)   # subject, give or take its outline
    rim = reach & ~core                                    # the outline itself

    ys, xs = np.nonzero(strokes)
    if len(xs) == 0:
        return "unknown", {"why": "no letters found"}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    box = np.zeros_like(strokes)
    box[y0:y1, x0:x1] = True
    area = float(box.sum())
    cover = float((box & subject).sum()) / area    # the subject reaches into the letters' box
    sure = float((box & core).sum()) / area        # ...by more than the outline's own error

    beside = float((strokes & ~reach).sum()) / max(1.0, float((box & ~reach).sum()))
    core_px = float((box & core).sum())
    under = float((strokes & core & box).sum()) / core_px if core_px >= 30 else 0.0
    through = under / beside if beside > 0 else 0.0

    # how much of the subject's outline, where it crosses the letters' box, has letters right against it
    rim_in = rim & box
    near = cv2.dilate(strokes.astype(np.uint8), _disc(2)).astype(bool)
    touch = float((rim_in & near).sum()) / float(rim_in.sum()) if rim_in.sum() >= 20 else 0.0

    nums = {"cover": round(cover, 3), "sure": round(sure, 3), "through": round(through, 3), "touch": round(touch, 3)}
    if cover < COVER_MIN and touch < TOUCH_MIN:
        return "clear", nums
    if sure >= SURE_MIN and through >= THROUGH_FRONT:
        return "front", nums
    if through > THROUGH_BEHIND:
        return "unknown", {**nums, "why": "letters partly on the subject"}
    if touch < TOUCH_MIN:
        return "clear", nums      # the subject is in the word's box, the letters keep away from it

    # No letters on the subject, and letters right against its outline. Cut by
    # it, or only standing next to it?
    if sure >= SURE_DEEP:
        return "behind", nums     # it is well inside the word: nothing else explains the gap
    prof = _profile(strokes, subject, (x0, x1, y0, y1))
    if prof is None:
        return "unknown", {**nums, "why": "too little of the word meets the subject to tell cut letters from whole ones"}
    span, cut, under = prof
    nums.update(span=round(span, 3), cut=round(cut, 3))
    if span >= SPAN_MIN and cut >= CUT_MIN and sure >= SURE_MIN:
        return "behind", nums
    if span < SPAN_NONE and under < UNDER_ALL:
        return "clear", nums      # the letters are whole where they meet it: it stands in their gaps or under them
    if span < SPAN_NONE:
        return "unknown", {**nums, "why": "the subject is as wide as the word: no whole letter to compare the others with"}
    return "unknown", {**nums, "why": "a few letters stop at the subject's outline: cut by it, or only shaped that way"}


def _dump(block, t, strokes, mask, label, layer):
    """Bench only (LAYERS_DUMP=<dir>): keep what judge() was given, so a rule
    change can be replayed without the detector (layers_bench/replay.py)."""
    out = os.environ.get("LAYERS_DUMP")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    name = f"{block['t0']:07.2f}_{t:07.2f}_{int(round(block['bbox'][0] * 1000)):04d}_{int(round(block['bbox'][1] * 1000)):04d}"
    np.savez_compressed(os.path.join(out, name + ".npz"), shape=np.array(strokes.shape),
                        strokes=np.packbits(strokes), mask=np.packbits(mask.astype(bool)),
                        meta=np.array([str(block.get("text", "")), label, layer, f"{t:.2f}", f"{block['t0']:.2f}"]))


def pick_blocks(blocks, duration, looks_like_text):
    """Display-size text worth judging, spread over the clip: [index]."""
    ids = [i for i, b in enumerate(blocks)
           if b.get("size_ratio", 0) >= MIN_SIZE
           and b["t1"] - b["t0"] < max(6.0, 0.8 * duration)
           and b["t1"] - b["t0"] >= 0.3
           and looks_like_text(b.get("text", ""))]
    ids.sort(key=lambda i: blocks[i]["t0"])
    if len(ids) > MAX_BLOCKS:
        keep = np.linspace(0, len(ids) - 1, MAX_BLOCKS).round().astype(int)
        ids = [ids[k] for k in sorted(set(keep.tolist()))]
    return ids


def settle(layers):
    """One verdict from the frames read. A layer is a fact about the whole
    hold: it takes a majority of the frames, and "behind" and "front" both
    seen is a contradiction, not a vote."""
    votes = {}
    for layer in layers:
        votes[layer] = votes.get(layer, 0) + 1
    if votes.get("behind") and votes.get("front"):
        return "unknown"
    top = max(votes, key=votes.get)
    return top if votes[top] * 2 > len(layers) else "unknown"


def analyse_block(cap, block, native_fps, subjects):
    """One block: a vote over a few frames of its hold. Returns the verdict
    dict and, for the sheet, the frame, subject mask and strokes it was read on."""
    t_in = max(block["t0"], block.get("t_full", block["t0"]))
    t_out = block["t1"]
    if t_out - t_in < 0.2:      # completes as it leaves: judge the whole sighting
        t_in = block["t0"]
    votes, seen, shown = {}, [], None
    for frac in SAMPLES:
        t = t_in + frac * (t_out - t_in)
        frame, idx = grab(cap, t, native_fps)
        if frame is None:
            continue
        found = subjects.find(idx, frame)
        h, w = frame.shape[:2]
        if not found:
            votes["clear"] = votes.get("clear", 0) + 1
            seen.append({"t": round(t, 2), "layer": "clear", "why": "no subject in frame"})
            shown = shown or (frame, np.zeros((h, w), bool), None, t)
            continue
        # the subject the text has to do with: the one its box overlaps most
        x0, y0, x1, y1 = _box_px(block["bbox"], w, h, pad=0.06)
        label, _score, mask = max(found, key=lambda s: int(s[2][y0:y1, x0:x1].sum()))
        if mask[y0:y1, x0:x1].sum() == 0:
            votes["clear"] = votes.get("clear", 0) + 1
            seen.append({"t": round(t, 2), "layer": "clear", "subject": label, "cover": 0.0})
            shown = shown or (frame, mask, None, t)
            continue
        strokes, why = text_strokes(frame, block["bbox"], mask)
        if strokes is None and why == "box is mostly subject":
            # The text pass READ this block, and its box lies on the subject:
            # letters behind a subject that covers them could not have been read.
            votes["on"] = votes.get("on", 0) + 1
            seen.append({"t": round(t, 2), "layer": "on", "subject": label, "cover": 1.0,
                         "why": "lies wholly on the subject"})
            shown = shown or (frame, mask, None, t)
            continue
        if strokes is None:
            votes["unknown"] = votes.get("unknown", 0) + 1
            seen.append({"t": round(t, 2), "layer": "unknown", "subject": label, "why": why})
            shown = shown or (frame, mask, None, t)
            continue
        layer, nums = judge(strokes, mask)
        _dump(block, t, strokes, mask, label, layer)
        votes[layer] = votes.get(layer, 0) + 1
        seen.append({"t": round(t, 2), "layer": layer, "subject": label, **nums})
        if shown is None or shown[2] is None or layer in ("behind", "front"):
            shown = (frame, mask, strokes, t)
    if not seen:
        return {"layer": "unknown", "why": "no frame could be read"}, None

    layer = settle([s["layer"] for s in seen])
    pick = next((s for s in seen if s["layer"] == layer), seen[0])
    out = {"layer": layer, "t": pick["t"], "frames": len(seen),
           "agree": round(votes.get(layer, 0) / len(seen), 2)}
    for key in ("subject", "cover", "sure", "through", "touch", "span", "cut", "why"):
        if key in pick:
            out[key] = pick[key]
    return out, shown


# ----------------------------------------------------------------- sheet

_TINT = {"behind": (60, 60, 235), "front": (60, 200, 60), "on": (200, 160, 60), "clear": (200, 200, 200), "unknown": (0, 200, 235)}


def tile(shown, block, verdict, n, tile_w=300):
    """One sheet tile: the frame, the subject outlined, the letters tinted, the verdict."""
    frame, mask, strokes, t = shown
    img = frame.copy()
    if strokes is not None:
        img[strokes] = (0.45 * img[strokes] + 0.55 * np.array((255, 220, 0))).astype(np.uint8)
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (40, 40, 240), 2)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _box_px(block["bbox"], w, h)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
    img = cv2.resize(img, (tile_w, round(h * tile_w / w)), interpolation=cv2.INTER_AREA)
    bar = np.zeros((34, tile_w, 3), np.uint8)
    layer = verdict["layer"]
    cv2.putText(bar, f"#{n} {t:.2f}s {layer.upper()}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                _TINT.get(layer, (255, 255, 255)), 1, cv2.LINE_AA)
    detail = " ".join(f"{k}={verdict[k]}" for k in ("subject", "sure", "touch", "span", "cut") if k in verdict)
    cv2.putText(bar, (detail or verdict.get("why", ""))[:46], (5, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def sheet(tiles, cols=4):
    """Tiles in a grid (they share a size)."""
    rows = []
    for k in range(0, len(tiles), cols):
        row = tiles[k:k + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)
