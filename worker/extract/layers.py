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
  verdict   three numbers decide it:
              cover   how much of the visible letters' box the subject covers
              through how dense the letters are UNDER the subject, against
                      how dense they are beside it (1 = they carry on over it)
              touch   how much of the word's width has letters running right
                      up to the subject's outline
            clear   the subject does not reach into the letters
            front   the letters carry on over the subject
            behind  the subject covers part of the word, the letters stop at
                    its outline, and there are none on it

The extractor's rule applies: a wrong label is worse than a missing one. A text
colour that cannot be told from the subject's, a subject the detector does not
see, a box that is mostly subject: all "unknown", with the reason. A sheet
image (kf_layers_NN.jpg) shows every verdict with the subject outlined and the
letters tinted, so a person or the annotate pass can overrule it by eye.

numpy + cv2 here; torch/torchvision are imported inside Subjects so the rest of
the extractor does not pay for them.
"""
import cv2
import numpy as np

WORK_W = 640            # analysis width; height follows the aspect
MIN_SIZE = 0.035        # blocks under this share of the frame height are captions, not display type
MAX_BLOCKS = 24
SAMPLES = (0.3, 0.5, 0.75)   # where in a block's settled hold to look
SUBJECT_SCORE = 0.6
SUBJECT_MIN_AREA = 0.012     # of the frame: smaller is a prop, not the subject
MASK_EDGE = 7           # px at WORK_W: how far the detector's outline can be off
COVER_MIN = 0.05        # the subject must cover this much of the letters' box to matter
THROUGH_FRONT = 0.5     # letters at least half as dense on the subject as beside it
THROUGH_BEHIND = 0.15
TOUCH_MIN = 0.15
MODEL = "maskrcnn_resnet50_fpn_v2"

# The COCO things that stand as the subject of an ad. Furniture, food and
# kitchenware are detected too, but a title "behind a chair" is not the look.
SUBJECT_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "handbag", "suitcase", "bottle", "cup", "cell phone", "laptop", "tv",
    "skateboard", "surfboard", "sports ball", "teddy bear", "clock", "vase",
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
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
        self.names = weights.meta["categories"]
        self._cache = {}

    def find(self, key, frame_bgr):
        """[(label, score, bool mask)] for the frame, biggest first. `key`
        caches by frame index: several blocks share a moment."""
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model([x])[0]
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

def judge(strokes, subject):
    """(layer, numbers) for one frame: the letters against the subject."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_EDGE + 1, 2 * MASK_EDGE + 1))
    sub = subject.astype(np.uint8)
    core = cv2.erode(sub, kernel).astype(bool)      # surely subject
    reach = cv2.dilate(sub, kernel).astype(bool)    # subject, give or take its outline
    rim = reach & ~core                             # the outline itself

    off = strokes & ~reach
    ys, xs = np.nonzero(strokes)
    if len(xs) == 0:
        return "unknown", {"why": "no letters found"}
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    box = np.zeros_like(strokes)
    box[y0:y1, x0:x1] = True
    area = float(box.sum())
    cover = float((box & core).sum()) / area

    beside = float(off.sum()) / max(1.0, float((box & ~reach).sum()))
    under = float((strokes & core & box).sum()) / max(1.0, float((box & core).sum()))
    through = under / beside if beside > 0 else 0.0

    cols = np.nonzero(strokes[:, x0:x1].any(axis=0))[0]
    meets = np.nonzero((strokes & rim)[:, x0:x1].any(axis=0))[0]
    touch = len(meets) / max(1, len(cols))

    nums = {"cover": round(cover, 3), "through": round(through, 3), "touch": round(touch, 3)}
    if cover < COVER_MIN:
        return "clear", nums
    if through >= THROUGH_FRONT:
        return "front", nums
    if through <= THROUGH_BEHIND and touch >= TOUCH_MIN:
        return "behind", nums
    if through <= THROUGH_BEHIND:
        # the subject reaches into the word's box but the letters do not meet it
        return "clear", nums
    return "unknown", {**nums, "why": "letters partly on the subject"}


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
            votes["front"] = votes.get("front", 0) + 1
            seen.append({"t": round(t, 2), "layer": "front", "subject": label, "cover": 1.0,
                         "why": "read on top of the subject"})
            shown = shown or (frame, mask, None, t)
            continue
        if strokes is None:
            votes["unknown"] = votes.get("unknown", 0) + 1
            seen.append({"t": round(t, 2), "layer": "unknown", "subject": label, "why": why})
            shown = shown or (frame, mask, None, t)
            continue
        layer, nums = judge(strokes, mask)
        votes[layer] = votes.get(layer, 0) + 1
        seen.append({"t": round(t, 2), "layer": layer, "subject": label, **nums})
        if shown is None or shown[2] is None or layer in ("behind", "front"):
            shown = (frame, mask, strokes, t)
    if not seen:
        return {"layer": "unknown", "why": "no frame could be read"}, None

    # A layer is a fact about the whole hold: it takes a majority of the frames
    # read, and "behind" and "front" both seen is a contradiction, not a vote.
    decided = {k: v for k, v in votes.items() if k != "unknown"}
    if decided.get("behind") and decided.get("front"):
        layer = "unknown"
    elif decided:
        layer = max(decided, key=decided.get)
        if decided[layer] * 2 <= len(seen) and layer != "clear":
            layer = "unknown"
    else:
        layer = "unknown"
    pick = next((s for s in seen if s["layer"] == layer), seen[0])
    out = {"layer": layer, "t": pick["t"], "frames": len(seen),
           "agree": round(votes.get(layer, 0) / len(seen), 2)}
    for key in ("subject", "cover", "through", "touch", "why"):
        if key in pick:
            out[key] = pick[key]
    return out, shown


# ----------------------------------------------------------------- sheet

_TINT = {"behind": (60, 60, 235), "front": (60, 200, 60), "clear": (200, 200, 200), "unknown": (0, 200, 235)}


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
    detail = " ".join(f"{k}={verdict[k]}" for k in ("subject", "cover", "through", "touch") if k in verdict)
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
