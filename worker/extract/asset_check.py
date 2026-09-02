"""asset_check: the automated pre-review checks on one candidate still
(render-asset-gate-spec.md §6). Flags only, never a verdict.

  python asset_check.py <image> <out.json> --slot plate [--palette <image>] [--threshold 12]

Writes {palette_distance, lighting_flags, text_detected, measurements}.

- palette_distance: 8 dominant colours of the candidate via k-means in Lab,
  each matched to its nearest of the palette frame's 8, weighted mean ΔE
  (CIE76). Null without a palette frame.
- lighting_flags (product_sheet / character_sheet_* only): specular_highlights,
  hard_shadow, not_neutral_bg — simple heuristics on a subject mask that is
  "not the border colour, largest blob". Wrong sometimes; cheap always.
- text_detected (plate / prop_* only): EasyOCR boxes with conf >= 0.5.
"""
import argparse
import json

import cv2
import numpy as np

MAX_W = 512
K = 8


def _load(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"could not read {path}")
    h, w = im.shape[:2]
    if w > MAX_W:
        im = cv2.resize(im, (MAX_W, round(h * MAX_W / w)), interpolation=cv2.INTER_AREA)
    return im


def _lab(im):
    """OpenCV Lab (0-255 scaled) -> real CIE L*a*b*."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def _clusters(lab, k=K, seed=0):
    from sklearn.cluster import KMeans

    px = lab.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(px), size=min(12000, len(px)), replace=False)
    sample = px[idx]
    km = KMeans(n_clusters=min(k, len(sample)), n_init=4, random_state=seed).fit(sample)
    counts = np.bincount(km.labels_, minlength=km.n_clusters).astype(float)
    return km.cluster_centers_, counts / counts.sum()


def palette_distance(cand_lab, pal_lab):
    c_centres, c_w = _clusters(cand_lab, seed=1)
    p_centres, _ = _clusters(pal_lab, seed=2)
    # ΔE76 from each candidate colour to its nearest palette colour.
    d = np.linalg.norm(c_centres[:, None, :] - p_centres[None, :, :], axis=2).min(axis=1)
    return float((d * c_w).sum())


def _subject_mask(im, lab):
    """Background = whatever the border band is coloured; subject = the largest
    blob of everything else."""
    h, w = im.shape[:2]
    band = max(4, min(h, w) // 25)
    border = np.concatenate([
        lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
        lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    fg = (dist > 12).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), bool), bg, border
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest, bg, border


def lighting_flags(im, lab):
    subject, _bg, border = _subject_mask(im, lab)
    out = {}
    n_subj = int(subject.sum())
    if n_subj < 100:
        out["subject_found"] = False
        return out
    out["subject_found"] = True

    L = lab[..., 0]
    spec = float((L[subject] > 95).mean())
    out["specular_ratio"] = round(spec, 4)
    out["specular_highlights"] = spec > 0.005

    # Hard shadow: strong luminance edges in a ring just outside the subject.
    ring = cv2.dilate(subject.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~subject
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ring_mag = mag[ring] if ring.any() else np.zeros(1)
    hard = float((ring_mag > 60).mean()) if ring.any() else 0.0
    out["shadow_edge_ratio"] = round(hard, 4)
    out["hard_shadow"] = hard > 0.15

    # Neutral background: the border band's saturation, in HSV terms.
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, w = im.shape[:2]
    band = max(4, min(h, w) // 25)
    s = np.concatenate([
        hsv[:band, :, 1].ravel(), hsv[-band:, :, 1].ravel(),
        hsv[:, :band, 1].ravel(), hsv[:, -band:, 1].ravel(),
    ]).astype(np.float32) / 255.0
    sat = float(np.median(s))
    out["bg_saturation"] = round(sat, 4)
    out["not_neutral_bg"] = sat > 0.08
    out["bg_lab"] = [round(float(v), 1) for v in np.median(border, axis=0)]
    return out


def text_detected(path):
    import easyocr

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    hits = []
    try:
        dets = reader.readtext(path, paragraph=False)
    except Exception as e:  # noqa: BLE001
        return {"hits": [], "error": str(e)[:200]}
    for quad, text, conf in dets:
        if conf < 0.5 or not text.strip():
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        hits.append({"text": text.strip(), "conf": round(float(conf), 3),
                     "bbox": [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]})
    return {"hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--slot", required=True)
    ap.add_argument("--palette", default=None)
    ap.add_argument("--threshold", type=float, default=12.0)
    args = ap.parse_args()

    im = _load(args.image)
    lab = _lab(im)
    result = {"palette_distance": None, "lighting_flags": None, "text_detected": None,
              "threshold": args.threshold}

    if args.palette:
        pal = _lab(_load(args.palette))
        result["palette_distance"] = round(palette_distance(lab, pal), 2)
        print(f"palette_distance {result['palette_distance']}")
    print("PROGRESS 40")

    slot = args.slot
    if slot == "product_sheet" or slot.startswith("character_sheet"):
        result["lighting_flags"] = lighting_flags(im, lab)
        print(f"lighting_flags {json.dumps(result['lighting_flags'])}")
    print("PROGRESS 60")

    if slot == "plate" or slot.startswith("prop_"):
        result["text_detected"] = text_detected(args.image)
        print(f"text_detected {len(result['text_detected']['hits'])} hit(s)")
    print("PROGRESS 95")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
