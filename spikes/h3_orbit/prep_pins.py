"""Build 1344x768 pin canvases for the Noah orbit: orig (4 photos), toy (12 Toyota TCV frames), mix (orig at
0/90/180/270 + Toyota at the 30-degree steps between, scaled to the photos). Wheels share one ground line."""
import os
import sys

import cv2
import numpy as np

ROOT = sys.argv[1]           # Downloads/h3-orbit-spike
OUT = sys.argv[2]
W, H = 1344, 768
BG = 238
os.makedirs(OUT, exist_ok=True)
ORIG = {0: "front", 90: "left", 180: "rear", 270: "right"}


def orig_crop(v):
    im = cv2.imread(os.path.join(ROOT, "orig", f"noah_{v}.png"))
    d = np.abs(im.astype(np.int16) - BG).max(axis=2) > 28
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(d)
    return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def toy_crop(i):
    im = cv2.imread(os.path.join(ROOT, "noah360", f"frame_{i:02d}.png"), cv2.IMREAD_UNCHANGED)
    a = im[:, :, 3:4].astype(np.float32) / 255
    flat = (im[:, :, :3] * a + BG * (1 - a)).astype(np.uint8)
    ys, xs = np.nonzero(im[:, :, 3] > 200)
    return flat[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def place(crop, s, ground):
    cw, ch = round(crop.shape[1] * s), round(crop.shape[0] * s)
    r = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    can = np.full((H, W, 3), BG, np.uint8)
    x, y = (W - cw) // 2, ground - ch
    can[y:y + ch, x:x + cw] = r
    return can


orig = {deg: orig_crop(v) for deg, v in ORIG.items()}
toy = {i * 30: toy_crop(i) for i in range(12)}
s_o = min(min(0.86 * W / c.shape[1], 0.70 * H / c.shape[0]) for c in orig.values())
s_t = min(min(0.86 * W / c.shape[1], 0.70 * H / c.shape[0]) for c in toy.values())
# Toyota -> photo scale from the four shared views (widths of the sides, heights of front/rear)
k = np.median([orig[90].shape[1] / toy[90].shape[1], orig[270].shape[1] / toy[270].shape[1],
               orig[0].shape[0] / toy[0].shape[0], orig[180].shape[0] / toy[180].shape[0]])
side_h_o = orig[90].shape[0] * s_o
ground_o = round((H + side_h_o) / 2)
ground_t = round((H + toy[90].shape[0] * s_t) / 2)
print(f"s_orig {s_o:.3f} s_toy {s_t:.3f} toy->orig k {k:.3f}")
for deg in range(0, 360, 30):
    if deg in orig:
        cv2.imwrite(os.path.join(OUT, f"orig_{deg:03d}.png"), place(orig[deg], s_o, ground_o))
        cv2.imwrite(os.path.join(OUT, f"mix_{deg:03d}.png"), place(orig[deg], s_o, ground_o))
    else:
        cv2.imwrite(os.path.join(OUT, f"mix_{deg:03d}.png"), place(toy[deg], s_o * k, ground_o))
    cv2.imwrite(os.path.join(OUT, f"toy_{deg:03d}.png"), place(toy[deg], s_t, ground_t))
tiles = [cv2.resize(cv2.imread(os.path.join(OUT, f"mix_{d:03d}.png")), (336, 192)) for d in range(0, 360, 30)]
cv2.imwrite(os.path.join(ROOT, "pins_mix_sheet.jpg"), np.vstack([np.hstack(tiles[r:r + 6]) for r in (0, 6)]))


# "toyn": Toyota frames placed with ONE scale and ONE offset for all 12 (keeps the render's own camera:
# fixed roof line, true turning axis). Per-view bbox placement made the car bob and drift.
raw = []
for i in range(12):
    im = cv2.imread(os.path.join(ROOT, "noah360", f"frame_{i:02d}.png"), cv2.IMREAD_UNCHANGED)
    a = im[:, :, 3:4].astype(np.float32) / 255
    raw.append((im[:, :, :3] * a + BG * (1 - a)).astype(np.uint8))
    ys, xs = np.nonzero(im[:, :, 3] > 200)
    box = (xs.min(), ys.min(), xs.max(), ys.max()) if i == 0 else (min(box[0], xs.min()), min(box[1], ys.min()),
                                                                 max(box[2], xs.max()), max(box[3], ys.max()))
ux0, uy0, ux1, uy1 = box
s = min(0.86 * W / (ux1 - ux0 + 1), 0.78 * H / (uy1 - uy0 + 1))
ox = W / 2 - s * (ux0 + ux1) / 2
oy = H / 2 - s * (uy0 + uy1) / 2
M = np.float32([[s, 0, ox], [0, s, oy]])
print(f"toyn: union {box}, scale {s:.3f}, offset ({ox:.1f}, {oy:.1f})")
for i, im in enumerate(raw):
    out = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(BG, BG, BG))
    cv2.imwrite(os.path.join(OUT, f"toyn_{i * 30:03d}.png"), out)
