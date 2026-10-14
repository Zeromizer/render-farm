"""Synthetic kinetic-type clips with KNOWN entrances and exits — the ground truth
the kinetic stage is tuned against. Each clip: 30 fps, 540x960, 3.4 s, one text
event entering at ENTRY_T and exiting at EXIT_T over a moving textured plate.

Writes <name>.mp4 + truth.json (labels + the coarse text block a 2 fps OCR pass
would have produced, which is the kinetic stage's input)."""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 540, 960, 30, 3.4
ENTRY_T, EXIT_T = 1.0, 2.4
FONT = "C:/Windows/Fonts/arialbd.ttf"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/synth"
FFMPEG = sys.argv[1] if len(sys.argv) > 1 else "ffmpeg"


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    return p ** 3


def back_out(p, s=2.2):
    p -= 1
    return p * p * ((s + 1) * p + s) + 1


def clamp(p):
    return max(0.0, min(1.0, p))


def plate(t):
    """Moving, busy background: drifting gradient + panning noise texture +
    a big soft 'car' blob — enough texture to fool a naive frame diff."""
    rng = np.random.default_rng(7)
    base = rng.integers(40, 200, (H // 8, W // 4 + 40, 3), dtype=np.uint8)
    tex = cv2.resize(base, (W + 160, H), interpolation=cv2.INTER_CUBIC)
    off = int(t * 40) % 160
    img = tex[:, off:off + W].astype(np.float32) * 0.55
    yy = np.linspace(0, 1, H)[:, None]
    img[..., 0] += 60 * yy
    img[..., 2] += 50 * (1 - yy)
    cx = int(W * (0.3 + 0.2 * math.sin(t)))
    cv2.circle(img, (cx, int(H * 0.62)), 150, (90, 90, 160), -1)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 2), 0, 255).astype(np.uint8)


def text_layer(words, size=64):
    font = ImageFont.truetype(FONT, size)
    text = " ".join(words)
    box = font.getbbox(text)
    tw, th = box[2] - box[0] + 20, box[3] - box[1] + 24
    layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((10 - box[0], 12 - box[1]), text, font=font, fill=(255, 255, 255, 255),
           stroke_width=3, stroke_fill=(20, 20, 20, 255))
    # word x-extents for stagger/typewriter reveals
    spans, x = [], 10
    for i, w in enumerate(words):
        seg = w + (" " if i < len(words) - 1 else "")
        wlen = font.getlength(w)
        spans.append((x, x + wlen))
        x += font.getlength(seg)
    return layer, spans


def composite(frame, layer, cx, cy, scale=1.0, alpha=1.0, blur=0.0, motion_blur=0.0, clip=None):
    """Place `layer` centred at (cx,cy) with transforms. clip=(x0,y0,x1,y1) in
    layer-relative 0-1 coords keeps only that part (wipes/typewriter)."""
    L = layer
    if clip is not None:
        x0, y0, x1, y1 = clip
        a = np.array(L)
        m = np.zeros(a.shape[:2], np.uint8)
        h, w = m.shape
        m[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = 1
        a[..., 3] = a[..., 3] * m
        L = Image.fromarray(a)
    if scale != 1.0:
        L = L.resize((max(1, int(L.width * scale)), max(1, int(L.height * scale))), Image.BICUBIC)
    if blur > 0.2:
        pad = int(blur * 3)
        big = Image.new("RGBA", (L.width + 2 * pad, L.height + 2 * pad), (0, 0, 0, 0))
        big.paste(L, (pad, pad))
        L = big.filter(ImageFilter.GaussianBlur(blur))
    if motion_blur > 1:
        k = int(motion_blur)
        a = np.array(L).astype(np.float32)
        pad = k
        a = np.pad(a, ((0, 0), (pad, pad), (0, 0)))
        kern = np.ones((1, k), np.float32) / k
        a = cv2.filter2D(a, -1, kern)
        L = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    a = np.array(L).astype(np.float32)
    a[..., 3] *= alpha
    x0 = int(cx - L.width / 2)
    y0 = int(cy - L.height / 2)
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + L.width), min(H, y0 + L.height)
    if fx1 <= fx0 or fy1 <= fy0:
        return frame
    sub = a[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    al = sub[..., 3:4] / 255.0
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    rgb = sub[..., :3][..., ::-1]  # RGB -> BGR
    frame[fy0:fy1, fx0:fx1] = (region * (1 - al) + rgb * al).astype(np.uint8)
    return frame


CX, CY = W / 2, H * 0.40

# name -> (words, entry_fn(p, t_local)->kwargs, entry_dur, exit_fn, exit_dur, truth)
def cases():
    def stay(**k):
        return {**k}

    return {
        "slide_left": (["SLIDE", "IN"], lambda p: dict(cx=CX + W * 0.45 * (1 - ease_out_cubic(p))), 0.35,
                       lambda p: dict(alpha=1 - p), 0.25,
                       {"entry": "slide", "entry_from": "right", "exit": "fade"}),
        "whip_right": (["WHIP"], lambda p: dict(cx=CX - W * 0.7 * (1 - ease_out_cubic(p)), motion_blur=40 * (1 - p)), 0.12,
                       lambda p: dict(cx=CX - W * 0.7 * ease_in_cubic(p), motion_blur=40 * p), 0.12,
                       {"entry": "whip", "entry_from": "left", "exit": "whip", "exit_to": "left"}),
        "pop": (["POP"], lambda p: dict(scale=max(0.05, 0.2 + 0.8 * back_out(p))), 0.30,
                lambda p: dict(scale=max(0.05, 1 - 0.95 * ease_in_cubic(p))), 0.20,
                {"entry": "scale_up", "overshoot": True, "exit": "scale_down"}),
        "slam": (["SLAM"], lambda p: dict(scale=1 + 1.6 * (1 - ease_out_cubic(p)), alpha=clamp(p * 3)), 0.15,
                 None, 0.0, {"entry": "scale_down", "exit": "cut"}),
        "fade": (["FADE", "ME"], lambda p: dict(alpha=p), 0.40, lambda p: dict(alpha=1 - p), 0.30,
                 {"entry": "fade", "exit": "fade"}),
        "typewriter": (["TYPE", "WRITER"], lambda p: dict(clip=(0, 0, max(0.001, p), 1)), 0.60, None, 0.0,
                       {"entry": "wipe", "entry_from": "left", "exit": "cut"}),
        "wipe_up": (["RISE", "UP"], lambda p: dict(clip=(0, 1 - max(0.001, ease_out_cubic(p)), 1, 1)), 0.30,
                    lambda p: dict(cy=CY + H * 0.3 * ease_in_cubic(p)), 0.25,
                    {"entry": "wipe", "entry_from": "bottom", "exit": "slide", "exit_to": "bottom"}),
        "stagger": (["ONE", "TWO", "THREE"], "stagger", 0.44, lambda p: dict(alpha=1 - p), 0.25,
                    {"entry": "stagger", "stagger_unit": "word", "exit": "fade"}),
        "cut": (["CUT"], None, 0.0, None, 0.0, {"entry": "cut", "exit": "cut"}),
        "blur_in": (["BLUR"], lambda p: dict(blur=14 * (1 - p), alpha=clamp(p * 2)), 0.30, None, 0.0,
                    {"entry": "blur_in", "exit": "cut"}),
        "rise": (["RISE"], lambda p: dict(cy=CY + H * 0.08 * (1 - ease_out_cubic(p)), alpha=p), 0.40,
                 lambda p: dict(alpha=1 - p), 0.25,
                 {"entry": "slide", "entry_from": "bottom", "exit": "fade"}),
    }


def render(name, spec):
    words, entry, e_dur, exit_, x_dur, truth = spec
    layer, spans = text_layer(words)
    path = f"{OUT}/{name}.mp4"
    proc = subprocess.Popen([FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "18", path], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        t = i / FPS
        f = plate(t)
        base = dict(cx=CX, cy=CY)
        if t < ENTRY_T:
            proc.stdin.write(f.tobytes())
            continue
        if t < EXIT_T:
            p = (t - ENTRY_T) / e_dur if e_dur else 1.0
            if entry == "stagger":
                # each word slides up 0.2 s, 0.12 s apart: composite word by word
                for wi, (sx0, sx1) in enumerate(spans):
                    wp = clamp((t - ENTRY_T - wi * 0.12) / 0.2)
                    if wp <= 0:
                        continue
                    clip = (max(0, (sx0 - 6) / layer.width), 0, min(1, (sx1 + 6) / layer.width), 1)
                    f = composite(f, layer, CX, CY + 60 * (1 - ease_out_cubic(wp)), alpha=wp, clip=clip)
            else:
                kw = {**base, **(entry(clamp(p)) if (entry and p < 1) else {})}
                f = composite(f, layer, **kw)
        else:
            p = (t - EXIT_T) / x_dur if x_dur else 1.0
            if exit_ and p < 1:
                f = composite(f, layer, **{**base, **exit_(clamp(p))})
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

    # the coarse block a 2 fps OCR pass would report: first grid sample where
    # the text is settled enough to read, last one before it leaves
    settle = ENTRY_T + (e_dur if entry != "stagger" else 0.44)
    t0 = math.ceil(settle * 2) / 2
    t1 = math.floor((EXIT_T - 0.01) * 2) / 2
    bw, bh = layer.width / W, layer.height / H
    bbox = [round(CX / W - bw / 2, 3), round(CY / H - bh / 2, 3), round(CX / W + bw / 2, 3), round(CY / H + bh / 2, 3)]
    return {"name": name, "file": f"{name}.mp4", "truth": {**truth, "entry_start": ENTRY_T, "entry_dur": e_dur,
            "exit_start": EXIT_T, "exit_dur": x_dur},
            "block": {"text": " ".join(words), "t0": t0, "t1": t1, "bbox": bbox}}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[2:]
    out = [render(n, s) for n, s in cases().items() if not only or n in only]
    with open(f"{OUT}/truth.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {len(out)} clips to {OUT}")
