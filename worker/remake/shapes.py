"""Keyframe painting on the Codex subscription at any canvas shape.

Codex's image_gen picks its own output shape from a few it supports. Viggle's importer (paint.pool_edit) refuses a
picture more than 3 % off the canvas, so a 9:16 canvas (0.575) failed every time: the tallest image_gen returns is
984x1599 (0.615). Fix (tested 2026-11-08, 2/2 first-try passes, background 31-32 dB, faces undistorted): wrap
paint.edit so image_gen gets the frame SQUEEZED to the nearest shape it returns, and the result is squeezed back.
Canvases already within 3 % of a supported shape are untouched (16:9 worked as is).
"""
import math
from pathlib import Path

# aspect ratios (w/h) image_gen was seen returning or accepted at: 1664x960, 3:2, 1:1, 2:3, 984x1599
SHAPES = (1664 / 960, 1.5, 1.0, 2 / 3, 984 / 1599)
TOL = 0.03


def nearest(aspect):
    """The supported shape closest to `aspect`, and whether the canvas needs squeezing to reach it."""
    s = min(SHAPES, key=lambda x: abs(math.log(x / aspect)))
    return s, abs(s - aspect) > TOL * aspect


def install(pt, log=print):
    """Wrap viggle_remake_mcp.paint.edit (the one call every sheet/paint/epaint goes through)."""
    from PIL import Image   # the driver venv has Pillow; the worker venv (tests, runners) does not
    orig = pt.edit

    def edit(d, images, prompt, size, out, tag, quality="low"):
        W, H = map(int, size.split("x"))
        shape, squeeze = nearest(W / H)
        if not squeeze or not images:
            return orig(d, images, prompt, size, out, tag, quality)
        src = Image.open(images[-1]).convert("RGB")         # the frame to edit is always last
        if shape < W / H:   # taller target: keep the width, stretch the height
            tw, th = W, round(W / shape / 16) * 16
        else:               # wider target: keep the height, stretch the width
            tw, th = round(H * shape / 16) * 16, H
        tp = Path(images[-1]).with_name(Path(images[-1]).stem + "_sq.png")
        src.resize((round(src.width * tw / W), round(src.height * th / H)), Image.LANCZOS).save(tp)
        log(f"squeeze {W}x{H} ({W / H:.3f}) -> {tw}x{th} ({shape:.3f}) for image_gen")
        im, usd = orig(d, images[:-1] + [str(tp)], prompt, f"{tw}x{th}", out, tag, quality)
        im = im.resize((W, H), Image.LANCZOS)
        im.save(out)
        return im, usd

    pt.edit = edit
