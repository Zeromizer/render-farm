"""Paths, constants and the video/image helpers every module shares. Nothing here reads the cwd: every function that
touches a session takes its directory `d` ($REMAKE_HOME/sessions/<sid>) explicitly, so one process can serve many
sessions at once (stdio now, streamable HTTP later)."""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HOME = Path(os.environ.get("REMAKE_HOME", "~/.viggle-remake")).expanduser().absolute()  # job files carry absolute paths
SESSIONS, QUEUE, WORKERS, CHARS = HOME / "sessions", HOME / "queue", HOME / "workers", HOME / "chars"
STOP = Path(os.environ.get("REMAKE_STOP", HOME / "STOP"))  # one image backend's quota stop, apart from another's
FPS = 24
# 90 .. 345 (17k+5): the frame counts one render takes. A shot gets one window up to the live workers' max_frames,
# because a seam inside a shot jumps the face
LENS = [17 * k + 5 for k in range(5, 21)]
P = 17  # a hop's prefix: the last 17 frames (one VAE clip) of the take it continues
ARM = "g2"  # the paint dir name (case/paint/g2), kept so session layouts stay comparable
_TTF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT = ImageFont.truetype(str(_TTF), 22) if _TTF.exists() else ImageFont.load_default(22)


def jr(d, name, default=None):
    p = d / "case" / name
    return json.loads(p.read_text()) if p.exists() else default


def jw(d, name, obj):
    (d / "case" / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def kind(d):
    return (d / "kind").read_text().strip()


def log(d, **kw):
    """One row of the session's ledger.jsonl (image calls, LLM calls, their estimated USD)."""
    with open(d / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps({"t": time.strftime("%F %T"), **kw}, ensure_ascii=False) + "\n")


def ledger(d):
    p = d / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def spent(d):
    return sum(r.get("usd", 0) for r in ledger(d))


def frames(video, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def pick(video, fs, W, H):
    """Only frames `fs` of a video, decoded at W x H (a full decode of a 2-minute cond is 3.4 GB)."""
    sel = "+".join(f"eq(n\\,{f})" for f in sorted(set(fs)))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select={sel},scale={W}:{H}", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return dict(zip(sorted(set(fs)), np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)))


def seg(video, a, b, W, H):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=between(n\\,{a}\\,{b - 1}),scale={W}:{H}",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def probe(src):
    r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(src)],
                                  capture_output=True, check=True).stdout)
    v = next(s for s in r["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rot = [sd["rotation"] for sd in v.get("side_data_list", []) if "rotation" in sd] or [v.get("tags", {}).get("rotate", 0)]
    if round(abs(float(rot[0]))) % 180 == 90:  # a phone clip: ffmpeg decodes it rotated
        w, h = h, w
    return w, h, float(r["format"]["duration"]), any(s["codec_type"] == "audio" for s in r["streams"])


def find_cuts(v, thr=22, spike=3):
    """Hard cuts as first-frame-of-new-shot indices: mean abs diff of a quarter-size thumbnail plus a colour-histogram
    jump, both needed so fast motion inside a shot does not count; plus an isolated jump (d over `spike` x the median d
    of the 4 frames each side: cuts measure 3-45x, in-shot motion <= 1.8x) for same-palette reverse shots. Candidates
    under 7 frames apart merge, unless the later one is a hard jump (a fast montage)."""
    small = v[:, ::4, ::4].astype(np.int16)
    d = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    hist = np.stack([np.concatenate([np.histogram(f[..., c], 16, (0, 256))[0] for c in range(3)]) for f in small])
    hist = hist / hist.sum(1, keepdims=True)
    hd = np.abs(hist[1:] - hist[:-1]).sum(1)
    cut = (d > thr) & (hd > 0.25)
    nb = np.array([np.median(np.r_[d[max(0, i - 4):i], d[i + 1:i + 5]]) for i in range(len(d))])
    cut |= (d > thr) & (d / np.maximum(nb, 1) > spike)
    # a hard cut between two dark shots stays under thr: on one clip a dim hand insert to a dim back view, d 19.0 at 19x
    # its neighbours, so the insert merged into the next shot and kept the old red sleeve. d > 8 at 8x adds 13 cuts on
    # 21 clips, all 13 real (the other 12 had been added to cuts.json by hand), and none on the 9 long takes
    cut |= (d > 8) & (d / np.maximum(nb, 1) > 8)
    # between two low-key shots (both frames under luma 30) a cut measures d 16-21 at 4-8x: one clip had 7 such cuts, all
    # added to cuts.json by hand. d > 12 at 4x there adds 10 cuts on 25 clips, all real (one nobody had found);
    # the same rule ungated adds 20 false ones on bright clips
    lum = small.mean(axis=(1, 2, 3))
    cut |= (d > 12) & (d / np.maximum(nb, 1) > 4) & (np.maximum(lum[1:], lum[:-1]) < 30)
    cuts, last = [], -10
    for i in np.where(cut)[0]:
        # montage cuts 5-6 frames apart measure d 56-72, motion just after a cut ~26; a run of consecutive candidates
        # (a whip, a strobe) stays one cut
        if i + 1 - last > 6 or (i + 1 - last > 3 and d[i] > 2 * thr):
            cuts.append(int(i) + 1)
        last = int(i) + 1
    return cuts


def find_strobes(v, thr=22, run=4):
    """Spans [a, b) where the frames alternate between two pictures every frame (one test film: a close-up in every
    other frame of a wide shot for 31 frames): f and f+2 match (d under a third of the jump between them) while f+1 differs.
    find_cuts keeps such a run as one cut. None in 13 other clips."""
    small = v[:, ::4, ::4].astype(np.int16)
    d1 = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    d2 = np.abs(small[2:] - small[:-2]).mean(axis=(1, 2, 3))
    hit = np.r_[(d1[:-1] > thr) & (d1[1:] > thr) & (3 * d2 < np.minimum(d1[:-1], d1[1:])), False]
    out, i = [], None
    for f, h in enumerate(hit):
        if h and i is None:
            i = f
        elif not h and i is not None:
            if f - i >= run:
                out.append([i + 1, f + 2])
            i = None
    return out


def label(im, text):
    im = im.copy()
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((6, 4), text, font=FONT)
    d.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=(0, 0, 0))
    d.text((6, 4), text, font=FONT, fill=(255, 220, 60))
    return im


def ticks(im, box=None):
    """0.1..0.9 marks on the bottom and left edges (not the top: the panel title sits top-left), so a box (fractions of the
    frame) can be read off the image, and `box` outlined. Without them codex guessed take_keyframe's box blind, x
    0.43-0.55 then 0.47-0.61 for a man at 0.60-0.75"""
    im = im.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    for k in range(1, 10):
        x, y = round(k * W / 10), round(k * H / 10)
        d.line((x, H - 14, x, H), fill=(255, 220, 60), width=2)
        d.line((0, y, 14, y), fill=(255, 220, 60), width=2)
        d.text((x + 3, H - 30), f".{k}", font=FONT, fill=(255, 220, 60), stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((16, y - 8), f".{k}", font=FONT, fill=(255, 220, 60), stroke_width=2, stroke_fill=(0, 0, 0))
    if box:
        d.rectangle((box[0] * W, box[1] * H, box[2] * W, box[3] * H), outline=(255, 60, 60), width=3)
    return im


def grid(images, cols, labels=None, cell=None):
    """Tile PIL images row-major; `cell` = (w, h) resize, default the first image's size; no gutters, so a painted
    grid splits back at exact multiples of the cell size."""
    cw, ch = cell or images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * cw, rows * ch), (0, 0, 0))
    for i, im in enumerate(images):
        im = im.convert("RGB").resize((cw, ch), Image.LANCZOS)
        out.paste(label(im, labels[i]) if labels else im, ((i % cols) * cw, (i // cols) * ch))
    return out


def split(im, cols, rows, n=None):
    cw, ch = im.width // cols, im.height // rows
    cells = [im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)) for r in range(rows) for c in range(cols)]
    return cells[:n] if n else cells


def contact(video, size, every_s=0.5, cols=6, cell_w=416):
    """Timestamped contact sheet of a whole clip decoded at its own `size` (W, H), portrait or landscape."""
    W, H = size
    v = frames(video, W, H)
    idx = list(range(0, len(v), round(every_s * FPS)))
    cell = (cell_w, round(cell_w * H / W))
    return grid([Image.fromarray(v[i]) for i in idx], cols, [f"f{i} {i / FPS:.2f}s" for i in idx], cell), idx
