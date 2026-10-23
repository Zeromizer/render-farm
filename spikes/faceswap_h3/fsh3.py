"""Face swap by H3 motion transfer: put the person from a photo into a target video's performance.

Method (what finally worked, 2026-10-23; see README.md for why):
  H3 ref2va reference-to-video: <Picture 1> = the person's photo (face + hair), <Picture 2> = the target's
  headphones cut out of the target video, <Video 1> = the target clip with its hair erased (motion only).
  10 s pieces with 2 s overlaps rendered on a one-shot RunPod RTX 5090; the best seed per piece is picked by
  motion sync; then chin-mark cleanup, colour-matched crossfade stitch, stubble-shadow reduction, sync check.

Steps (each writes into out/<name>/):
  prep        --face photo.jpg --target video.mp4   24 fps copy + 448x768 working clip
  masks       SAM3 (local ComfyUI) masks of hair / headphones / face / hand on the working clip
  cutout      --frame N   headphones cut out of a clean frame (no hand in front, both cups visible)
  mole        --frame N --x X --y Y   OPTIONAL: track a facial mole in the target and fill it
  erase-hair  the motion reference: hair (grown) minus face/hands/headphones, filled from the surroundings
  pieces      [--seeds 43,44] [--only 390]   reference pieces + H3 graphs
  cloud       render the graphs on a one-shot 5090 pod (~80 s per 10 s piece), download the renders
  score       best seed per piece by motion sync with the target; flags pieces that drifted
  finish      SAM3 face masks on the picks, chin cleanup, stitch, stubble reduction, sync report
Run with the render-farm .venv-independent venv in this folder (requirements.txt).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "worker"))
from videogen import graphs, graphs_sam3  # noqa: E402  (pure graph builders from the farm)

import comfy  # noqa: E402

W, H, FPS = 448, 768, 24
PIECE, STRIDE = 243, 195          # 10 s H3 pieces (17k+5), 2 s overlaps (15-frame overlaps gave visible jumps)
E = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))

DEFAULT_CONFIG = {
    # Identity text for <Picture 1>; keep it positive (never name unwanted things). Edit per person.
    "face_desc": ("his face, wide round face shape with full cheeks, short rounded clean-shaven chin with smooth, clear, "
                  "even skin on his chin and jaw, and his short thick messy black hair with a textured fringe over the "
                  "forehead and short neat sides"),
    # Gestures / framing / outfit / room of the target, so the regenerated scene matches it. Edit per target.
    "scene_desc": ("the same hand gestures with his index finger and open hand toward the lens, the same close handheld "
                   "selfie framing and camera motion. He wears the same light blue t-shirt with a yellow print, in the "
                   "same dim room under a grey ceiling."),
    "headphones": True,
    "seeds": [43, 44],
}


# ---------------------------------------------------------------- helpers
def work(name, *p):
    d = os.path.join(HERE, "out", name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *p)


def cfg(name):
    p = work(name, "config.json")
    if not os.path.exists(p):
        json.dump(DEFAULT_CONFIG, open(p, "w"), indent=1)
    return json.load(open(p))


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def read(path, gray=False):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return fr
        fr.append(f[:, :, 0] if gray else f)


class Writer:
    """Raw BGR frames -> mp4 (optionally with an audio track from another file)."""
    def __init__(self, path, w, h, audio=None, crf=14, lossless=False):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
        if audio:
            cmd += ["-i", audio, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264"] + (["-crf", "0", "-pix_fmt", "yuv444p"] if lossless else ["-crf", str(crf), "-pix_fmt", "yuv420p"])
        self.p = subprocess.Popen(cmd + [path], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def write_mask(path, masks):
    w = Writer(path, W, H, lossless=True)
    for m in masks:
        w.write(cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
    w.close()


def read_mask(path, n=None):
    ms = [cv2.resize(m, (W, H)) > 127 for m in read(path, gray=True)]
    return ms if n is None else (ms + [ms[-1]] * (n - len(ms)))[:n]


def sam3(video_path, noun, tag, objects):
    """Run SAM3 tracking for a noun on the local ComfyUI; return per-frame union masks."""
    name = comfy.put_input(video_path, f"fsh3_{tag}.mp4")
    files = comfy.run(graphs_sam3.build(name, noun, f"fsh3/{tag}_{noun}", objects=objects))
    tracks = [read(f, gray=True) for f in sorted(files) if f.endswith(".mp4") and f"{tag}_{noun}_obj" in f.replace("\\", "/")]
    n = max(len(t) for t in tracks)
    return [np.max([cv2.resize(t[i], (W, H)) for t in tracks if i < len(t)], axis=0) > 127 for i in range(n)]


def motion_energy(path, start=0, n=10 ** 6):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev, e = None, []
    while len(e) < n:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (112, 192)).astype(np.float32)
        e.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return np.array(e)


def window_corrs(a, b, win=72, step=24):
    n = min(len(a), len(b))
    return [float(np.corrcoef(a[s + 1:s + win], b[s + 1:s + win])[0, 1]) for s in range(0, n - win + 1, step)]


def piece_starts(n):
    s = list(range(0, n - PIECE, STRIDE))
    return s + ([n - PIECE] if n - PIECE not in s else [])


# ---------------------------------------------------------------- steps
def prep(a):
    shutil.copyfile(a.face, work(a.name, "face.jpg"))
    shutil.copyfile(a.target, work(a.name, "target.mp4"))
    ff("-i", a.target, "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", "-b:a", "192k",
       work(a.name, "target24.mp4"))
    ff("-i", work(a.name, "target24.mp4"), "-vf",
       f"crop='min(iw,ih*{W}/{H})':'min(ih,iw*{H}/{W})',scale={W}:{H},setsar=1", "-c:v", "libx264", "-crf", "10",
       "-c:a", "copy", work(a.name, "src448.mp4"))
    cfg(a.name)
    print("prepped", len(read(work(a.name, "src448.mp4"), True)), "frames; edit", work(a.name, "config.json"))


def masks(a):
    os.makedirs(work(a.name, "masks"), exist_ok=True)
    for noun, objs in (("hair", 4), ("headphones", 4), ("face", 3), ("hand", 3)):
        write_mask(work(a.name, "masks", f"{noun}.mp4"), sam3(work(a.name, "src448.mp4"), noun, f"{a.name}_src", objs))
        print("mask", noun)


def cutout(a):
    f = read(work(a.name, "src448.mp4"))[a.frame]
    m = read_mask(work(a.name, "masks", "headphones.mp4"))[a.frame].astype(np.uint8)
    m = cv2.GaussianBlur(cv2.dilate(m, E(5)).astype(np.float32), (5, 5), 0)[..., None]
    out = (f * m + 255 * (1 - m)).astype(np.uint8)
    ys, xs = np.where(m[..., 0] > 0.1)
    cv2.imwrite(work(a.name, "headphones.png"), out[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40])
    print("headphones cutout from frame", a.frame)


def mole(a):
    """Optional: LK-track a dark facial mole from a seed point (both directions), snap to the dark spot, Telea fill."""
    src = read(work(a.name, "src448.mp4"))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in src]
    med = [cv2.medianBlur(g, 15) for g in gray]

    def snap(i, x, y, r):
        x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)      # a drifting track can leave the frame
        d = gray[i].astype(np.float32) - med[i].astype(np.float32)
        x0, y0 = int(max(x - r, 0)), int(max(y - r, 0))
        win = d[y0:int(y + r + 1), x0:int(x + r + 1)]
        if win.size == 0:
            return x, y, 0.0
        yy, xx = np.unravel_index(np.argmin(win), win.shape)
        return x0 + xx, y0 + yy, float(win.min())
    x, y, _ = snap(a.frame, a.x, a.y, 22)
    pos = {a.frame: (float(x), float(y), -99.0)}
    lk = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for step in (1, -1):
        i, p = a.frame, np.array([[pos[a.frame][:2]]], np.float32)
        while 0 <= i + step < len(src):
            j = i + step
            q, st, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[j], p, None, **lk)
            if not st[0][0]:
                q = p
            q = np.clip(q, [0, 0], [W - 1, H - 1]).astype(np.float32)
            sx, sy, sc = snap(j, q[0][0][0], q[0][0][1], 5)
            if sc < -10:
                q = np.array([[[sx, sy]]], np.float32)
            pos[j] = (float(q[0][0][0]), float(q[0][0][1]), sc)
            p, i = q, j
    w = Writer(work(a.name, "src448_nomole.mp4"), W, H, audio=work(a.name, "src448.mp4"), crf=10)
    for i, f in enumerate(src):
        x, y, c = pos[i]
        if c < -10:
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (int(round(x)), int(round(y))), 5, 255, -1)
            f = cv2.inpaint(f, m, 5, cv2.INPAINT_TELEA)
        w.write(f)
    w.close()
    print("mole filled where visible")


def erase_hair(a):
    base = work(a.name, "src448_nomole.mp4") if os.path.exists(work(a.name, "src448_nomole.mp4")) else work(a.name, "src448.mp4")
    src = read(base)
    n = len(src)
    hair, hp, face, hand = (read_mask(work(a.name, "masks", f"{k}.mp4"), n) for k in ("hair", "headphones", "face", "hand"))
    w = Writer(work(a.name, "ref.mp4"), W, H, audio=work(a.name, "src448.mp4"), crf=12)
    for i, f in enumerate(src):
        m = cv2.dilate(hair[i].astype(np.uint8), E(51))
        for keep, r in ((face[i], 3), (hand[i], 9), (hp[i], 3)):
            m[cv2.dilate(keep.astype(np.uint8), E(r)) > 0] = 0
        w.write(cv2.inpaint(f, m * 255, 9, cv2.INPAINT_TELEA))
    w.close()
    print("motion reference with hair erased:", work(a.name, "ref.mp4"), f"(from {os.path.basename(base)})")


def prompt(c):
    hp = (" He wears the light grey over-ear headphones from <Picture 2> on his head, the headband arching over the top of "
          "his hair and the large ear cups covering both ears, exactly as the headphones are worn in <Video 1>.") if c["headphones"] else ""
    return (f"The man from <Picture 1> performs the exact performance of the person in <Video 1>. Keep his identity from "
            f"<Picture 1>: {c['face_desc']}.{hp} Copy <Video 1> exactly: the same head movements, facial expressions, lip "
            f"movements and speech timing, {c['scene_desc']}")


def pieces(a):
    c = cfg(a.name)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else c["seeds"]
    n = len(read(work(a.name, "ref.mp4"), True))
    starts = [int(s) for s in a.only.split(",")] if a.only else piece_starts(n)
    for d in ("pieces", "graphs"):
        os.makedirs(work(a.name, d), exist_ok=True)
    for f in glob.glob(work(a.name, "graphs", "*.json")):
        os.remove(f)                                  # the cloud step renders exactly what is in graphs/
    imgs = ["face.jpg"] + (["headphones.png"] if c["headphones"] else [])
    for s in starts:
        ff("-i", work(a.name, "ref.mp4"), "-vf", f"select='between(n\\,{s}\\,{s + PIECE - 1})',setpts=N/24/TB",
           "-af", f"atrim=start={s / FPS}:duration={PIECE / FPS},asetpts=N/SR/TB", "-c:v", "libx264", "-crf", "12",
           "-c:a", "aac", work(a.name, "pieces", f"ref_{s:03d}.mp4"))
        for seed in seeds:
            g, _ = graphs.build("r2v", {"prompt": prompt(c), "duration_s": 10, "resolution": "480p", "ratio": "9:16", "seed": seed},
                                {"ref_images": imgs, "ref_videos": [f"ref_{s:03d}.mp4"]}, f"fsh3/p{s:03d}_s{seed}")
            g["cond"]["inputs"].update(width=W, height=H, length=PIECE)
            json.dump(g, open(work(a.name, "graphs", f"p{s:03d}_s{seed}.json"), "w"), indent=1)
    print("pieces", starts, "seeds", seeds)


def cloud_run(a):
    import cloud
    job = a.name
    cloud.put(os.path.join(HERE, "pod", "pod.sh"), "fsh3/pod.sh")
    cloud.put(os.path.join(HERE, "pod", "run_graphs.py"), "fsh3/run_graphs.py")
    for sub in ("in", "graphs", "out"):
        cloud.rm_prefix(f"fsh3/jobs/{job}/{sub}/")
    cloud.put(work(job, "face.jpg"), f"fsh3/jobs/{job}/in/face.jpg")
    if os.path.exists(work(job, "headphones.png")):
        cloud.put(work(job, "headphones.png"), f"fsh3/jobs/{job}/in/headphones.png")
    gs = sorted(glob.glob(work(job, "graphs", "*.json")))
    for g in gs:
        s = re.search(r"p(\d+)_s", os.path.basename(g)).group(1)
        cloud.put(work(job, "pieces", f"ref_{s}.mp4"), f"fsh3/jobs/{job}/in/ref_{s}.mp4")
        cloud.put(g, f"fsh3/jobs/{job}/graphs/{os.path.basename(g)}")
    pod = cloud.create_pod(job)
    cloud.wait_job(job, pod)
    os.makedirs(work(job, "renders"), exist_ok=True)
    for k in cloud.ls(f"fsh3/jobs/{job}/out/"):
        if k.endswith(".mp4") or k.endswith("stats.jsonl"):
            cloud.get(k, work(job, "renders", os.path.basename(k)))
    print(open(work(job, "renders", "stats.jsonl")).read() if os.path.exists(work(job, "renders", "stats.jsonl")) else "no stats")


def score(a):
    src = motion_energy(work(a.name, "src448.mp4"))
    picks, report = {}, []
    n = len(src)
    for s in piece_starts(n):
        best = None
        for f in sorted(glob.glob(work(a.name, "renders", f"p{s:03d}_s*_00001_.mp4"))):
            cs = window_corrs(src[s:s + PIECE], motion_energy(f))
            m = (float(np.mean(cs)), float(min(cs)))
            report.append(f"piece {s:3d} {os.path.basename(f):24s} mean {m[0]:.2f} min {m[1]:5.2f}")
            if best is None or m > best[0]:
                best = (m, f)
        if best:
            picks[str(s)] = {"file": best[1], "mean": best[0][0], "min": best[0][1]}
            if best[0][0] < 0.45:
                report.append(f"  !! piece {s} drifted in every seed: re-roll with `pieces --only {s} --seeds 45,46,47` + `cloud`")
    json.dump(picks, open(work(a.name, "picks.json"), "w"), indent=1)
    print("\n".join(report))


def _chin_clean(frames, face):
    """Small dark specks in the chin zone (bottom 22% of the face, lips/mouth excluded) -> Telea fill."""
    out = []
    for f, fm in zip(frames, face):
        ys, xs = np.where(fm)
        if len(ys) < 500:
            out.append(f)
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = y1 - y0, x1 - x0
        zone = np.zeros((H, W), bool)
        zone[int(y0 + .78 * h):y1 + 1, int(x0 + .15 * w):int(x1 - .15 * w) + 1] = True
        zone &= cv2.erode(fm.astype(np.uint8), E(5)) > 0
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        lips = cv2.morphologyEx(((lab[..., 1] > np.median(lab[..., 1][fm]) + 7) & fm).astype(np.uint8), cv2.MORPH_CLOSE, E(9))
        cnts, _ = cv2.findContours(lips, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouth = np.zeros((H, W), np.uint8)
        big = [c for c in cnts if cv2.contourArea(c) > 80]
        if big:
            cv2.drawContours(mouth, big, -1, 1, -1)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        d = g.astype(np.float32) - cv2.medianBlur(g, 15).astype(np.float32)
        k, lbl, st, _ = cv2.connectedComponentsWithStats(((d < -15) & zone & ~(cv2.dilate(mouth, E(9)) > 0)).astype(np.uint8))
        m = np.zeros((H, W), np.uint8)
        for j in range(1, k):
            if 3 <= st[j, cv2.CC_STAT_AREA] <= 60:
                m[lbl == j] = 1
        out.append(cv2.inpaint(f, cv2.dilate(m, E(5)) * 255, 5, cv2.INPAINT_TELEA) if m.any() else f)
    return out


def _stitch(pieces_, total, blend=24):
    """LAB chain colour match on the overlaps + crossfade (capped at the overlap length)."""
    lab = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    xf = [None]

    def apply(k, f):
        if xf[k] is None:
            return f
        ms, ss, md, sd = xf[k]
        return cv2.cvtColor(np.clip((lab(f) - ms) / ss * sd + md, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    for k in range(1, len(pieces_)):
        (a0, fa), (b0, fb) = pieces_[k - 1], pieces_[k]
        ov = range(b0, a0 + len(fa))
        A = np.stack([lab(apply(k - 1, fa[f - a0])) for f in ov]).reshape(-1, 3)
        B = np.stack([lab(fb[f - b0]) for f in ov]).reshape(-1, 3)
        xf.append((B.mean(0), B.std(0), A.mean(0), A.std(0)))
    out = []
    for f in range(total):
        cov = [k for k, (s, fr) in enumerate(pieces_) if s <= f < s + len(fr)]
        if len(cov) > 1:
            a_, b_ = cov[-2], cov[-1]
            a_end = pieces_[a_][0] + len(pieces_[a_][1])
            bl = min(blend, a_end - pieces_[b_][0])
            if f >= a_end - bl:
                w = (f - (a_end - bl) + 1) / (bl + 1)
                out.append(cv2.addWeighted(apply(a_, pieces_[a_][1][f - pieces_[a_][0]]), 1 - w,
                                           apply(b_, pieces_[b_][1][f - pieces_[b_][0]]), w, 0))
                continue
            k = a_
        else:
            k = cov[0]
        out.append(apply(k, pieces_[k][1][f - pieces_[k][0]]))
    return out


def _destubble(frames, face, k_l=0.75):
    """Chin/jaw stubble shadow: ADD a smooth L/a/b offset toward a temporally averaged cheek tone + 5 px luminance
    closing for dots. Never replaces pixels with a blurred copy and never uses a per-frame reference colour
    (both made Shawn's rejected v3 blurry and flickery). Zone = face mask averaged over +-3 frames, ramped in from
    66-80% of the face height, lips/mouth excluded."""
    n = len(frames)
    boxes = []
    for fm in face:
        ys, xs = np.where(fm)
        boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], float) if len(ys) > 500 else None)
    sbox = lambda i, r=4: (lambda bs: np.median(bs, 0) if bs else None)([boxes[j] for j in range(max(0, i - r), min(n, i + r + 1)) if boxes[j] is not None])
    smask = lambda i, r=3: np.mean([face[j].astype(np.float32) for j in range(max(0, i - r), min(n, i + r + 1))], 0)
    labs = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in frames]
    refs = []
    for i in range(n):
        b = sbox(i)
        if b is None:
            refs.append(None)
            continue
        x0, y0, x1, y1 = b
        h, w = y1 - y0, x1 - x0
        ch = np.zeros((H, W), bool)
        ch[int(y0 + .45 * h):int(y0 + .62 * h), int(x0 + .12 * w):int(x0 + .33 * w)] = True
        ch[int(y0 + .45 * h):int(y0 + .62 * h), int(x1 - .33 * w):int(x1 - .12 * w)] = True
        ch &= face[i]
        refs.append(np.median(labs[i][ch], 0) if ch.sum() > 50 else None)
    out = []
    yy = np.arange(H, dtype=np.float32)[:, None]
    for i, f in enumerate(frames):
        b = sbox(i)
        rs = [refs[j] for j in range(max(0, i - 7), min(n, i + 8)) if refs[j] is not None]
        if b is None or not rs:
            out.append(f)
            continue
        ref = np.mean(rs, 0)
        x0, y0, x1, y1 = b
        h = y1 - y0
        lab = labs[i]
        inbox = np.zeros((H, W), bool)
        inbox[int(y0):int(y1), int(x0):int(x1)] = True
        lips = cv2.morphologyEx(((lab[..., 1] > ref[1] + 7) & inbox).astype(np.uint8), cv2.MORPH_CLOSE, E(9))
        cnts, _ = cv2.findContours(lips, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouth = np.zeros((H, W), np.uint8)
        big = [c for c in cnts if cv2.contourArea(c) > 80]
        if big:
            cv2.drawContours(mouth, big, -1, 1, -1)
        zone = cv2.erode(smask(i), E(7)) * np.clip((yy - (y0 + .66 * h)) / (.14 * h), 0, 1)
        zone[cv2.dilate(mouth, E(13)) > 0] = 0
        zone = cv2.GaussianBlur(zone, (0, 0), 6)
        low = cv2.GaussianBlur(lab, (0, 0), 6)
        lab2 = lab.copy()
        closed = cv2.morphologyEx(lab[..., 0].astype(np.uint8), cv2.MORPH_CLOSE, E(5)).astype(np.float32)
        lab2[..., 0] += zone * 0.7 * (closed - lab[..., 0])
        lab2[..., 0] += zone * k_l * np.clip(ref[0] - low[..., 0], 0, 28)
        lab2[..., 1] += zone * 0.6 * np.clip(ref[1] - low[..., 1], -8, 8)
        lab2[..., 2] += zone * 0.6 * np.clip(ref[2] - low[..., 2], -8, 8)
        out.append(cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    return out


def finish(a):
    picks = json.load(open(work(a.name, "picks.json")))
    total = len(read(work(a.name, "src448.mp4"), True))
    os.makedirs(work(a.name, "clean"), exist_ok=True)
    pcs, face_global = [], {}
    for s in sorted(picks, key=int):
        s_i = int(s)
        fr = read(picks[s]["file"])
        fm = sam3(picks[s]["file"], "face", f"{a.name}_p{s_i:03d}", 1)
        fm = (fm + [fm[-1]] * len(fr))[:len(fr)]
        fr = _chin_clean(fr, fm)
        pcs.append((s_i, fr))
        for i, m in enumerate(fm):
            face_global[s_i + i] = m              # later pieces win in overlaps (the crossfade ends on them)
        print("piece", s_i, "chin cleaned")
    frames = _stitch(pcs, total)
    face = [face_global.get(i, np.zeros((H, W), bool)) for i in range(total)]
    frames = _destubble(frames, face)
    w = Writer(work(a.name, "final.mp4"), W, H, audio=work(a.name, "target.mp4"), crf=14)
    src = read(work(a.name, "src448.mp4"))
    ws = Writer(work(a.name, "final_side-by-side.mp4"), 2 * W, H, audio=work(a.name, "target.mp4"), crf=16)
    for i, f in enumerate(frames):
        w.write(f)
        ws.write(np.hstack([src[min(i, len(src) - 1)], f]))
    w.close()
    ws.close()
    cs = window_corrs(motion_energy(work(a.name, "src448.mp4")), motion_energy(work(a.name, "final.mp4")))
    print("sync per 3 s window:", " ".join(f"{c:.2f}" for c in cs), "(~0.8 normal, <0.3 = that stretch drifted)")
    print("final:", work(a.name, "final.mp4"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "masks", "cutout", "mole", "erase-hair", "pieces", "cloud", "score", "finish"):
        p = sub.add_parser(c)
        p.add_argument("--name", required=True)
        if c == "prep":
            p.add_argument("--face", required=True)
            p.add_argument("--target", required=True)
        if c in ("cutout", "mole"):
            p.add_argument("--frame", type=int, required=True)
        if c == "mole":
            p.add_argument("--x", type=int, required=True)
            p.add_argument("--y", type=int, required=True)
        if c == "pieces":
            p.add_argument("--seeds")
            p.add_argument("--only")
    a = ap.parse_args()
    {"prep": prep, "masks": masks, "cutout": cutout, "mole": mole, "erase-hair": erase_hair, "pieces": pieces,
     "cloud": cloud_run, "score": score, "finish": finish}[a.cmd](a)
