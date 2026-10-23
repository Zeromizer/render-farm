"""Face swap by H3 motion transfer: put the person from a photo into a target video's performance.

Method (what finally worked, 2026-10-23; see README.md for why):
  H3 ref2va reference-to-video: <Picture 1> = the person's photo (face + hair), <Picture 2..> = props to keep
  (e.g. the target's headphones) cut out of the target video, <Video 1> = the target clip with its hair erased
  (motion only). 10 s pieces with 2 s overlaps rendered on a one-shot RunPod RTX 5090, one seed per piece; pieces
  whose motion drifts from the target are re-rolled with the next seed. Then chin-mark cleanup, colour-matched
  crossfade stitch, optional stubble-shadow reduction, sync report.

Steps (each writes into out/<name>/; settings live in out/<name>/config.json):
  prep        --face photo.jpg --target video.mp4 --consent "who consented, to what"   working clip + provenance
  masks       SAM3 (local ComfyUI) masks of face / hair / hand / each prop, main person only
  cutout      prop cut-outs from the best frame (largest, no hand in front); --noun N --frame F to override
  describe    fill empty face_desc / scene_desc / prop sentences with the local Qwen3-VL 4B (--dry: print only)
  mole        --frame N --x X --y Y   OPTIONAL: track a facial mole in the target and fill it
  erase-hair  the motion reference: hair (grown) minus face/hands/props, filled from the surroundings
  pieces      reference pieces + H3 graphs, first seed only ([--seeds 43,44] [--only 390] to force)
  render      cloud (one-shot 5090 pod) -> score -> re-roll drifted pieces with the next seed, until in sync
  cloud       render the pending graphs once;  score  best seed per piece + which pieces drifted
  finish      SAM3 face masks on the picks, chin cleanup, stitch, optional stubble reduction, sync report
Every step is a plain function taking a Job, so an engine can call them directly.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The stages live in worker/faceswap so the face_swap engine (runners/face_swap.py via engine.py) runs the same
# code; the one-shot pod launcher and the CLI's job folders stay in the spike (spikes/faceswap_h3).
SPIKE = os.path.join(HERE, "..", "..", "spikes", "faceswap_h3")
sys.path.insert(0, os.path.join(HERE, ".."))
from videogen import graphs, graphs_sam3  # noqa: E402  (pure graph builders from the farm)

import comfy  # noqa: E402

FPS = 24
PIECE, STRIDE = 243, 195          # 10 s H3 pieces (17k+5), 2 s overlaps (15-frame overlaps gave visible jumps)
# Working canvas per aspect. H3 needs the width divisible by 32 (432x768 crashed); 448x768 is the size that was
# validated at 9:16. The others follow the same rule (both sides /32, ~480p) but have not been rendered yet.
SIZES = {"9:16": (448, 768), "16:9": (768, 448), "1:1": (576, 576), "3:4": (480, 640), "4:3": (640, 480)}
VLM = "qwen3vl_4b_fp8_scaled.safetensors"   # Krea 2's text encoder; also on the cloud volume
E = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))

DEFAULT_CONFIG = {
    # How the prompt names the person in <Picture 1>. Set per job by whoever submits it; never guessed.
    "subject": "the person",
    "possessive": "their",
    # Identity text for <Picture 1> and the target's gestures / framing / outfit / room. Empty = `describe` fills
    # them. Keep them positive: naming unwanted things (blemish, beard, ...) tends to add them.
    "face_desc": "",
    "scene_desc": "",
    # Props of the target to keep, each becomes <Picture 2>, <Picture 3>, ... e.g. [{"noun": "headphones"}].
    # "sentence" (with {pic} for the picture tag) is filled by `describe`; "cutout_frame" by `cutout`.
    "props": [],
    "ratio": "auto",                 # one of SIZES, or auto = nearest to the target
    "mole": None,                    # {"frame", "x", "y"} when the target has a mole H3 would copy
    "chin_clean": True,              # small dark specks on the output chin (cheap, safe)
    "destubble": False,              # chin shadow reduction: ONLY for clean-shaven subjects (it lightens beards)
    "seeds": [43, 44, 45, 46],       # first seed for every piece; the rest only for pieces that drift
    "sync_min_mean": 0.45,           # a piece drifted if its mean window sync is below this ...
    "sync_min_window": 0.3,          # ... or any 3 s window is below this (both from one 34 s test video)
}

# The 2026-10-23 single-prop config ({"headphones": true}, he/his hard-coded): migrated so the old job's prompt
# and graphs come out byte-identical.
LEGACY_HP = ("He wears the light grey over-ear headphones from {pic} on his head, the headband arching over the top of "
             "his hair and the large ear cups covering both ears, exactly as the headphones are worn in <Video 1>.")


# ---------------------------------------------------------------- job
class Job:
    def __init__(self, name, root=None):
        self.name = name
        self.dir = os.path.join(root or os.path.join(SPIKE, "out"), name)
        os.makedirs(self.dir, exist_ok=True)

    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    @property
    def cfg(self):
        path = self.p("config.json")
        c = json.load(open(path)) if os.path.exists(path) else {}
        if "headphones" in c:
            hp = c.pop("headphones")
            c.setdefault("subject", "the man")
            c.setdefault("possessive", "his")
            c.setdefault("props", [{"noun": "headphones", "sentence": LEGACY_HP}] if hp else [])
        return {**DEFAULT_CONFIG, **c}

    def save_cfg(self, c):
        json.dump(c, open(self.p("config.json"), "w"), indent=1)

    @property
    def src(self):                    # the working clip (older jobs called it src448.mp4)
        return self.p("src448.mp4") if os.path.exists(self.p("src448.mp4")) else self.p("work.mp4")

    @property
    def size(self):
        r = self.cfg["ratio"]
        if r in SIZES:
            return SIZES[r]
        f = first_frame(self.src)
        return f.shape[1], f.shape[0]


def slug(noun):
    return re.sub(r"[^a-z0-9]+", "_", noun.lower()).strip("_")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- video io
def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def frames(path, gray=False):
    """Stream frames (BGR, or the first channel of a mask video)."""
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            cap.release()
            return
        yield f[:, :, 0] if gray else f


def first_frame(path):
    return next(frames(path))


def count(path):
    return sum(1 for _ in frames(path, gray=True))


class Seq:
    """Forward-only random access by frame index (pieces are consumed in timeline order)."""
    def __init__(self, path, gray=False):
        self.it, self.i, self.f = frames(path, gray), -1, None

    def get(self, idx):
        assert idx >= self.i, "Seq is forward-only"
        while self.i < idx:
            nxt = next(self.it, None)
            if nxt is None:
                break
            self.i, self.f = self.i + 1, nxt
        return self.f


class Writer:
    """Raw BGR frames -> mp4 (optionally with an audio track from another file). rgb=True is truly lossless."""
    def __init__(self, path, w, h, audio=None, crf=14, lossless=False, rgb=False):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
        if audio:
            cmd += ["-i", audio, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        if rgb:
            cmd += ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "bgr24"]
        else:
            cmd += ["-c:v", "libx264"] + (["-crf", "0", "-pix_fmt", "yuv444p"] if lossless else ["-crf", str(crf), "-pix_fmt", "yuv420p"])
        self.p = subprocess.Popen(cmd + [path], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def write_mask(path, masks, size):
    w = Writer(path, *size, lossless=True)
    for m in masks:
        w.write(cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
    w.close()


def read_mask(path, size, n=None):
    ms = [cv2.resize(m, size) > 127 for m in frames(path, gray=True)]
    return ms if n is None else (ms + [ms[-1]] * (n - len(ms)))[:n]


# ---------------------------------------------------------------- SAM3 + track selection
def sam3_tracks(video_path, noun, tag, objects, size):
    """SAM3 tracking for a noun on the local ComfyUI; one list of per-frame masks per tracked object."""
    name = comfy.put_input(video_path, f"fsh3_{tag}.mp4")
    files = comfy.run(graphs_sam3.build(name, noun, f"fsh3/{tag}_{slug(noun)}", objects=objects))
    out = []
    for f in sorted(files):
        if f.endswith(".mp4") and f"{tag}_{slug(noun)}_obj" in f.replace("\\", "/"):
            t = [cv2.resize(m, size) > 127 for m in frames(f, gray=True)]
            if any(m.any() for m in t):
                out.append(t)
    return out


def pad(masks, n, size):
    if not masks:
        return [np.zeros(size[::-1], bool)] * n
    return (masks + [masks[-1]] * n)[:n]


def main_track(tracks):
    """The main person = the largest track over the clip (a second face in frame would otherwise merge in)."""
    return max(tracks, key=lambda t: sum(int(m.sum()) for m in t)) if tracks else []


def boxes(masks, min_px=500):
    """Per-frame bounding boxes of a mask track; frames without one borrow the nearest frame's box."""
    bs = []
    for m in masks:
        ys, xs = np.where(m)
        bs.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], float) if len(ys) > min_px else None)
    missing = sum(b is None for b in bs)
    known = [i for i, b in enumerate(bs) if b is not None]
    if known:
        bs = [b if b is not None else bs[min(known, key=lambda k: abs(k - i))] for i, b in enumerate(bs)]
    return bs, missing


def near_face(tracks, face_boxes, reach):
    """Union of the tracks that sit within `reach` face-heights of the main face in most frames."""
    keep = []
    for t in tracks:
        hits = tot = 0
        for m, b in zip(t, face_boxes):
            if b is None or not m.any():
                continue
            x0, y0, x1, y1 = b
            r = reach * (y1 - y0)
            tot += 1
            hits += bool(m[max(0, int(y0 - r)):int(y1 + r) + 1, max(0, int(x0 - r)):int(x1 + r) + 1].any())
        if tot and hits / tot >= 0.5:
            keep.append(t)
    if not keep:
        return []
    return [np.max([t[i] for t in keep if i < len(t)], axis=0) for i in range(max(len(t) for t in keep))]


# ---------------------------------------------------------------- sync
def motion_energy(path):
    prev, e = None, []
    for f in frames(path):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (112, 192)).astype(np.float32)
        e.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return np.array(e)


def window_corrs(a, b, win=72, step=24):
    n = min(len(a), len(b))
    return [float(np.corrcoef(a[s + 1:s + win], b[s + 1:s + win])[0, 1]) for s in range(0, n - win + 1, step)]


def piece_starts(n):
    if n <= PIECE:              # one piece (a 10 s clip is 240 frames < 243: n - PIECE went negative, E2E 2026-10-24)
        return [0]
    s = list(range(0, n - PIECE, STRIDE))
    return s + ([n - PIECE] if n - PIECE not in s else [])


def piece_len(n):
    """Frames per piece for an n-frame clip: PIECE, or for a clip shorter than that the next 17k+5 (H3's grid),
    its reference padded with the last frame; finish() trims the result back to the clip."""
    return PIECE if n > PIECE else min(PIECE, 5 + 17 * max(0, -(-(n - 5) // 17)))


# ---------------------------------------------------------------- steps
def record_provenance(job, consent):
    if not (consent or "").strip():
        raise SystemExit("a consent statement is required (who consented, to what use)")
    json.dump({"face_sha256": sha256(job.p("face.jpg")), "target_sha256": sha256(job.p("target.mp4")),
               "consent": consent.strip(), "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
              open(job.p("provenance.json"), "w"), indent=1)


def prep(job, face, target, consent):
    if not (consent or "").strip():
        raise SystemExit("a consent statement is required (who consented, to what use)")
    shutil.copyfile(face, job.p("face.jpg"))
    shutil.copyfile(target, job.p("target.mp4"))
    record_provenance(job, consent)
    ff("-i", target, "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", "-b:a", "192k",
       job.p("target24.mp4"))
    c = job.cfg
    if c["ratio"] not in SIZES:
        h, w = first_frame(job.p("target24.mp4")).shape[:2]
        c["ratio"] = min(SIZES, key=lambda r: abs(np.log((SIZES[r][0] / SIZES[r][1]) / (w / h))))
    job.save_cfg(c)
    W, H = SIZES[c["ratio"]]
    ff("-i", job.p("target24.mp4"), "-vf", f"crop='min(iw,ih*{W}/{H})':'min(ih,iw*{H}/{W})',scale={W}:{H},setsar=1",
       "-c:v", "libx264", "-crf", "10", "-c:a", "copy", job.p("work.mp4"))
    print(f"prepped {count(job.src)} frames at {W}x{H} ({c['ratio']}); set subject/possessive/props in", job.p("config.json"))


def masks(job):
    """Face first (main person = largest track), then hair / hand / props kept only near that face."""
    c, size = job.cfg, job.size
    os.makedirs(job.p("masks"), exist_ok=True)
    n = count(job.src)
    face = pad(main_track(sam3_tracks(job.src, "face", f"{job.name}_src", 3, size)), n, size)
    write_mask(job.p("masks", "face.mp4"), face, size)
    fb, missing = boxes(face)
    print(f"mask face ({missing} of {n} frames without a face)")
    for noun, objs, reach in [("hair", 4, 1.0), ("hand", 3, 2.5)] + [(p["noun"], 4, 1.0) for p in c["props"]]:
        m = pad(near_face(sam3_tracks(job.src, noun, f"{job.name}_src", objs, size), fb, reach), n, size)
        write_mask(job.p("masks", f"{slug(noun)}.mp4"), m, size)
        print("mask", noun)


def best_prop_frame(pm, face, hand):
    """A frontal, typical-distance view of the prop: no hand over it, not cut by the frame edge, not shrunk by
    occlusion, the face near its median size and the prop centred on the face. (Largest-area picked a side-on
    close-up with one cup cut off.)"""
    fb, _ = boxes(face)
    areas = np.array([m.sum() for m in pm], float)
    med_area = np.median(areas[areas > 0]) if (areas > 0).any() else 0
    fa = [(b[2] - b[0]) * (b[3] - b[1]) if b is not None else 0 for b in fb]
    med_face = np.median([a for a in fa if a]) if any(fa) else 1

    def score(i):
        m, b = pm[i], fb[i]
        if b is None or areas[i] < 0.8 * med_area:
            return -np.inf
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            return -np.inf
        if i < len(hand) and (cv2.dilate(hand[i].astype(np.uint8), E(15)) > 0)[m].any():
            return -np.inf
        xs = np.where(m.any(0))[0]
        off = abs((xs.min() + xs.max()) / 2 - (b[0] + b[2]) / 2) / max(b[2] - b[0], 1)
        return -abs(np.log(fa[i] / med_face)) - off
    best = max(range(len(pm)), key=score)
    return best if np.isfinite(score(best)) else int(np.argmax(areas))


def cutout(job, noun=None, frame=None):
    """Each prop cut out on white from a clean frontal frame (best_prop_frame), or from --frame."""
    c, size = job.cfg, job.size
    hand = read_mask(job.p("masks", "hand.mp4"), size)
    face = read_mask(job.p("masks", "face.mp4"), size)
    for p in c["props"]:
        if noun and p["noun"] != noun:
            continue
        pm = read_mask(job.p("masks", f"{slug(p['noun'])}.mp4"), size)
        f_i = best_prop_frame(pm, face, hand) if frame is None else frame
        img = Seq(job.src).get(f_i)
        m = cv2.GaussianBlur(cv2.dilate(pm[f_i].astype(np.uint8), E(5)).astype(np.float32), (5, 5), 0)[..., None]
        out = (img * m + 255 * (1 - m)).astype(np.uint8)
        ys, xs = np.where(m[..., 0] > 0.1)
        cv2.imwrite(job.p(f"{slug(p['noun'])}.png"),
                    out[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40])
        p["cutout_frame"] = int(f_i)
        print(p["noun"], "cut out from frame", f_i)
    job.save_cfg(c)


def vlm(prompt, image, max_length=800):
    """One short answer about an image from the local Qwen3-VL 4B (Krea 2's TE) through core TextGenerate.
    Thinking mode ON: without it this checkpoint reasons out loud in the answer ("Wait, I need to...") or returns
    nothing; with it the decoded text is just the answer. Greedy, so repeatable. Its video input made up the outfit
    (a jacket and tie for a t-shirt), so describe() asks about single frames instead."""
    g = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": VLM, "type": "krea2"}},
         "2": {"class_type": "LoadImage", "inputs": {"image": comfy.put_input(image, "fsh3_vlm" + os.path.splitext(image)[1])}},
         "3": {"class_type": "TextGenerate", "inputs": {"clip": ["1", 0], "prompt": prompt, "image": ["2", 0],
                                                        "max_length": max_length, "sampling_mode": "off", "thinking": True}},
         "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}}
    text = re.sub(r"\s+", " ", comfy.run_outputs(g)["4"]["text"][0]).strip()
    text = re.sub(r"^(assistant\b:?\s*)", "", text, flags=re.I).strip().strip("\"'").rstrip(".").strip()
    return text[:1].lower() + text[1:]


def describe(job, dry=False):
    """Fill the EMPTY text fields from the photo, a frontal target frame and the prop cut-outs. Review the result:
    the model is small (check it did not name flaws, and that the outfit/room match the target)."""
    c, size = job.cfg, job.size
    S = c["subject"][0].upper() + c["subject"][1:]
    if not c["face_desc"]:
        t = vlm("List the visible face and hair traits of this person as comma-separated short phrases, each an "
                "adjective plus a noun, in this format: 'oval face shape, high cheekbones, pointed chin, smooth even "
                "skin, long wavy brown hair with a side parting'. Cover face shape, cheeks, chin, skin and hair. Plain "
                "positive words only, no blemishes or flaws. List only.", job.p("face.jpg"))
        c["face_desc"] = f"{c['possessive']} face, {t}"
    if not c["scene_desc"]:
        face = read_mask(job.p("masks", "face.mp4"), size)
        cv2.imwrite(job.p("describe_frame.jpg"), Seq(job.src).get(best_prop_frame(face, face, [])))
        ask = lambda q: vlm(q + " Phrase only.", job.p("describe_frame.jpg"))
        wear = ask("What is the person wearing on their upper body? One short noun phrase with colour and print, "
                   "e.g. 'a red hoodie with a white logo'.")
        room = ask("Describe the room and lighting behind the person in under 12 words.")
        framing = ask("Describe the camera framing of this shot in under 10 words, e.g. 'close handheld selfie framing'.")
        c["scene_desc"] = (f"the same hand gestures, the same {re.sub(r'^(the|a|an) ', '', framing)} and camera motion. "
                           f"{S} wears the same {re.sub(r'^(the|a|an) ', '', wear)}, in the same {re.sub(r'^(the|a|an) ', '', room)}.")
    for p in c["props"]:
        if not p.get("sentence"):
            t = vlm(f"What colour and type are these {p['noun']}? Answer with one short noun phrase, e.g. 'black "
                    f"on-ear headphones'. Phrase only.", job.p(f"{slug(p['noun'])}.png"))
            p["sentence"] = f"{S} wears the {re.sub(r'^(the|a|an) ', '', t)} from {{pic}}, worn exactly as in <Video 1>."
    if dry:
        print(json.dumps({k: c[k] for k in ("face_desc", "scene_desc", "props")}, indent=1))
    else:
        job.save_cfg(c)
    print(prompt(c))


def mole(job, frame, x, y):
    """Optional: LK-track a dark facial mole from a seed point (both directions), snap to the dark spot, Telea fill."""
    c = job.cfg
    c["mole"] = {"frame": frame, "x": x, "y": y}
    job.save_cfg(c)
    W, H = job.size
    src = list(frames(job.src))
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
    x, y, _ = snap(frame, x, y, 22)
    pos = {frame: (float(x), float(y), -99.0)}
    lk = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for step in (1, -1):
        i, p = frame, np.array([[pos[frame][:2]]], np.float32)
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
    w = Writer(nomole_path(job), W, H, audio=job.src, crf=10)
    for i, f in enumerate(src):
        x, y, c_ = pos[i]
        if c_ < -10:
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (int(round(x)), int(round(y))), 5, 255, -1)
            f = cv2.inpaint(f, m, 5, cv2.INPAINT_TELEA)
        w.write(f)
    w.close()
    print("mole filled where visible")


def nomole_path(job):
    return job.src.replace(".mp4", "_nomole.mp4")


def erase_hair(job):
    c, size = job.cfg, job.size
    base = nomole_path(job) if c["mole"] and os.path.exists(nomole_path(job)) else job.src
    n = count(base)
    keeps = [(read_mask(job.p("masks", "face.mp4"), size, n), 3), (read_mask(job.p("masks", "hand.mp4"), size, n), 9)]
    keeps += [(read_mask(job.p("masks", f"{slug(p['noun'])}.mp4"), size, n), 3) for p in c["props"]]
    hair = read_mask(job.p("masks", "hair.mp4"), size, n)
    w = Writer(job.p("ref.mp4"), *size, audio=job.src, crf=12)
    for i, f in enumerate(frames(base)):
        m = cv2.dilate(hair[i].astype(np.uint8), E(51))
        for k, r in keeps:
            m[cv2.dilate(k[i].astype(np.uint8), E(r)) > 0] = 0
        w.write(cv2.inpaint(f, m * 255, 9, cv2.INPAINT_TELEA))
    w.close()
    print("motion reference with hair erased:", job.p("ref.mp4"), f"(from {os.path.basename(base)})")


def prompt(c):
    if not c["face_desc"] or not c["scene_desc"] or any(not p.get("sentence") for p in c["props"]):
        raise SystemExit("face_desc / scene_desc / prop sentences are empty: run `describe` or fill config.json")
    S = c["subject"][0].upper() + c["subject"][1:]
    props = "".join(" " + p["sentence"].format(pic=f"<Picture {i + 2}>") for i, p in enumerate(c["props"]))
    return (f"{S} from <Picture 1> performs the exact performance of the person in <Video 1>. Keep {c['possessive']} "
            f"identity from <Picture 1>: {c['face_desc']}.{props} Copy <Video 1> exactly: the same head movements, "
            f"facial expressions, lip movements and speech timing, {c['scene_desc']}")


def ref_images(c):
    return ["face.jpg"] + [f"{slug(p['noun'])}.png" for p in c["props"]]


def _graph(job, c, s, seed):
    W, H = job.size
    ratio = c["ratio"] if c["ratio"] in SIZES else min(SIZES, key=lambda r: abs(SIZES[r][0] / SIZES[r][1] - W / H))
    g, _ = graphs.build("r2v", {"prompt": prompt(c), "duration_s": 10, "resolution": "480p", "ratio": ratio, "seed": seed},
                        {"ref_images": ref_images(c), "ref_videos": [f"ref_{s:03d}.mp4"]}, f"fsh3/p{s:03d}_s{seed}")
    g["cond"]["inputs"].update(width=W, height=H, length=piece_len(count(job.p("ref.mp4"))))
    json.dump(g, open(job.p("graphs", f"p{s:03d}_s{seed}.json"), "w"), indent=1)


def pieces(job, seeds=None, only=None):
    """Reference pieces + graphs. Default: the first configured seed for every piece (re-rolls come from `render`)."""
    c = job.cfg
    seeds = seeds or c["seeds"][:1]
    n = count(job.p("ref.mp4"))
    starts, L = only or piece_starts(n), piece_len(n)
    for d in ("pieces", "graphs"):
        os.makedirs(job.p(d), exist_ok=True)
    for f in glob.glob(job.p("graphs", "*.json")):
        os.remove(f)
    for s in starts:
        ff("-i", job.p("ref.mp4"), "-vf", f"select='between(n\\,{s}\\,{s + L - 1})',setpts=N/24/TB,"
           f"tpad=stop_mode=clone:stop={L}", "-frames:v", str(L),
           "-af", f"atrim=start={s / FPS}:duration={L / FPS},asetpts=N/SR/TB", "-c:v", "libx264", "-crf", "12",
           "-c:a", "aac", job.p("pieces", f"ref_{s:03d}.mp4"))
        for seed in seeds:
            _graph(job, c, s, seed)
    print("pieces", starts, "seeds", seeds)


def render_path(job, s, seed):
    return job.p("renders", f"p{s:03d}_s{seed}_00001_.mp4")


def pending(job):
    out = []
    for g in sorted(glob.glob(job.p("graphs", "*.json"))):
        s, seed = map(int, re.match(r"p(\d+)_s(\d+)", os.path.basename(g)).groups())
        if not os.path.exists(render_path(job, s, seed)):
            out.append((g, s))
    return out


def cloud_run(job):
    """Render the pending graphs on one one-shot 5090 pod and download the results (the spike's CLI path; the
    engine passes its own run= that renders on the worker's ComfyUI)."""
    sys.path.insert(0, SPIKE)
    import cloud
    todo = pending(job)
    if not todo:
        print("nothing pending")
        return
    jid = job.name
    cloud.put(os.path.join(SPIKE, "pod", "pod.sh"), "fsh3/pod.sh")
    cloud.put(os.path.join(SPIKE, "pod", "run_graphs.py"), "fsh3/run_graphs.py")
    for sub in ("in", "graphs", "out"):
        cloud.rm_prefix(f"fsh3/jobs/{jid}/{sub}/")
    for img in ref_images(job.cfg):
        cloud.put(job.p(img), f"fsh3/jobs/{jid}/in/{img}")
    for g, s in todo:
        cloud.put(job.p("pieces", f"ref_{s:03d}.mp4"), f"fsh3/jobs/{jid}/in/ref_{s:03d}.mp4")
        cloud.put(g, f"fsh3/jobs/{jid}/graphs/{os.path.basename(g)}")
    pod = cloud.create_pod(jid)
    cloud.wait_job(jid, pod)
    os.makedirs(job.p("renders"), exist_ok=True)
    for k in cloud.ls(f"fsh3/jobs/{jid}/out/"):
        if k.endswith(".mp4"):
            cloud.get(k, job.p("renders", os.path.basename(k)))
        elif k.endswith("stats.jsonl"):
            cloud.get(k, job.p("renders", "stats_tmp.jsonl"))
            open(job.p("renders", "stats.jsonl"), "a").write(open(job.p("renders", "stats_tmp.jsonl")).read())
            os.remove(job.p("renders", "stats_tmp.jsonl"))
    left = pending(job)
    if left:
        print("!! not rendered:", [os.path.basename(g) for g, _ in left], "(see renders/stats.jsonl)")


def score(job, quiet=False):
    """Best seed per piece by motion sync with the target. Returns (picks, starts that drifted in every seed)."""
    c = job.cfg
    src = motion_energy(job.src)
    picks, failing, report = {}, [], []
    for s in piece_starts(len(src)):
        best = None
        for f in sorted(glob.glob(job.p("renders", f"p{s:03d}_s*_00001_.mp4"))):
            cs = window_corrs(src[s:s + PIECE], motion_energy(f))
            m = (float(np.mean(cs)), float(min(cs)))
            report.append(f"piece {s:3d} {os.path.basename(f):24s} mean {m[0]:.2f} min {m[1]:5.2f}")
            if best is None or m > best[0]:
                best = (m, f)
        if best is None:
            failing.append(s)
            continue
        picks[str(s)] = {"file": best[1], "mean": best[0][0], "min": best[0][1]}
        if best[0][0] < c["sync_min_mean"] or best[0][1] < c["sync_min_window"]:
            failing.append(s)
            report.append(f"  !! piece {s} drifted (mean {best[0][0]:.2f}, min {best[0][1]:.2f})")
    json.dump(picks, open(job.p("picks.json"), "w"), indent=1)
    if not quiet:
        print("\n".join(report))
    return picks, failing


def reroll(job, starts):
    """Graphs with the next untried seed for each drifted piece. Returns how many were queued."""
    c, k = job.cfg, 0
    for s in starts:
        tried = {int(m) for m in re.findall(rf"p{s:03d}_s(\d+)_", " ".join(os.listdir(job.p("renders"))))}
        nxt = [seed for seed in c["seeds"] if seed not in tried]
        if nxt:
            _graph(job, c, s, nxt[0])
            k += 1
            print(f"piece {s}: re-roll with seed {nxt[0]}")
        else:
            print(f"!! piece {s}: all seeds {c['seeds']} drifted; add seeds to config.json or accept the best one")
    return k


def render(job, run=cloud_run):
    """cloud -> score -> re-roll the drifted pieces, until every piece is in sync or the seeds run out."""
    while True:
        run(job)
        picks, failing = score(job)
        if not failing or not reroll(job, failing):
            return picks


# ---------------------------------------------------------------- finish
def chin_clean(f, fm):
    """Small dark specks in the chin zone (bottom 22% of the face, lips/mouth excluded) -> Telea fill."""
    H, W = f.shape[:2]
    ys, xs = np.where(fm)
    if len(ys) < 500:
        return f
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
    return cv2.inpaint(f, cv2.dilate(m, E(5)) * 255, 5, cv2.INPAINT_TELEA) if m.any() else f


def lab(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)


class Stitch:
    """Streams the timeline from the cleaned pieces: LAB chain colour match on the overlaps + crossfade capped at
    the overlap length. Holds one frame per piece, not the clip. Also streams the face mask (later pieces win)."""
    def __init__(self, job, starts, blend=24, total=None):
        self.job, self.starts, self.blend, self.total = job, starts, blend, total
        self.lens = {s: count(self.clean(s)) for s in starts}
        self.xf = {starts[0]: None}
        for a0, b0 in zip(starts, starts[1:]):
            ov = range(b0, a0 + self.lens[a0])
            ra, rb = Seq(self.clean(a0)), Seq(self.clean(b0))
            A = np.stack([lab(self.apply(a0, ra.get(f - a0))) for f in ov]).reshape(-1, 3)
            B = np.stack([lab(rb.get(f - b0)) for f in ov]).reshape(-1, 3)
            self.xf[b0] = (B.mean(0), B.std(0), A.mean(0), A.std(0))

    def clean(self, s):
        return self.job.p("clean", f"p{s:03d}.mp4")

    def apply(self, s, f):
        if self.xf[s] is None:
            return f
        ms, ss, md, sd = self.xf[s]
        return cv2.cvtColor(np.clip((lab(f) - ms) / ss * sd + md, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def __iter__(self):
        """Yields (frame, face_mask) for the whole timeline."""
        rd = {s: Seq(self.clean(s)) for s in self.starts}
        mk = {s: Seq(self.job.p("clean", f"face_p{s:03d}.mp4"), gray=True) for s in self.starts}
        size = self.job.size
        end = self.starts[-1] + self.lens[self.starts[-1]]
        for f in range(min(end, self.total) if self.total else end):
            cov = [s for s in self.starts if s <= f < s + self.lens[s]]
            last = cov[-1]
            m = cv2.resize(mk[last].get(f - last), size) > 127
            if len(cov) > 1:
                a, b = cov[-2], cov[-1]
                a_end = a + self.lens[a]
                bl = min(self.blend, a_end - b)
                if f >= a_end - bl:
                    w = (f - (a_end - bl) + 1) / (bl + 1)
                    yield cv2.addWeighted(self.apply(a, rd[a].get(f - a)), 1 - w, self.apply(b, rd[b].get(f - b)), w, 0), m
                    continue
                k = a
            else:
                k = cov[0]
            yield self.apply(k, rd[k].get(f - k)), m


def destubble(stream, n, k_l=0.75):
    """Chin/jaw stubble shadow on a (frame, face_mask) stream: ADD a smooth L/a/b offset toward a temporally
    averaged cheek tone + 5 px luminance closing for dots. Never replaces pixels with a blurred copy and never uses
    a per-frame reference colour (both made Shawn's rejected v3 blurry and flickery). Zone = face mask averaged over
    +-3 frames, ramped in from 66-80% of the face height, lips/mouth excluded. Three passes over `stream()` keep
    only a 7-frame window in memory. Only for clean-shaven subjects: it lightens a real beard too."""
    raw = []
    for _, m in stream():
        ys, xs = np.where(m)
        raw.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], float) if len(ys) > 500 else None)
    sbox = lambda i, r=4: (lambda bs: np.median(bs, 0) if bs else None)([raw[j] for j in range(max(0, i - r), min(n, i + r + 1)) if raw[j] is not None])
    refs = []
    for i, (f, m) in enumerate(stream()):
        b = sbox(i)
        if b is None:
            refs.append(None)
            continue
        H, W = f.shape[:2]
        x0, y0, x1, y1 = b
        h, w = y1 - y0, x1 - x0
        ch = np.zeros((H, W), bool)
        ch[int(y0 + .45 * h):int(y0 + .62 * h), int(x0 + .12 * w):int(x0 + .33 * w)] = True
        ch[int(y0 + .45 * h):int(y0 + .62 * h), int(x1 - .33 * w):int(x1 - .12 * w)] = True
        ch &= m
        refs.append(np.median(lab(f)[ch], 0) if ch.sum() > 50 else None)
    win, it, nxt_j = collections.deque(), iter(stream()), 0     # (index, frame, mask) for frames i-3 .. i+3
    for i in range(n):
        while nxt_j <= min(n - 1, i + 3):
            item = next(it, None)
            if item is None:
                break
            win.append((nxt_j, *item))
            nxt_j += 1
        while win and win[0][0] < i - 3:
            win.popleft()
        f = next(fr for j, fr, _ in win if j == i)
        b = sbox(i)
        rs = [refs[j] for j in range(max(0, i - 7), min(n, i + 8)) if refs[j] is not None]
        if b is None or not rs:
            yield f
            continue
        H, W = f.shape[:2]
        yy = np.arange(H, dtype=np.float32)[:, None]
        ref = np.mean(rs, 0)
        x0, y0, x1, y1 = b
        h = y1 - y0
        L = lab(f)
        inbox = np.zeros((H, W), bool)
        inbox[int(y0):int(y1), int(x0):int(x1)] = True
        lips = cv2.morphologyEx(((L[..., 1] > ref[1] + 7) & inbox).astype(np.uint8), cv2.MORPH_CLOSE, E(9))
        cnts, _ = cv2.findContours(lips, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouth = np.zeros((H, W), np.uint8)
        big = [c for c in cnts if cv2.contourArea(c) > 80]
        if big:
            cv2.drawContours(mouth, big, -1, 1, -1)
        smask = np.mean([m.astype(np.float32) for j, _, m in win if i - 3 <= j <= i + 3], 0)
        zone = cv2.erode(smask, E(7)) * np.clip((yy - (y0 + .66 * h)) / (.14 * h), 0, 1)
        zone[cv2.dilate(mouth, E(13)) > 0] = 0
        zone = cv2.GaussianBlur(zone, (0, 0), 6)
        low = cv2.GaussianBlur(L, (0, 0), 6)
        L2 = L.copy()
        closed = cv2.morphologyEx(L[..., 0].astype(np.uint8), cv2.MORPH_CLOSE, E(5)).astype(np.float32)
        L2[..., 0] += zone * 0.7 * (closed - L[..., 0])
        L2[..., 0] += zone * k_l * np.clip(ref[0] - low[..., 0], 0, 28)
        L2[..., 1] += zone * 0.6 * np.clip(ref[1] - low[..., 1], -8, 8)
        L2[..., 2] += zone * 0.6 * np.clip(ref[2] - low[..., 2], -8, 8)
        yield cv2.cvtColor(np.clip(L2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def finish(job):
    c, size = job.cfg, job.size
    picks = json.load(open(job.p("picks.json")))
    starts = sorted(int(s) for s in picks)
    os.makedirs(job.p("clean"), exist_ok=True)
    for s in starts:
        f = picks[str(s)]["file"]
        n_p = count(f)
        fm = pad(main_track(sam3_tracks(f, "face", f"{job.name}_p{s:03d}", 2, size)), n_p, size)
        w = Writer(job.p("clean", f"p{s:03d}.mp4"), *size, rgb=True)
        for i, fr in enumerate(frames(f)):
            w.write(chin_clean(fr, fm[i]) if c["chin_clean"] else fr)
        w.close()
        write_mask(job.p("clean", f"face_p{s:03d}.mp4"), fm, size)
        print("piece", s, "cleaned" if c["chin_clean"] else "masked")
    n_src = count(job.src)                # a short clip's one piece is padded to H3's grid: trim it back
    st = Stitch(job, starts, total=n_src)
    n = min(starts[-1] + st.lens[starts[-1]], n_src)
    out = destubble(lambda: iter(st), n) if c["destubble"] else (fr for fr, _ in st)
    W, H = size
    w = Writer(job.p("final.mp4"), W, H, audio=job.p("target.mp4"), crf=14)
    ws = Writer(job.p("final_side-by-side.mp4"), 2 * W, H, audio=job.p("target.mp4"), crf=16)
    src = Seq(job.src)                    # past its end, get() keeps returning the last frame
    for i, fr in enumerate(out):
        w.write(fr)
        ws.write(np.hstack([src.get(i), fr]))
    w.close()
    ws.close()
    cs = window_corrs(motion_energy(job.src), motion_energy(job.p("final.mp4")))
    prov = json.load(open(job.p("provenance.json"))) if os.path.exists(job.p("provenance.json")) else {}
    json.dump({"final_sha256": sha256(job.p("final.mp4")), "side_by_side_sha256": sha256(job.p("final_side-by-side.mp4")),
               **prov, "prompt": prompt(c), "picks": {s: os.path.basename(p["file"]) for s, p in picks.items()},
               "sync_windows": [round(x, 3) for x in cs], "chin_clean": c["chin_clean"], "destubble": c["destubble"]},
              open(job.p("final.json"), "w"), indent=1)
    if not prov:
        print("!! no provenance.json (no consent recorded for this job)")
    print("sync per 3 s window:", " ".join(f"{x:.2f}" for x in cs), "(~0.8 normal, <0.3 = that stretch drifted)")
    print("final:", job.p("final.mp4"))


# ---------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = ("prep", "masks", "cutout", "describe", "mole", "erase-hair", "pieces", "render", "cloud", "score", "finish")
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        if name == "prep":
            p.add_argument("--face", required=True)
            p.add_argument("--target", required=True)
            p.add_argument("--consent", required=True, help="who consented, to what use (stored with input hashes)")
        if name == "cutout":
            p.add_argument("--noun")
            p.add_argument("--frame", type=int)
        if name == "describe":
            p.add_argument("--dry", action="store_true")
        if name == "mole":
            p.add_argument("--frame", type=int, required=True)
            p.add_argument("--x", type=int, required=True)
            p.add_argument("--y", type=int, required=True)
        if name == "pieces":
            p.add_argument("--seeds")
            p.add_argument("--only")
    a = ap.parse_args()
    job = Job(a.name)
    ints = lambda s: [int(x) for x in s.split(",")] if s else None
    {"prep": lambda: prep(job, a.face, a.target, a.consent),
     "masks": lambda: masks(job),
     "cutout": lambda: cutout(job, a.noun, a.frame),
     "describe": lambda: describe(job, a.dry),
     "mole": lambda: mole(job, a.frame, a.x, a.y),
     "erase-hair": lambda: erase_hair(job),
     "pieces": lambda: pieces(job, ints(a.seeds), ints(a.only)),
     "render": lambda: render(job),
     "cloud": lambda: cloud_run(job),
     "score": lambda: score(job),
     "finish": lambda: finish(job)}[a.cmd]()
