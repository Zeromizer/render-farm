"""Face swap, pixel side: put a person from a photo into a video's performance with
MiniMax H3 reference-to-video ("motion transfer"), for the face_swap runner.

Why this and not a face-swap model: the BFS head-swap LoRAs (LTX-2.5 / LTX-2.3 / H3,
tested 2026-10-23) kept the target's long hair and jaw, and H3 masked inpaint regrows
hair from the surrounding context. H3 r2v with <Picture 1> = the person's photo and
<Video 1> = the target with its hair painted out takes only the motion from the video
and the identity from the photo: Shawn accepted that one (render-farm spike
spikes/faceswap_h3, 127d95f). Background, clothes and hands are regenerated too, so
the result follows the video closely but is not pixel-identical to it.

Subcommands, each driven by a spec JSON the runner writes:

  prep    trim the target to the requested range at 24 fps, centre-crop it to the
          generation canvas (448x768 portrait, 768x448 landscape, 576x576 square)
          -> src.mp4 (+ its audio), and plan the pieces -> plan.json
  build   pick the subject's tracks out of the SAM 3.1 masks the runner made (face,
          hair, hands, props), cut each prop out onto white as an extra reference
          picture, paint the hair out of every frame (so H3 takes the hair from the
          photo) -> ref.mp4, and cut it into overlapping pieces ref_<start>.mp4
  score   how closely each render follows the target's motion: per-frame motion
          energy, Pearson-correlated over 3 s windows -> mean / min per render
  finish  clean the chin (the target's moles and stubble shadow leak through),
          colour-match each piece to the one before it over their overlap, crossfade
          the joins, put the target's sound back -> out.mp4, a proof sheet and the
          report

Run in the planar_patch venv (opencv-python-headless + numpy); ffmpeg as lettering.
Video is streamed frame by frame wherever it can be, so long clips do not have to fit
in RAM.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "lettering"))
import lettering as L  # noqa: E402  (io helpers: one implementation)

FPS = L.FPS
emit = L.emit
log = lambda s: print(s, flush=True)  # noqa: E731

# Canvases: H3 needs both sides divisible by 32 (432 wide crashed the sampler); 448x768
# is what the accepted render used.
CANVAS = {"portrait": (448, 768), "landscape": (768, 448), "square": (576, 576)}
# Pieces: 10 s with 2 s overlaps on the cloud card (a 5090 renders one in ~80 s); the
# 16 GB PC peaked at ~31/31 GB host RAM on 5 s at this size, so it gets 5 s pieces.
PIECES = {"cloud": (243, 195), "pc": (124, 100)}
MASK_ON = 127
MOTION_WINDOW, MOTION_STEP = 72, 24
XFADE = 24


def ffmpeg():
    return L.find_ffmpeg_tool("ffmpeg")


def probe_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def frames(path):
    """Yield the frames of a video, BGR."""
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    finally:
        cap.release()


def mask_frames(path, size):
    """Yield one boolean mask per frame of a SAM mask video, at size (w, h)."""
    for f in frames(path):
        g = f[..., 0] if f.ndim == 3 else f
        if (g.shape[1], g.shape[0]) != size:
            g = cv2.resize(g, size, interpolation=cv2.INTER_NEAREST)
        yield g > MASK_ON


class Writer:
    """Frames piped into ffmpeg one at a time (lettering.write_clip needs them all in memory)."""

    def __init__(self, path, size, audio_from=None, audio_start=0.0, seconds=None, lossless=False):
        w, h = size
        cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
        if audio_from:
            cmd += ["-ss", f"{audio_start:.4f}", "-i", audio_from, "-map", "0:v:0", "-map", "1:a?"]
        cmd += (["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p"] if lossless
                else ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"])
        if audio_from:
            cmd += ["-c:a", "aac", "-b:a", "192k"] + (["-t", f"{seconds:.4f}"] if seconds else [])
        self.path, self.n = path, 0
        self.p = subprocess.Popen(cmd + [path], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def canvas_for(w, h):
    r = w / h
    if r < 0.8:
        return "portrait", CANVAS["portrait"]
    if r > 1.25:
        return "landscape", CANVAS["landscape"]
    return "square", CANVAS["square"]


def crop_to(f, size):
    """Centre-crop to the canvas aspect, then scale to it."""
    W, H = size
    h, w = f.shape[:2]
    if w / h > W / H:
        cw = int(round(h * W / H))
        x = (w - cw) // 2
        f = f[:, x:x + cw]
    else:
        ch = int(round(w * H / W))
        y = (h - ch) // 2
        f = f[y:y + ch]
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)


def plan_pieces(n, piece, stride):
    """Starts of overlapping pieces covering n frames; the last one ends on the last frame.
    A clip shorter than a piece is one piece, padded by holding its last frame."""
    if n <= piece:
        return [0]
    starts = list(range(0, n - piece, stride))
    if starts[-1] != n - piece:
        starts.append(n - piece)
    return starts


# ---------------------------------------------------------------- prep

def prep(spec):
    work = spec["work_dir"]
    tgt = spec["target"]
    t0 = float(spec.get("start_s") or 0.0)
    t1 = spec.get("end_s")
    emit("PHASE", "trimming the target to 24 fps")
    t24 = os.path.join(work, "target24.mp4")
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t0:.3f}", "-i", tgt]
    if t1 is not None:
        cmd += ["-t", f"{float(t1) - t0:.3f}"]
    cmd += ["-vf", f"fps={FPS}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
            t24]
    subprocess.run(cmd, check=True)
    n, w, h = probe_frames(t24)
    if n < 24:
        raise RuntimeError(f"the target is too short ({n} frames at 24 fps); give at least 1 s of video")
    shape, size = canvas_for(w, h)
    emit("PHASE", f"cropping to the {size[0]}x{size[1]} canvas")
    src = os.path.join(work, "src.mp4")
    out = Writer(src, size, audio_from=t24 if L.has_audio(t24) else None, lossless=True)
    count = 0
    for f in frames(t24):
        out.write(crop_to(f, size))
        count += 1
    out.close()
    piece, stride = PIECES["cloud" if spec.get("cloud") else "pc"]
    starts = plan_pieces(count, piece, stride)
    warnings = []
    if (w, h) != size and min(w, h) > min(size):
        warnings.append(f"the result is generated at {size[0]}x{size[1]} (the target is {w}x{h}); "
                        f"upscale it afterwards if it needs to be larger")
    if abs(w / h - size[0] / size[1]) > 0.02:
        warnings.append(f"the target ({w}x{h}) is centre-cropped to {size[0]}:{size[1]}")
    plan = {"frames": count, "canvas": list(size), "shape": shape, "piece": piece, "stride": stride, "starts": starts,
            "target_size": [w, h], "audio": L.has_audio(t24), "warnings": warnings}
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
    emit("PROGRESS", 100)
    log(f"prep: {count} frames, canvas {size[0]}x{size[1]} ({shape}), pieces at {starts}")


# ---------------------------------------------------------------- tracks

def track_stats(path, size):
    """Per-frame area and centroid of one mask video: (areas[n], cx[n], cy[n], boxes[n])."""
    areas, cxs, cys, boxes = [], [], [], []
    for m in mask_frames(path, size):
        ys, xs = np.nonzero(m)
        areas.append(len(xs))
        if len(xs):
            cxs.append(xs.mean())
            cys.append(ys.mean())
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        else:
            cxs.append(np.nan)
            cys.append(np.nan)
            boxes.append(None)
    return np.array(areas, float), np.array(cxs), np.array(cys), boxes


def pick_subject(face_paths, size):
    """The subject's face track: big and central over the clip (a second person in shot
    must not merge into the face or the hair, as a max over every track did)."""
    W, H = size
    best, best_score = None, -1.0
    for p in face_paths:
        a, cx, cy, boxes = track_stats(p, size)
        if not a.any():
            continue
        seen = a > 0
        off = np.nanmean(np.hypot((cx[seen] - W / 2) / W, (cy[seen] - H / 2) / H))
        score = a.mean() * (1.0 - min(0.9, off))
        if score > best_score:
            best, best_score = (p, a, boxes), score
    return best


def person_region(boxes, size, n):
    """Per-frame box around the subject's head (face box grown to take in the hair and
    anything worn on the head), held across frames where the face is not found."""
    W, H = size
    out, last = [], None
    for b in boxes[:n] + [None] * max(0, n - len(boxes)):
        if b is not None:
            x0, y0, x1, y1 = b
            bw, bh = x1 - x0, y1 - y0
            last = (max(0, int(x0 - 0.9 * bw)), max(0, int(y0 - 1.1 * bh)), min(W, int(x1 + 0.9 * bw)),
                    min(H, int(y1 + 0.6 * bh)))
        out.append(last)
    first = next((r for r in out if r is not None), (0, 0, W, H))
    return [r or first for r in out]


def belongs(path, region, size, share=0.3):
    """True when most of a track lies in the subject's head region (their hair, their cap)."""
    inside = total = 0
    for m, r in zip(mask_frames(path, size), region):
        x0, y0, x1, y1 = r
        inside += int(m[y0:y1, x0:x1].sum())
        total += int(m.sum())
    return total > 0 and inside / total >= share


def union_video(paths, size, n, dilate=0, dest=None):
    """Union of several mask videos -> list of boolean masks (dilated), or written to dest."""
    out = [np.zeros((size[1], size[0]), bool) for _ in range(n)]
    for p in paths:
        for i, m in enumerate(mask_frames(p, size)):
            if i < n:
                out[i] |= m
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        out = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in out]
    return out


# ---------------------------------------------------------------- build

def cutout(src_path, masks, size, dest, margin=40):
    """A prop on white, cut from the frame where it is largest with no hand over it."""
    hands = masks["hands"]
    best_i, best_a = None, 0
    for i, m in enumerate(masks["prop"]):
        clear = m & ~hands[i] if i < len(hands) else m
        if m.sum() and clear.sum() / m.sum() > 0.97 and m.sum() > best_a:
            best_i, best_a = i, int(m.sum())
    if best_i is None:
        best_i = int(np.argmax([m.sum() for m in masks["prop"]]))
        best_a = int(masks["prop"][best_i].sum())
    if not best_a:
        return None
    frame = next(f for i, f in enumerate(frames(src_path)) if i == best_i)
    # dilated 5 px and feathered onto white, as in the accepted render
    m = cv2.GaussianBlur(cv2.dilate(masks["prop"][best_i].astype(np.uint8),
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32),
                         (5, 5), 0)[..., None]
    canvas = (frame * m + 255 * (1 - m)).astype(np.uint8)
    ys, xs = np.nonzero(m[..., 0] > 0.1)
    x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x1, y1 = min(size[0], xs.max() + margin), min(size[1], ys.max() + margin)
    cv2.imwrite(dest, canvas[y0:y1, x0:x1])
    return {"frame": best_i, "area": best_a, "path": dest}


def build(spec):
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    size, n = tuple(plan["canvas"]), plan["frames"]
    src = os.path.join(work, "src.mp4")
    report = {"warnings": list(plan.get("warnings") or [])}

    emit("PHASE", "finding the subject")
    subject = pick_subject(spec["masks"].get("face") or [], size)
    if subject is None:
        raise RuntimeError("no face was found in the target video; it needs a person facing the camera")
    face_path, face_area, face_boxes = subject
    seen = float((face_area > 0).mean())
    report["face_seen"] = round(seen, 3)
    if seen < 0.6:
        report["warnings"].append(f"the face is visible in only {seen:.0%} of the frames; expect a weaker likeness")
    region = person_region(face_boxes, size, n)
    hair = [p for p in spec["masks"].get("hair") or [] if belongs(p, region, size)]
    props = {}
    for noun, paths in (spec["masks"].get("props") or {}).items():
        mine = [p for p in paths if belongs(p, region, size, share=0.2)]
        if not mine:
            report["warnings"].append(f"could not find {noun!r} on the subject; it is left to the model")
        props[noun] = mine
    emit("PROGRESS", 20)

    face = union_video([face_path], size, n, dilate=3)
    hands = union_video(spec["masks"].get("hand") or [], size, n, dilate=9)
    worn = [union_video(p, size, n, dilate=3) for p in props.values() if p]
    keep = [face[i] | hands[i] | np.any([w[i] for w in worn], axis=0) if worn else face[i] | hands[i]
            for i in range(n)]

    emit("PHASE", "cutting out the props")
    refs = []
    for k, (noun, paths) in enumerate(props.items()):
        if not paths:
            continue
        got = cutout(src, {"prop": union_video(paths, size, n), "hands": hands}, size,
                     os.path.join(work, f"prop_{k}.png"))
        if got:
            refs.append({"noun": noun, **got})
    report["props"] = refs
    emit("PROGRESS", 40)

    erase = spec.get("hair", "from_photo") == "from_photo"
    emit("PHASE", "painting the hair out" if erase else "writing the motion reference")
    ref_path = os.path.join(work, "ref.mp4")
    out = Writer(ref_path, size, lossless=True)
    if erase:
        hair_m = union_video(hair, size, n, dilate=51)
        if not hair:
            report["warnings"].append("no hair was found on the subject (bald or covered); nothing to paint out")
    for i, f in enumerate(frames(src)):
        if i >= n:
            break
        if erase and hair_m[i].any():
            m = (hair_m[i] & ~keep[i]).astype(np.uint8) * 255
            f = cv2.inpaint(f, m, 9, cv2.INPAINT_TELEA)
        out.write(f)
        if i % 24 == 0:
            emit("PROGRESS", 40 + int(45 * i / n))
    out.close()

    emit("PHASE", "cutting pieces")
    piece = plan["piece"]
    pieces = []
    for s in plan["starts"]:
        dest = os.path.join(work, f"ref_{s}.mp4")
        cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", ref_path]
        if plan["audio"]:
            cmd += ["-ss", f"{s / FPS:.4f}", "-i", src, "-map", "0:v:0", "-map", "1:a?"]
        # select by frame number, so a piece is exactly `piece` frames; a short clip holds its last frame
        cmd += ["-vf", f"select='between(n,{s},{s + piece - 1})',setpts=N/{FPS}/TB,tpad=stop_mode=clone:stop={piece}",
                "-frames:v", str(piece), "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
        if plan["audio"]:
            cmd += ["-c:a", "aac", "-t", f"{piece / FPS:.4f}"]
        subprocess.run(cmd + [dest], check=True)
        pieces.append({"start": s, "path": dest})
    report["pieces"] = pieces
    report["frames"] = n
    json.dump(report, open(os.path.join(work, "build.json"), "w"), indent=1)
    emit("PROGRESS", 100)
    log(f"build: {len(pieces)} piece(s), {len(refs)} prop picture(s), hair {'painted out' if erase else 'kept'}")


# ---------------------------------------------------------------- score

def motion_energy(path, size, start=0, count=None):
    """Mean absolute frame difference per frame on a small grey copy."""
    sw, sh = max(16, size[0] // 4), max(16, size[1] // 4)
    prev, out = None, []
    for i, f in enumerate(frames(path)):
        if i < start:
            continue
        if count is not None and i >= start + count:
            break
        g = cv2.cvtColor(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return np.array(out)


def windowed_corr(a, b):
    n = min(len(a), len(b))
    vals = []
    for s in range(1, max(2, n - MOTION_WINDOW + 1), MOTION_STEP):
        x, y = a[s:s + MOTION_WINDOW], b[s:s + MOTION_WINDOW]
        if len(x) < 12 or x.std() < 1e-6 or y.std() < 1e-6:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return vals


def score(spec):
    """spec: work_dir, renders [{start, path}] -> score.json [{start, path, mean, min, windows}]"""
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    size, piece = tuple(plan["canvas"]), plan["piece"]
    src = os.path.join(work, "src.mp4")
    out = []
    for r in spec["renders"]:
        want = motion_energy(src, size, r["start"], min(piece, plan["frames"] - r["start"]))
        got = motion_energy(r["path"], size, 0, len(want))
        w = windowed_corr(want, got)
        out.append({**r, "mean": round(float(np.mean(w)), 3) if w else 0.0,
                    "min": round(float(np.min(w)), 3) if w else 0.0, "windows": [round(v, 3) for v in w]})
    json.dump(out, open(spec["out"], "w"), indent=1)


# ---------------------------------------------------------------- finish
# Ported from the accepted spike (spikes/faceswap_h3/fsh3.py, 127d95f): _chin_clean,
# _stitch and _destubble, with the same constants, made to stream.

def E(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))


def mouth_of(lab, inside, red_above):
    """Lips/mouth: redder than the face (LAB a* above red_above) inside `inside`, closed,
    only blobs over 80 px."""
    lips = cv2.morphologyEx(((lab[..., 1] > red_above) & inside).astype(np.uint8), cv2.MORPH_CLOSE, E(9))
    cnts, _ = cv2.findContours(lips, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mouth = np.zeros(inside.shape, np.uint8)
    big = [c for c in cnts if cv2.contourArea(c) > 80]
    if big:
        cv2.drawContours(mouth, big, -1, 1, -1)
    return mouth


def chin_clean(f, fm):
    """Small dark specks in the chin zone (bottom 22% of the face, lips/mouth excluded) ->
    Telea fill. The target's moles leak through H3; this takes them out of the new face."""
    ys, xs = np.where(fm)
    if len(ys) < 500:
        return f
    H, W = fm.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    zone = np.zeros((H, W), bool)
    zone[int(y0 + .78 * h):y1 + 1, int(x0 + .15 * w):int(x1 - .15 * w) + 1] = True
    zone &= cv2.erode(fm.astype(np.uint8), E(5)) > 0
    lab = _lab(f)
    mouth = mouth_of(lab, fm, np.median(lab[..., 1][fm]) + 7)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    d = g.astype(np.float32) - cv2.medianBlur(g, 15).astype(np.float32)
    k, lbl, st, _ = cv2.connectedComponentsWithStats(
        ((d < -15) & zone & ~(cv2.dilate(mouth, E(9)) > 0)).astype(np.uint8))
    m = np.zeros((H, W), np.uint8)
    for j in range(1, k):
        if 3 <= st[j, cv2.CC_STAT_AREA] <= 60:
            m[lbl == j] = 1
    return cv2.inpaint(f, cv2.dilate(m, E(5)) * 255, 5, cv2.INPAINT_TELEA) if m.any() else f


def _lab(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)


def _apply(f, x):
    """A piece's colour correction: its overlap's LAB mean/std mapped onto the previous one's."""
    if x is None:
        return f
    ms, ss, md, sd = x
    return cv2.cvtColor(np.clip((_lab(f) - ms) / ss * sd + md, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def stitch(picks, n, piece, size, dest, chin=True, blend=XFADE):
    """LAB chain colour match on the overlaps + a crossfade at the END of each overlap
    (capped at its length), written as it goes. Each piece is loaded once (chin-cleaned);
    only the previous piece is held, for the match and the blend. Returns (joins, the
    subject's face mask per timeline frame: later pieces win in overlaps, as the
    crossfade ends on them)."""
    out = Writer(dest, size, lossless=True)
    face_t = [None] * n
    joins = []
    prev, written = None, 0      # prev = (start, frames, colour correction)
    for k, p in enumerate(picks):
        s, count = p["start"], min(piece, n - p["start"])
        fm = list(mask_frames(p["face_mask"], size)) if p.get("face_mask") else []
        fr, missing = [], 0
        for i, f in enumerate(frames(p["path"])):
            if i >= count:
                break
            if (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
            m = fm[min(i, len(fm) - 1)] if fm else None
            if m is not None and m.sum() >= 500:
                if chin:
                    f = chin_clean(f, m)
            else:
                missing += 1
            face_t[s + i] = m
            fr.append(f)
        if not fr:
            raise RuntimeError(f"the render for the piece at frame {s} has no frames")
        if len(fr) < count:  # a short render: hold its last frame
            fr += [fr[-1]] * (count - len(fr))
        p["face_missing_frames"] = missing

        xf = None
        if prev is not None:
            a0, fa, xa = prev
            a_end = a0 + len(fa)
            ov = range(s, a_end)
            A = np.stack([_lab(_apply(fa[t - a0], xa)) for t in ov]).reshape(-1, 3)
            B = np.stack([_lab(fr[t - s]) for t in ov]).reshape(-1, 3)
            xf = (B.mean(0), B.std(0) + 1e-3, A.mean(0), A.std(0) + 1e-3)
            bl = min(blend, a_end - s)
            for t in range(written, a_end - bl):
                out.write(_apply(fa[t - a0], xa))
            diffs = []
            for t in range(a_end - bl, a_end):
                a_, b_ = _apply(fa[t - a0], xa), _apply(fr[t - s], xf)
                w = (t - (a_end - bl) + 1) / (bl + 1)
                diffs.append(float(np.abs(a_.astype(np.int16) - b_).mean()))
                out.write(cv2.addWeighted(a_, 1 - w, b_, w, 0))
            joins.append({"at_frame": a_end - bl, "crossfade": bl, "mismatch": round(float(np.mean(diffs)), 2)})
            written = a_end
        prev = (s, fr, xf)
        emit("PROGRESS", int(70 * (k + 1) / len(picks)))
    a0, fa, xa = prev
    for t in range(written, a0 + len(fa)):
        out.write(_apply(fa[t - a0], xa))
    out.close()
    if out.n != n:
        log(f"warning: stitched {out.n} frames for a {n}-frame clip")
    return joins, face_t


def destubble(src, face_t, size, dest, k_l=0.75):
    """Chin/jaw stubble shadow: ADD a smooth L/a/b offset toward a temporally averaged
    cheek tone + a 5 px luminance closing for dots. Never replaces pixels with a blurred
    copy and never uses a per-frame reference colour (both made Shawn's rejected v3
    blurry and flickery). Zone = face mask averaged over +-3 frames, ramped in from 66-80%
    of the face height, lips/mouth excluded. Off by default: it would lighten a real beard.
    Streams with a 15-frame window (the cheek tone averages +-7 frames)."""
    W, H = size
    n = len(face_t)
    empty = np.zeros((H, W), bool)
    face = [m if m is not None else empty for m in face_t]
    boxes = []
    for fm in face:
        ys, xs = np.where(fm)
        boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], float) if len(ys) > 500 else None)

    def sbox(i, r=4):
        bs = [boxes[j] for j in range(max(0, i - r), min(n, i + r + 1)) if boxes[j] is not None]
        return np.median(bs, 0) if bs else None

    def smask(i, r=3):
        return np.mean([face[j].astype(np.float32) for j in range(max(0, i - r), min(n, i + r + 1))], 0)

    yy = np.arange(H, dtype=np.float32)[:, None]
    labs, refs = {}, {}
    reader = frames(src)
    loaded = 0

    def load_through(last):
        """Read frames in order up to `last`, keeping each one's LAB and cheek tone."""
        nonlocal loaded
        while loaded <= last:
            f = next(reader, None)
            if f is None:
                return
            j = loaded
            lab = _lab(f)
            labs[j] = lab
            b = sbox(j)
            ref = None
            if b is not None:
                x0, y0, x1, y1 = b
                h, w = y1 - y0, x1 - x0
                ch = np.zeros((H, W), bool)
                ch[int(y0 + .45 * h):int(y0 + .62 * h), int(x0 + .12 * w):int(x0 + .33 * w)] = True
                ch[int(y0 + .45 * h):int(y0 + .62 * h), int(x1 - .33 * w):int(x1 - .12 * w)] = True
                ch &= face[j]
                ref = np.median(lab[ch], 0) if ch.sum() > 50 else None
            refs[j] = ref
            loaded += 1

    out = Writer(dest, size, lossless=True)
    for i in range(n):
        load_through(min(n - 1, i + 7))
        if i not in labs:
            break
        lab = labs[i]
        b = sbox(i)
        rs = [refs[j] for j in range(max(0, i - 7), min(n, i + 8)) if refs.get(j) is not None]
        if b is None or not rs:
            res = lab
        else:
            ref = np.mean(rs, 0)
            x0, y0, x1, y1 = b
            h = y1 - y0
            inbox = np.zeros((H, W), bool)
            inbox[int(y0):int(y1), int(x0):int(x1)] = True
            mouth = mouth_of(lab, inbox, ref[1] + 7)
            zone = cv2.erode(smask(i), E(7)) * np.clip((yy - (y0 + .66 * h)) / (.14 * h), 0, 1)
            zone[cv2.dilate(mouth, E(13)) > 0] = 0
            zone = cv2.GaussianBlur(zone, (0, 0), 6)
            low = cv2.GaussianBlur(lab, (0, 0), 6)
            res = lab.copy()
            closed = cv2.morphologyEx(lab[..., 0].astype(np.uint8), cv2.MORPH_CLOSE, E(5)).astype(np.float32)
            res[..., 0] += zone * 0.7 * (closed - lab[..., 0])
            res[..., 0] += zone * k_l * np.clip(ref[0] - low[..., 0], 0, 28)
            res[..., 1] += zone * 0.6 * np.clip(ref[1] - low[..., 1], -8, 8)
            res[..., 2] += zone * 0.6 * np.clip(ref[2] - low[..., 2], -8, 8)
        out.write(cv2.cvtColor(np.clip(res, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
        labs.pop(i - 7, None)
        if i % 48 == 0:
            emit("PROGRESS", 70 + int(15 * i / n))
    out.close()


def finish(spec):
    """spec: work_dir, picks [{start, path, face_mask, seed, tries, mean, min}], chin_clean,
    destubble, out, proof, report, warnings"""
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    size, n, piece = tuple(plan["canvas"]), plan["frames"], plan["piece"]
    picks = sorted(spec["picks"], key=lambda p: p["start"])

    emit("PHASE", "stitching")
    stitched = os.path.join(work, "stitched.mp4")
    joins, face_t = stitch(picks, n, piece, size, stitched, chin=spec.get("chin_clean", True))
    if spec.get("destubble"):
        emit("PHASE", "lifting the stubble shadow")
        clean = os.path.join(work, "destubbled.mp4")
        destubble(stitched, face_t, size, clean)
        stitched = clean

    emit("PHASE", "putting the sound back")
    src = os.path.join(work, "src.mp4")
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", stitched]
    if plan["audio"]:
        cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-t", f"{n / FPS:.4f}",
            spec["out"]]
    subprocess.run(cmd, check=True)

    emit("PHASE", "sync check")
    sync = windowed_corr(motion_energy(src, size), motion_energy(spec["out"], size))
    proof_sheet(spec["proof"], src, spec["out"], size, n)
    report = {"frames": n, "canvas": list(size),
              "pieces": [{k: p.get(k) for k in ("start", "seed", "tries", "mean", "min", "face_missing_frames")}
                         for p in picks],
              "joins": joins, "sync_mean": round(float(np.mean(sync)), 3) if sync else None,
              "sync_min": round(float(np.min(sync)), 3) if sync else None,
              "warnings": list(spec.get("warnings") or [])}
    if sync and min(sync) < 0.3:
        bad = [round((1 + i * MOTION_STEP) / FPS, 1) for i, v in enumerate(sync) if v < 0.3]
        report["warnings"].append(f"the motion drifts from the video around {bad} s; check those moments")
    with open(spec["report"], "w") as f:
        json.dump(report, f, indent=1)
    emit("PROGRESS", 100)
    log(f"finish: {n} frames, sync mean {report['sync_mean']} min {report['sync_min']}, joins "
        f"{[j['mismatch'] for j in joins]}")


def proof_sheet(path, src, out, size, n, cols=6):
    """Original over result at six moments."""
    at = [int(i * (n - 1) / (cols - 1)) for i in range(cols)]
    tw = 180
    th = int(tw * size[1] / size[0])

    def grab(p):
        got = {}
        for i, f in enumerate(frames(p)):
            if i in at:
                got[i] = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
            if i > at[-1]:
                break
        return [got.get(i, np.zeros((th, tw, 3), np.uint8)) for i in at]

    rows = []
    for label, p in (("video", src), ("swapped", out)):
        band = np.hstack(grab(p))
        head = np.full((22, band.shape[1], 3), 30, np.uint8)
        cv2.putText(head, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([head, band]))
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    cmds = {"prep": prep, "build": build, "score": score, "finish": finish}
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit("usage: face_swap.py prep|build|score|finish <spec.json>")
    cmds[sys.argv[1]](json.load(open(sys.argv[2])))
