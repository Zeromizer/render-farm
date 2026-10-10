"""Lettering fix, pixel side: correct number plates and badges on a generated clip by
letting MiniMax H3 propagate the REAL lettering from a reference photo.

Why this and not a corner pin (planar_patch): a pasted artwork never sits in the
footage the way the model's own pixels do (perspective drift through a turn, sharp
letters on motion blur). H3 cannot spell from a prompt, but it holds a video
consistent: whatever lettering is visible on the plate in some frames, it carries
into the masked ones. So the real lettering is pasted where the car's pose matches
the reference (the "anchor" frames, left unmasked), the element is masked
everywhere else, and H3 redraws it there with its own perspective, blur and light.
Proven on an ATTO 1 take 2026-10-10: 5/5 seeds read "BYD ATTO 1" through the
turn-in; text in the prompt alone gave the wrong font or garbled glyphs.

Two subcommands, both driven by a spec JSON the runner writes:

  prep     align the reference to the anchor frame, track every element, paste the
           real lettering on the anchor run, cut a follow-crop window (the elements
           stay large and nearly still on the generation canvas), write the H3
           inputs (source / video mask / audio mask) and plan.json
  compose  composite each H3 take back over the anchored clip, score the takes
           against the reference lettering, write the best one, every take, a proof
           sheet and a report

Run in the planar_patch venv (opencv-python-headless + numpy); ffmpeg from PATH or
the worker's usual lookup.
"""
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract"))
from common import find_ffmpeg_tool  # noqa: E402

FPS = 24
MAX_WINDOW = 362          # H3's trained range tops out near 15 s at 24 fps
CANVAS_AREA = 960 * 544   # follow-crop canvas: ~0.52 MP, plenty for a 16 GB card at 15 s


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


# ---------------------------------------------------------------- io

def read_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode any frame from {path}")
    return frames, fps


def write_clip(path, frames, fps, audio_from=None, audio_start=0.0, lossless=False):
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-"]
    if audio_from:
        cmd += ["-ss", f"{audio_start:.4f}", "-i", audio_from, "-map", "0:v:0", "-map", "1:a?"]
    if lossless:
        cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if audio_from:
        # -t, not -shortest: a source whose audio ends a hair early must not cost the
        # clip its last frame (a 158-frame take came back as 157)
        cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{len(frames) / fps:.4f}"]
    cmd += [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def has_audio(path):
    ffprobe = find_ffmpeg_tool("ffprobe")
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(r.stdout.strip())


def silent_wav(path, seconds):
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=stereo", "-t", f"{seconds:.4f}", path], check=True)


# ---------------------------------------------------------------- geometry

def snap_len(n):
    """Largest 17k+5 <= n (H3's frame grid)."""
    return max(5, n - ((n - 5) % 17))


def gen_dims(aspect, area=CANVAS_AREA):
    w = math.sqrt(area * aspect)
    return max(256, int(round(w / 32)) * 32), max(256, int(round(w / aspect / 32)) * 32)


def window_for(n_frames, anchor):
    """(start, length): the largest 17k+5 window inside the clip that contains the anchor,
    preferring to keep the anchor at the end it is nearest to."""
    length = min(snap_len(n_frames), MAX_WINDOW)
    if anchor >= n_frames - length:
        return n_frames - length, length
    if anchor < length:
        return 0, length
    return anchor - length + 1, length


def box_of(quad):
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def clamp_box(b, w, h):
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]


# ---------------------------------------------------------------- reference alignment

def align_reference(ref, frame, boxes_ref, log):
    """Homography reference -> frame. AKAZE features + RANSAC for the gross fit (the
    reference may be another size, crop or aspect), then ECC refined on the elements'
    neighbourhood. Returns (H, ecc_score, inliers)."""
    fh, fw = frame.shape[:2]
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # AKAZE in OpenCV 4 (the worker venv pins <5); OpenCV 5 moved it out of the main
    # namespace, so fall back to ORB there (both binary descriptors -> Hamming).
    make = getattr(cv2, "AKAZE_create", None) or (lambda: cv2.ORB_create(4000))
    det = make()
    k1, d1 = det.detectAndCompute(rg, None)
    k2, d2 = det.detectAndCompute(fg, None)
    H, inliers = None, 0
    if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.8 * n.distance]
        if len(good) >= 8:
            src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 10:
        # same framing (the usual i2v case: the reference IS the last frame): plain resize
        log(f"feature fit weak ({inliers} inliers); assuming the reference is the frame resized")
        H = np.array([[fw / ref.shape[1], 0, 0], [0, fh / ref.shape[0], 0], [0, 0, 1]], np.float64)
    # ECC refine on the union of the elements (+ margin) in FRAME coords
    quads = [cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), H)
             for b in boxes_ref]
    u = box_of(np.concatenate([q.reshape(-1, 2) for q in quads]))
    m = max(40.0, 0.6 * max(u[2] - u[0], u[3] - u[1]))
    u = clamp_box([u[0] - m, u[1] - m, u[2] + m, u[3] + m], fw, fh)
    warped = cv2.warpPerspective(rg, H, (fw, fh), flags=cv2.INTER_LINEAR)
    tmask = np.zeros((fh, fw), np.uint8)
    tmask[int(u[1]):int(u[3]), int(u[0]):int(u[2])] = 255
    try:
        cc, W = cv2.findTransformECC(fg.astype(np.float32), warped.astype(np.float32), np.eye(3, dtype=np.float32),
                                     cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                                     tmask, 3)
        # template = frame, input = warped ref: warped(W x) ~ frame(x)  ->  frame = W^-1 applied to warped
        H = np.linalg.inv(W.astype(np.float64)) @ H
    except cv2.error:
        cc = -1.0
    return H / H[2, 2], float(cc), inliers


# ---------------------------------------------------------------- tracking

def track_element(grays, anchor, box, lo, hi):
    """Template-match an element box from the anchor frame outwards, updating the
    template every frame (slow pose change, motion blur). Returns {frame: (x0, y0, score)}."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    H, W = grays[0].shape
    out = {anchor: (x0, y0, 1.0)}
    for rng in (range(anchor - 1, lo - 1, -1), range(anchor + 1, hi + 1)):
        px, py = x0, y0
        tpl = grays[anchor][y0:y1, x0:x1]
        prev_dx = prev_dy = 0
        for i in rng:
            sx = max(40, 2 * w // 3)
            sy = max(30, h)
            cx, cy = px + prev_dx, py + prev_dy   # constant-velocity prediction
            ax0, ay0 = max(0, cx - sx), max(0, cy - sy)
            ax1, ay1 = min(W, cx + w + sx), min(H, cy + h + sy)
            win = grays[i][ay0:ay1, ax0:ax1]
            if win.shape[0] < h or win.shape[1] < w:
                break
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            nx, ny = ax0 + loc[0], ay0 + loc[1]
            if nx <= 0 or ny <= 0 or nx + w >= W or ny + h >= H:
                break   # touching the frame edge: entering/leaving, not fully visible -> leave it alone
            out[i] = (nx, ny, float(score))
            if score < 0.35:
                break   # lost: stop rather than wander; the caller borrows a neighbour's motion
            prev_dx, prev_dy = nx - px, ny - py
            px, py = nx, ny
            tpl = grays[i][ny:ny + h, nx:nx + w]
    return out


def fill_from(track, primary, anchor, frames):
    """Frames an element's own track did not reach (or held weakly) borrow the primary
    element's displacement since the anchor."""
    ax, ay, _ = track[anchor]
    pax, pay, _ = primary[anchor]
    out = {}
    for i in frames:
        t = track.get(i)
        if t and t[2] >= 0.45:
            out[i] = (t[0], t[1])
        elif i in primary:
            out[i] = (ax + primary[i][0] - pax, ay + primary[i][1] - pay)
    return out


# ---------------------------------------------------------------- prep

def prep(spec):
    emit("PHASE", "decoding")
    frames, fps = read_clip(spec["clip"])
    n = len(frames)
    fh, fw = frames[0].shape[:2]
    ref = cv2.imread(spec["reference"], cv2.IMREAD_COLOR)
    if ref is None:
        raise RuntimeError("cannot read the reference image")
    anchor = int(spec.get("anchor_frame", -1))
    anchor = n - 1 if anchor < 0 else min(anchor, n - 1)
    elements = spec["elements"]
    log = lambda s: print(s, flush=True)
    print(f"{n} frames {fw}x{fh} @ {fps:.3f}, anchor {anchor}", flush=True)

    emit("PHASE", "aligning reference")
    boxes_ref = [e["box"] for e in elements]
    H, cc, inliers = align_reference(ref, frames[anchor], boxes_ref, log)
    print(f"MEASURED align ecc={cc:.3f} inliers={inliers}", flush=True)
    if cc < float(spec.get("min_align", 0.8)):
        raise RuntimeError(f"the reference does not line up with frame {anchor} of the clip (alignment score "
                           f"{cc:.2f} < {spec.get('min_align', 0.8)}). Pass the frame whose pose matches the "
                           f"reference as anchor_frame, or a reference of the same shot.")
    ref_warp = cv2.warpPerspective(ref, H, (fw, fh), flags=cv2.INTER_CUBIC)

    # element boxes on the anchor frame
    anchor_boxes = []
    for e in elements:
        b = e["box"]
        q = cv2.perspectiveTransform(np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2), H)
        anchor_boxes.append(clamp_box(box_of(q), fw, fh))

    start, length = window_for(n, anchor)
    lo, hi = start, start + length - 1
    emit("PHASE", "tracking elements")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    tracks = [track_element(grays, anchor, b, lo, hi) for b in anchor_boxes]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in anchor_boxes]
    primary = tracks[int(np.argmax(areas))]
    win = range(lo, hi + 1)
    pos = [fill_from(t, primary, anchor, win) for t in tracks]
    for e, t in zip(elements, tracks):
        sc = [s for _, _, s in t.values()]
        print(f"MEASURED track {e['name']} frames={len(t)} min_score={min(sc):.2f} "
              f"weak={sum(1 for s in sc if s < 0.45)}", flush=True)

    # anchor run: contiguous frames around the anchor where every element sits within
    # `settle` px of its anchor position -> the real lettering is pasted there, unmasked
    settle = float(spec.get("settle_px", 2.5))
    def settled(i):
        for k, b in enumerate(anchor_boxes):
            p = pos[k].get(i)
            if p is None or abs(p[0] - b[0]) > settle or abs(p[1] - b[1]) > settle:
                return False
        return True
    a0 = a1 = anchor
    while a0 - 1 >= lo and settled(a0 - 1):
        a0 -= 1
    while a1 + 1 <= hi and settled(a1 + 1):
        a1 += 1
    print(f"MEASURED anchor_run {a0}-{a1} ({a1 - a0 + 1} frames)", flush=True)

    # paste the real lettering on the anchor run (per element, translated by its track)
    emit("PHASE", "pasting reference lettering")
    anchored = [f.copy() for f in frames]
    pad_a = int(spec.get("paste_pad", 4))
    for k, b in enumerate(anchor_boxes):
        m = np.zeros((fh, fw), np.uint8)
        cv2.rectangle(m, (int(b[0]) - pad_a, int(b[1]) - pad_a), (int(b[2]) + pad_a, int(b[3]) + pad_a), 255, -1)
        mf = cv2.GaussianBlur(m, (0, 0), 2.5).astype(np.float32) / 255.0
        for i in range(a0, a1 + 1):
            dx, dy = pos[k][i][0] - b[0], pos[k][i][1] - b[1]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            r = cv2.warpAffine(ref_warp, M, (fw, fh))
            a = cv2.warpAffine(mf, M, (fw, fh))[..., None]
            anchored[i] = np.clip(anchored[i] * (1 - a) + r * a, 0, 255).astype(np.uint8)

    # masks: every window frame outside the anchor run where the element was located
    pad_frac = float(spec.get("mask_pad", 0.25))
    masks = {}
    for i in win:
        if a0 <= i <= a1:
            continue
        boxes = []
        for k, b in enumerate(anchor_boxes):
            p = pos[k].get(i)
            if p is None:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            pad = max(10.0, pad_frac * max(w, h) * 0.5 + 0.15 * h)
            boxes.append(clamp_box([p[0] - pad, p[1] - pad, p[0] + w + pad, p[1] + h + pad], fw, fh))
        if boxes:
            masks[i] = boxes

    # follow-crop: a fixed-size window riding the primary element, centred on the union
    u = box_of(np.array([[b[0], b[1]] for b in anchor_boxes] + [[b[2], b[3]] for b in anchor_boxes]))
    uw, uh = u[2] - u[0], u[3] - u[1]
    rw = max(2.4 * uw, 2.2 * uh * 16 / 9, 240.0)
    rh = rw * 9 / 16
    rw, rh = min(rw, fw), min(rh, fh)
    W, Hc = gen_dims(rw / rh)
    rh = rw * Hc / W
    if rh > fh:
        rh = fh
        rw = rh * W / Hc
    rw, rh = int(round(rw)), int(round(rh))
    pidx = int(np.argmax(areas))
    pb = anchor_boxes[pidx]
    off = ((u[0] + u[2]) / 2 - pb[0], (u[1] + u[3]) / 2 - pb[1])
    cen = np.array([[pos[pidx][i][0] + off[0], pos[pidx][i][1] + off[1]] if i in pos[pidx]
                    else [(u[0] + u[2]) / 2, (u[1] + u[3]) / 2] for i in win], np.float32)
    sm = 3
    padc = np.pad(cen, ((sm, sm), (0, 0)), mode="edge")
    ker = np.ones(2 * sm + 1) / (2 * sm + 1)
    cen = np.stack([np.convolve(padc[:, d], ker, mode="valid") for d in range(2)], 1)
    regs = []
    for cx, cy in cen:
        regs.append((int(round(min(max(0, cx - rw / 2), fw - rw))), int(round(min(max(0, cy - rh / 2), fh - rh))), rw, rh))
    print(f"MEASURED window {lo}-{hi} ({length} f) crop {rw}x{rh} -> canvas {W}x{Hc} "
          f"(x{W / rw:.2f}), masked frames {len(masks)}", flush=True)

    emit("PHASE", "writing generation inputs")
    work = spec["work_dir"]
    src_c, vm_c, am_c = [], [], []
    for k, i in enumerate(win):
        rx, ry, w_, h_ = regs[k]
        src_c.append(cv2.resize(anchored[i][ry:ry + h_, rx:rx + w_], (W, Hc), interpolation=cv2.INTER_AREA))
        m = np.zeros((fh, fw), np.uint8)
        for b in masks.get(i, []):
            m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
        mc = cv2.resize(m[ry:ry + h_, rx:rx + w_], (W, Hc), interpolation=cv2.INTER_NEAREST)
        vm_c.append(cv2.cvtColor(mc, cv2.COLOR_GRAY2BGR))
        am_c.append(np.zeros((Hc, W, 3), np.uint8))
    audio = spec["clip"] if has_audio(spec["clip"]) else None
    if audio is None:
        audio = os.path.join(work, "silence.wav")
        silent_wav(audio, n / fps + 1)
    write_clip(os.path.join(work, "gen_src.mp4"), src_c, FPS, audio_from=audio, audio_start=lo / fps, lossless=True)
    write_clip(os.path.join(work, "gen_mask.mp4"), vm_c, FPS, lossless=True)
    write_clip(os.path.join(work, "gen_amask.mp4"), am_c, FPS, lossless=True)
    write_clip(os.path.join(work, "anchored.mp4"), anchored, fps, lossless=True)

    plan = {"fps": fps, "frames": n, "width": fw, "height": fh, "anchor": anchor, "window": [lo, hi],
            "length": length, "canvas": [W, Hc], "regs": regs, "anchor_run": [a0, a1],
            "masks": {str(k): v for k, v in masks.items()}, "anchor_boxes": anchor_boxes,
            "tracks": [{str(i): list(p) for i, p in t.items()} for t in pos],
            "names": [e["name"] for e in elements], "align": {"ecc": cc, "inliers": inliers},
            "warnings": []}
    if a1 - a0 + 1 < 6:
        plan["warnings"].append(f"only {a1 - a0 + 1} anchor frame(s): the car never holds the reference pose for "
                                f"long, so the real lettering has little to propagate from")
    if not masks:
        raise RuntimeError("nothing to fix: the elements match the reference pose in every frame of the window")
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
    emit("PROGRESS", "100")


# ---------------------------------------------------------------- compose

def _grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def compose(spec):
    work = spec["work_dir"]
    plan = json.load(open(os.path.join(work, "plan.json")))
    anchored, fps = read_clip(os.path.join(work, "anchored.mp4"))
    fh, fw = anchored[0].shape[:2]
    lo, hi = plan["window"]
    regs = plan["regs"]
    masks = {int(k): v for k, v in plan["masks"].items()}
    feather = int(spec.get("feather", 8))
    names = plan["names"]
    anchor = plan["anchor"]
    a0, a1 = plan["anchor_run"]

    # reference appearance of each element (from the anchored anchor frame) for scoring
    refs = []
    for b in plan["anchor_boxes"]:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        refs.append(_grad(anchored[anchor][y0:y1, x0:x1]))
    tracks = [{int(k): v for k, v in t.items()} for t in plan["tracks"]]

    results = []
    for ti, take in enumerate(spec["takes"]):
        emit("PHASE", f"compositing take {ti + 1}")
        gen, _ = read_clip(take)
        out = [f.copy() for f in anchored]
        for k, i in enumerate(range(lo, hi + 1)):
            if k >= len(gen) or i not in masks:
                continue
            rx, ry, rw, rh = regs[k]
            m = np.zeros((fh, fw), np.uint8)
            for b in masks[i]:
                m[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 255
            mf = cv2.GaussianBlur(cv2.dilate(m, np.ones((feather, feather), np.uint8)), (0, 0), feather / 2.0)
            a = (mf[ry:ry + rh, rx:rx + rw].astype(np.float32) / 255.0)[..., None]
            back = cv2.resize(gen[k], (rw, rh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            reg = out[i][ry:ry + rh, rx:rx + rw].astype(np.float32)
            out[i][ry:ry + rh, rx:rx + rw] = np.clip(reg * (1 - a) + back * a, 0, 255).astype(np.uint8)
        # score: lettering similarity to the reference on masked frames (gradient NCC at
        # the tracked box, reference resized to it) minus a flicker penalty
        sims, flick = [], []
        for e, ref_g in enumerate(refs):
            b = plan["anchor_boxes"][e]
            w, h = int(round(b[2] - b[0])), int(round(b[3] - b[1]))
            prev = None
            for i in sorted(masks):
                p = tracks[e].get(i)
                if p is None:
                    continue
                x, y = int(round(p[0])), int(round(p[1]))
                if x < 0 or y < 0 or x + w > fw or y + h > fh:
                    continue
                crop = _grad(out[i][y:y + h, x:x + w])
                sims.append(_ncc(crop, ref_g))
                if prev is not None:
                    flick.append(float(np.abs(crop - prev).mean()))
                prev = crop
        score = (float(np.mean(sims)) if sims else 0.0) - 0.002 * (float(np.mean(flick)) if flick else 0.0)
        path = os.path.join(work, f"take{ti + 1}.mp4")
        write_clip(path, out, fps, audio_from=spec["clip"] if has_audio(spec["clip"]) else None)
        results.append({"take": ti + 1, "seed": spec["seeds"][ti], "score": round(score, 4),
                        "lettering_similarity": round(float(np.mean(sims)) if sims else 0.0, 4),
                        "flicker": round(float(np.mean(flick)) if flick else 0.0, 3), "path": path})
        print(f"MEASURED take {ti + 1} seed {spec['seeds'][ti]} score {score:.4f}", flush=True)
        emit("PROGRESS", str(int(100 * (ti + 1) / (len(spec["takes"]) + 1))))

    best = max(results, key=lambda r: r["score"])
    shutil.copy(best["path"], spec["out"])
    emit("PHASE", "proof sheet")
    proof_sheet(spec, plan, read_clip(spec["clip"])[0], read_clip(best["path"])[0], best, results)
    report = {"best_take": best["take"], "takes": [{k: v for k, v in r.items() if k != "path"} for r in results],
              "anchor_frame": anchor, "anchor_run": plan["anchor_run"], "window": plan["window"],
              "elements": names, "align": plan["align"], "warnings": plan["warnings"]}
    json.dump(report, open(spec["report"], "w"), indent=1)
    emit("PROGRESS", "100")


def proof_sheet(spec, plan, original, fixed, best, results):
    """Per element: original over fixed, zoomed, at six frames spread over the masked
    range plus one anchor frame; a header with the take scores."""
    masks = sorted(int(k) for k in plan["masks"])
    tracks = [{int(k): v for k, v in t.items()} for t in plan["tracks"]]
    fh, fw = original[0].shape[:2]
    tile_w = 260
    rows = []
    for e, name in enumerate(plan["names"]):
        own = [i for i in masks if i in tracks[e]] or masks
        picks = [own[int(round(t * (len(own) - 1)))] for t in np.linspace(0, 1, 6)] + [plan["anchor_run"][0]]
        b = plan["anchor_boxes"][e]
        w, h = b[2] - b[0], b[3] - b[1]
        cw, ch = max(w * 2.2, 80), max(h * 3.0, 50)
        top, bot = [], []
        for i in picks:
            p = tracks[e].get(i, (b[0], b[1]))
            cx, cy = p[0] + w / 2, p[1] + h / 2
            x0 = int(min(max(0, cx - cw / 2), fw - cw)); y0 = int(min(max(0, cy - ch / 2), fh - ch))
            sz = (tile_w, int(tile_w * ch / cw))
            o = cv2.resize(original[i][y0:y0 + int(ch), x0:x0 + int(cw)], sz, interpolation=cv2.INTER_CUBIC)
            f = cv2.resize(fixed[i][y0:y0 + int(ch), x0:x0 + int(cw)], sz, interpolation=cv2.INTER_CUBIC)
            for t, lab in ((o, f"{i} before"), (f, f"{i} fixed" if i not in range(plan['anchor_run'][0], plan['anchor_run'][1] + 1) else f"{i} anchor")):
                cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            top.append(o)
            bot.append(f)
        band = np.vstack([np.hstack(top), np.hstack(bot)])
        label = np.full((24, band.shape[1], 3), 30, np.uint8)
        cv2.putText(label, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(np.vstack([label, band]))
    width = max(r.shape[1] for r in rows)
    head = np.full((30, width, 3), 20, np.uint8)
    txt = "MEASURED takes: " + "  ".join(f"#{r['take']} seed {r['seed']} score {r['score']:.3f}" for r in results) + \
          f"   -> best #{best['take']}"
    cv2.putText(head, txt[:160], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 160), 1)
    sheet = np.vstack([head] + [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows])
    cv2.imwrite(spec["proof"], sheet)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("prep", "compose"):
        sys.exit("usage: lettering.py prep|compose <spec.json>")
    s = json.load(open(sys.argv[2]))
    prep(s) if sys.argv[1] == "prep" else compose(s)
