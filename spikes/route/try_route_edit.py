"""Test driver: change the path a car takes in an EXISTING clip (an edit, not a new video),
with MiniMax H3 on the render PC's ComfyUI through the farm's python engine.

  clip -> SAM 3.1 tracks the car (its old path, frame by frame)
  -> a clean background video: the car taken out of every frame (masked temporal median:
     locked-off camera only for now)
  -> the car (cut from frame 0) dragged along the NEW route over that background, turned to
     face its direction of travel = the cut-and-drag reference
  -> edit region per frame = where the car was  +  where it will be
  -> H3 renders, then the source is pasted back outside the region (pixel-exact elsewhere)

Render modes:
  "inpaint"  H3 masked inpaint (vlo's h3 inpaint chain) whose masked area STARTS from the
             reference, sampled from step k of the schedule (seeded like Time-to-Move, but the
             rest of the clip is held by the inpaint mask the whole way)
  "ttm"      vlo's Time-to-Move workflow as-is (whole clip seeded from the reference, the
             car held for the opening steps, everything regenerated) + paste-back

params.args = [<job json string>, <out dir>]. Job JSON:
  {"clip": <signed url>, "noun": "red car", "at": [x, y] (0-1, the car in frame 0),
   "route": [[x, y], ...] (0-1; the first point is replaced by the car's centre), "hold": 6,
   "end": <arrival frame>, "ease": "inout", "prompt": "...",
   "renders": [{"name", "mode": "inpaint"|"ttm", "k": 1, "ttm": [1, 3], "steps": 8,
                "family": "ref2va"|"fl2va", "seed"}]}
Writes <out>/{reference.mp4, region.mp4, plate.png, route.png, <name>_raw.mp4, <name>.mp4}.
"""
import json
import math
import os
import subprocess
import sys
import time
import uuid

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "worker"))
sys.path.insert(0, HERE)
from videogen import graphs_inpaint, graphs_sam3  # noqa: E402
import try_route as tr  # noqa: E402

COMFY = tr.COMFY
T = tr.T
log = tr.log


def run_outputs(graph, timeout_s=1800):
    """Queue a graph and return its /history outputs dict."""
    r = tr.retry(lambda: httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T))
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:3000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            e = httpx.get(COMFY + f"/history/{pid}", timeout=60).json().get(pid)
        except httpx.TimeoutException:
            continue
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}")
            return e.get("outputs") or {}
        time.sleep(2)
    raise RuntimeError("timed out")


def fetch(f, dest):
    params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
    with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in resp.iter_bytes(1 << 20):
                out.write(chunk)


def read_frames(path):
    import cv2
    c = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(f)
    return out


def write_video(path, frames, fps=24, lossless=False, audio_from=None):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-bf", "0"] + (["-qp", "0"] if lossless else ["-crf", "12"]) + ["-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")


def sam_tracks(src_path, noun, out, prefix):
    """Every SAM 3.1 track of `noun` in a clip: a list of per-frame bool mask lists."""
    name = tr.upload(src_path)
    outs = run_outputs(graphs_sam3.build(name, noun, prefix, objects=4))
    tracks = []
    for node_out in outs.values():
        for f in node_out.get("videos", []) or node_out.get("images", []) or []:
            if f.get("type") != "output" or "_obj" not in f["filename"]:
                continue
            dest = os.path.join(out, "_" + os.path.basename(f["filename"]))
            fetch(f, dest)
            ms = [fr[..., 2] > 127 for fr in read_frames(dest)]
            os.remove(dest)
            if ms and any(m.any() for m in ms):
                tracks.append(ms)
    return tracks


def rendered_car(raw_path, noun, new, n, out):
    """Where the car actually is in H3's render: the SAM track that sits on the drawn path
    most. H3 runs a little off and behind the drawn path, so pasting back by the drawn path
    clipped the car."""
    import cv2
    import numpy as np
    k = np.ones((81, 81), np.uint8)
    near = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in new]
    best = None
    for ms in sam_tracks(raw_path, noun, out, "route_edit/samraw"):
        ms = ms[:n] + [np.zeros_like(ms[0])] * max(0, n - len(ms))
        score = sum(int((m & near[f]).sum()) for f, m in enumerate(ms))
        if best is None or score > best[0]:
            best = (score, ms)
    if best is None or best[0] == 0:
        raise RuntimeError("SAM found no car on the drawn path in the render")
    return best[1]


def track_car(src_path, noun, at, n, w, h, out):
    """Per-frame bool masks of the car under `at` in frame 0 (SAM 3.1 video tracking)."""
    import numpy as np
    best = None
    for ms in sam_tracks(src_path, noun, out, "route_edit/sam"):
        if not ms[0].any():
            continue
        ys, xs = np.nonzero(ms[0])
        d = (xs.mean() / w - at[0]) ** 2 + (ys.mean() / h - at[1]) ** 2
        hit = ms[0][min(int(at[1] * h), h - 1), min(int(at[0] * w), w - 1)]
        score = (0 if hit else 1, d)
        if best is None or score < best[0]:
            best = (score, ms)
    if best is None:
        raise RuntimeError(f"SAM found no {noun!r} in frame 0")
    ms = best[1][:n] + [best[1][-1]] * max(0, n - len(best[1]))
    return [m.copy() for m in ms]


def clean_plate(frames, old, grow=15):
    """Locked-off camera: per-pixel median over the frames where the car is not."""
    import cv2
    import numpy as np
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    cover = np.stack([cv2.dilate(m.astype(np.uint8), k) > 0 for m in old])  # n,h,w
    stack = np.stack(frames)  # n,h,w,3
    n, h, w, _ = stack.shape
    plate = np.zeros((h, w, 3), np.uint8)
    for y0 in range(0, h, 48):
        blk = stack[:, y0:y0 + 48].astype(np.float32)
        cv = cover[:, y0:y0 + 48]
        blk[cv] = np.nan
        med = np.nanmedian(blk, axis=0)
        fallback = np.median(stack[:, y0:y0 + 48], axis=0)
        med = np.where(np.isnan(med), fallback, med)
        plate[y0:y0 + 48] = np.clip(med, 0, 255).astype(np.uint8)
    return plate, cover


def build_edit_reference(frames, old, cover, plate, route_n, n, hold, end, ease, out):
    """The source with the car lifted out and redrawn along the new route."""
    import cv2
    import numpy as np
    h, w = frames[0].shape[:2]
    m0 = old[0]
    ang, (cx, cy) = tr.car_axis(m0)
    route = [(cx, cy)] + [(x * w, y * h) for x, y in route_n[1:]]
    ps = tr.poses(route, n, hold, ease, end)
    h0 = ps[min(n - 1, hold + 2)][2]
    # the car's nose: its long axis, the way the car was going in the source (first frames of
    # its old track), not the way the new route goes
    ys, xs = np.nonzero(old[min(n - 1, 12)])
    if len(xs) and math.hypot(xs.mean() - cx, ys.mean() - cy) > 3:
        h_src = math.atan2(ys.mean() - cy, xs.mean() - cx)
    else:
        h_src = h0
    if math.cos(ang - h_src) < 0:
        ang += math.pi
    alpha = cv2.GaussianBlur(m0.astype(np.float32), (0, 0), 1.2)
    car = frames[0].astype(np.float32)
    feather = [cv2.GaussianBlur(c.astype(np.float32), (0, 0), 3)[..., None] for c in cover]
    refs, news = [], []
    for f in range(n):
        bg = frames[f].astype(np.float32) * (1 - feather[f]) + plate.astype(np.float32) * feather[f]
        x, y, hd = ps[f]
        ramp = min(1.0, max(f - hold, 0) / 8.0)
        dth = math.atan2(math.sin(hd - ang), math.cos(hd - ang)) * ramp
        M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(dth), 1.0)
        M[0, 2] += x - cx
        M[1, 2] += y - cy
        c = cv2.warpAffine(car, M, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR)[..., None]
        refs.append(np.clip(bg * (1 - a) + c * a, 0, 255).astype(np.uint8))
        news.append(a[..., 0] > 0.5)
    pic = frames[0].copy()
    for f in range(0, n, 3):
        cv2.drawContours(pic, cv2.findContours(old[f].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 200, 255), 1)
    for (x0, y0, _), (x1, y1, _) in zip(ps, ps[1:]):
        cv2.line(pic, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, "route.png"), pic)
    return refs, news, ps


def regions(old, new, n, g_old=18, g_new=30, t_pad=2):
    """Edit region per frame: where the car was + where it goes, widened in space and over
    a few neighbouring frames (H3 masks whole 4-frame latent groups anyway)."""
    import cv2
    import numpy as np
    ko = np.ones((2 * g_old + 1, 2 * g_old + 1), np.uint8)
    kn = np.ones((2 * g_new + 1, 2 * g_new + 1), np.uint8)
    base = [(cv2.dilate(old[f].astype(np.uint8), ko) | cv2.dilate(new[f].astype(np.uint8), kn)) > 0 for f in range(n)]
    out = []
    for f in range(n):
        u = np.zeros_like(base[0])
        for j in range(max(0, f - t_pad), min(n, f + t_pad + 1)):
            u |= base[j]
        out.append(u)
    return out


def matched_plate(frame, plate, cover, sigma=25):
    """The clean plate brought to this frame's grade: a smooth correction field measured on
    the static pixels around (frame - plate, where the two agree and no car is). H3 footage
    drifts in brightness over a clip, so the plain plate showed as a dark car-shaped patch.
    (A Poisson paste smeared the old car's shadow and edge pixels in from the mask border.)"""
    import cv2
    import numpy as np
    p = plate.astype(np.float32)
    d = frame.astype(np.float32) - p
    valid = ((np.abs(d).max(2) < 20) & ~cover).astype(np.float32)
    num = cv2.GaussianBlur(d * valid[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(valid, (0, 0), sigma)[..., None]
    return p + num / np.maximum(den, 1e-3)


def _soft(m, grow, sigma):
    import cv2
    import numpy as np
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    return cv2.GaussianBlur((cv2.dilate(m.astype(np.uint8), k) > 0).astype(np.float32), (0, 0), sigma)[..., None]


def compose(frames, old, car, plate, res, old_grow=48, car_grow=8, cover_grow=60):
    """Only the car comes from H3.

    The old car: H3 cars darken the road round them out to ~40-50 px (soft shadow, -15
    levels at the body, -3 at 25 px), so taking out only the body left a "visibly invisible
    car" - its shadow halo driving down the road. So the graded clean plate (built with the
    same wide exclusion, so it is shadow-free) goes over the body + 48 px, feathered wide;
    anything else passing close (much brighter or darker than the road: the other car) keeps
    its own pixels.

    The new car is pasted where SAM finds it in the render, not where it was drawn (H3 runs
    off and behind the drawn path, so pasting by the drawn path clipped it)."""
    import cv2
    import numpy as np
    n = len(frames)
    pf = plate.astype(np.float32)
    kc = np.ones((2 * cover_grow + 1,) * 2, np.uint8)
    ko = np.ones((2 * old_grow + 1,) * 2, np.uint8)
    sc = old_grow / 48.0  # the fixed sizes below were tuned with old_grow 48 on 832-wide footage
    kt = np.ones((2 * max(1, round(6 * sc)) + 1,) * 2, np.uint8)
    ke = np.ones((2 * max(1, round(4 * sc)) + 1,) * 2, np.uint8)
    out = []
    for f in range(n):
        base = frames[f].astype(np.float32)
        if old[f].any():
            m = old[f].astype(np.uint8)
            fill = matched_plate(frames[f], plate, cv2.dilate(m, kc) > 0, sigma=25 * sc)
            reg = cv2.dilate(m, ko) > 0
            tight = cv2.dilate(m, kt) > 0
            other = (np.abs(base - pf).max(2) > 45) & ~tight
            other = cv2.dilate(other.astype(np.uint8), ke) > 0
            a = cv2.GaussianBlur((reg & ~other).astype(np.float32), (0, 0), 10 * sc)[..., None]
            a = np.maximum(a, cv2.GaussianBlur(tight.astype(np.float32), (0, 0), 2 * sc)[..., None])
            base = base * (1 - a) + fill * a
        u = np.zeros_like(car[0])
        for j in range(max(0, f - 1), min(n, f + 2)):
            u |= car[j]
        ac = _soft(u, car_grow, 2.5 * sc)
        out.append(np.clip(base * (1 - ac) + res[f].astype(np.float32) * ac, 0, 255).astype(np.uint8))
    return out


def inpaint_graph(ref_name, vmask_name, amask_name, prompt, w, h, n, seed, prefix, steps, k, family, first_name):
    g, _ = graphs_inpaint.build(ref_name, vmask_name, amask_name, prompt, w, h, n, seed, prefix,
                                steps=steps, turbo=True, family=family,
                                first_frame=first_name if family == "fl2va" else None)
    # keep the reference in the masked area (no "cleared" blank) and start the schedule at
    # step k, so the car starts from the dragged cut-out rather than from nothing
    g["feather"]["inputs"]["audio_latent"] = ["amasked", 0]
    del g["blank"], g["cleared"]
    if k > 0:
        g["split"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sched", 0], "step": int(k)}}
        g["sample"]["inputs"]["sigmas"] = ["split", 1]
    return g


def main():
    import cv2
    import numpy as np
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    tr.clear_out(out)
    src = os.path.join(out, "source.mp4")
    with httpx.stream("GET", job["clip"], timeout=T, follow_redirects=True) as r:
        r.raise_for_status()
        with open(src, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    frames = read_frames(src)
    h, w = frames[0].shape[:2]
    n = len(frames)
    n = ((n - 5) // 17) * 17 + 5
    frames = frames[:n]
    if w % 32 or h % 32:
        raise RuntimeError(f"spike wants a 32-aligned clip, got {w}x{h}")
    log(f"source {w}x{h}, {n} frames")
    tr.ensure_comfy()
    try:
        t0 = time.monotonic()
        old = track_car(src, job.get("noun", "car"), job["at"], n, w, h, out)
        log(f"sam track: {time.monotonic() - t0:.0f}s, car in {sum(m.any() for m in old)}/{n} frames")
        tr.free()
        # pixel sizes below were tuned on 832-wide footage; shadows and cars scale with the frame
        px = lambda v: max(1, int(round(v * w / 832)))  # noqa: E731
        plate, cover = clean_plate(frames, old, grow=px(60))
        cv2.imwrite(os.path.join(out, "plate.png"), plate)
        refs, new, _ = build_edit_reference(frames, old, cover, plate, job["route"], n, int(job.get("hold", 6)),
                                            job.get("end"), job.get("ease", "inout"), out)
        reg = regions(old, new, n, px(int(job.get("grow_old", 18))), px(int(job.get("grow_new", 30))))
        write_video(os.path.join(out, "reference.mp4"), refs, audio_from=src)
        write_video(os.path.join(out, "region.mp4"), [cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for m in reg], lossless=True)
        write_video(os.path.join(out, "amask.mp4"), [np.zeros((h, w, 3), np.uint8)] * n, lossless=True)
        k21 = np.ones((2 * px(10) + 1,) * 2, np.uint8)
        write_video(os.path.join(out, "refmask.mp4"),
                    [cv2.cvtColor(cv2.dilate(m.astype(np.uint8) * 255, k21), cv2.COLOR_GRAY2BGR) for m in new], lossless=True)
        cv2.imwrite(os.path.join(out, "first.png"), frames[0])
        names = {f: tr.upload(os.path.join(out, f)) for f in ("reference.mp4", "region.mp4", "amask.mp4", "refmask.mp4", "first.png")}
        # paste-back mask: the region, a little wider in space and time, feathered
        comp = regions(old, new, n, px(int(job.get("grow_old", 18)) + 6), px(int(job.get("grow_new", 30)) + 6), t_pad=3)
        comp = [cv2.GaussianBlur(m.astype(np.float32), (0, 0), 5)[..., None] for m in comp]
        old_grow, car_grow = px(int(job.get("old_grow", 48))), px(int(job.get("car_grow", 8)))
        write_video(os.path.join(out, "oldmask.mp4"), [cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for m in old], lossless=True)
        for rd in job["renders"]:
            raw = os.path.join(out, f"{rd['name']}_raw.mp4")
            seed = int(rd.get("seed", 6332))
            steps = int(rd.get("steps", 8))
            if rd["mode"] == "inpaint":
                g = inpaint_graph(names["reference.mp4"], names["region.mp4"], names["amask.mp4"], job["prompt"], w, h, n,
                                  seed, f"route_edit/{rd['name']}", steps, int(rd.get("k", 1)), rd.get("family", "ref2va"),
                                  names["first.png"])
            else:
                g = tr.ttm_graph(names["reference.mp4"], names["refmask.mp4"], names["first.png"], job["prompt"], w, h, n,
                                 seed, f"route_edit/{rd['name']}", steps, True, rd.get("ttm", [1, 3]))
            t = tr.run(g, raw, ".mp4")
            res = read_frames(raw)[:n]
            res += [res[-1]] * (n - len(res))
            res = [cv2.resize(r_, (w, h)) if r_.shape[:2] != (h, w) else r_ for r_ in res]
            final = [np.clip(frames[f] * (1 - comp[f]) + res[f] * comp[f], 0, 255).astype(np.uint8) for f in range(n)]
            write_video(os.path.join(out, f"{rd['name']}_v1.mp4"), final, audio_from=src)
            log(f"render {rd['name']}: {t:.0f}s")
            car = rendered_car(raw, job.get("noun", "car"), new, n, out)
            tr.free()
            final = compose(frames, old, car, plate, res, old_grow, car_grow, cover_grow=px(60))
            write_video(os.path.join(out, f"{rd['name']}.mp4"), final, audio_from=src)
            write_video(os.path.join(out, f"{rd['name']}_carmask.mp4"),
                        [cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for m in car], lossless=True)
    finally:
        tr.free()


if __name__ == "__main__":
    main()
