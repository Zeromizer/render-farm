"""Test driver: drive a car along a route in a still, with MiniMax H3 + vlo Time-to-Move,
on the render PC's ComfyUI through the farm's python engine.

  still -> SAM 3.1 "car" mask -> Krea 2 cleans the road under it (clean plate)
  -> the car cut out and moved along a smoothed route, turned to face its direction
     of travel (the "cut-and-drag" reference) + a mask of where it is
  -> H3 fl2va image-to-video from the still, vloTimeToMove holding the car to the
     reference for the opening steps, then free to render it properly

params.args = [<job json string>, <out dir>]. Job JSON:
  {"still": <signed url>, "scene": "<prompt for the clean plate: the place without the car>",
   "prompt": "<the shot>", "route": [[x, y], ...] (0-1 of the frame; the first point is
   replaced by the car's own centre), "frames": 121, "width": 832, "height": 480,
   "ease": "inout" | "linear", "hold": 6 (frames still at the start),
   "renders": [{"name", "steps", "turbo", "ttm": [start, end], "seed"}]}
Several cars: "cars": [{"at": [x, y] (0-1, a point on that car), "route": [...], "hold": 6,
   "end": <frame it arrives, default the last>, "ease"}] in place of "route"; SAM finds every
   car, each one is picked by its point, all are removed from the plate and each is dragged
   along its own route.
Writes <out>/{plate.png, mask.png, route.png, reference.mp4, refmask.mp4, <name>.mp4,
compare.mp4}.
"""
import json
import math
import os
import sys
import time
import uuid

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))
from videogen import graphs, graphs_clean  # noqa: E402

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)
SAM = "sam3.1_multiplex_fp16.safetensors"


def log(*a):
    print(*a, flush=True)


def ensure_comfy(wait_s=600):
    """ComfyUI must answer before anything is sent: a server still starting (or busy loading
    a model) takes the connection and then stalls the request."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        try:
            if httpx.get(COMFY + "/system_stats", timeout=10).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise RuntimeError(f"ComfyUI did not answer /system_stats within {wait_s}s")


def retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn()
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if k == tries - 1:
                raise
            log(f"   comfy call failed ({exc}); waiting for ComfyUI and retrying")
            ensure_comfy()


def upload(path, sub="route_try"):
    def go():
        with open(path, "rb") as f:
            r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                           data={"subfolder": sub, "type": "input", "overwrite": "true"}, timeout=T)
        r.raise_for_status()
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]
    return retry(go)


def run(graph, dest, ext, timeout_s=1800):
    r = retry(lambda: httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T))
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
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}\n"
                                   f"{''.join(err.get('traceback') or [])[-1500:]}")
            for node_out in (e.get("outputs") or {}).values():
                for key in ("images", "videos", "gifs"):
                    for f in node_out.get(key, []) or []:
                        if f.get("type", "output") == "output" and f["filename"].lower().endswith(ext):
                            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                                resp.raise_for_status()
                                with open(dest, "wb") as out:
                                    for chunk in resp.iter_bytes(1 << 20):
                                        out.write(chunk)
                            return time.monotonic() - t0
            raise RuntimeError(f"no {ext} in outputs")
        time.sleep(2)
    httpx.post(COMFY + "/interrupt", timeout=10)
    raise RuntimeError(f"timed out after {timeout_s}s")


def free():
    try:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


def sam_graph(image, noun, prefix, threshold=0.4):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": noun, "clip": ["1", 1]}},
        "4": {"class_type": "SAM3_Detect", "inputs": {"model": ["1", 0], "image": ["2", 0], "conditioning": ["3", 0],
                                                      "threshold": threshold, "refine_iterations": 2, "individual_masks": False}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": prefix}},
    }


def ttm_graph(ref_name, mask_name, first_name, prompt, w, h, length, seed, prefix, steps, turbo, ttm):
    fam = "fl2va"
    g = {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": graphs.CHECKPOINTS[fam], "weight_dtype": "default"}}}
    model = ["unet", 0]
    if turbo:
        lora = graphs.TURBO_LORAS[fam].get(steps) or graphs.TURBO_LORAS[fam][8]
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": lora, "strength_model": 1.0}}
        model = ["lora", 0]
    g["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": graphs.TEXT_ENCODER, "type": "minimax", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": graphs.VIDEO_VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": graphs.AUDIO_VAE}}
    g["ref_v"] = {"class_type": "LoadVideo", "inputs": {"file": ref_name}}
    g["ref_p"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["ref_v", 0]}}
    g["m_v"] = {"class_type": "LoadVideo", "inputs": {"file": mask_name}}
    g["m_p"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["m_v", 0]}}
    g["mask"] = {"class_type": "ImageToMask", "inputs": {"image": ["m_p", 0], "channel": "red"}}
    g["mthr"] = {"class_type": "ThresholdMask", "inputs": {"mask": ["mask", 0], "value": 0.5}}
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["ref_p", 0], "vae": ["vae", 0]}}
    g["first"] = {"class_type": "LoadImage", "inputs": {"image": first_name}}
    g["cond"] = {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": w, "height": h,
                            "length": length, "first_frame": ["first", 0]}}
    if ttm:
        g["ttm"] = {"class_type": "vloTimeToMove",
                    "inputs": {"model": model, "reference_latents": ["enc", 0], "mask": ["mthr", 0],
                               "start_step": int(ttm[0]), "end_step": int(ttm[1])}}
        smodel = ["ttm", 0]
    else:
        smodel = model
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": smodel, "conditioning": ["cond", 0]}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": int(steps), "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["cond", 1]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 24.0, "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix, "format": "mp4"}}
    return g


# ---------------------------------------------------------------- route geometry

def catmull_rom(pts, per=64):
    """Dense points along a Catmull-Rom spline through pts (end points repeated)."""
    import numpy as np
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = (np.array(p[j], float) for j in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(np.array(pts[-1], float))
    return np.array(out)


def poses(route_px, n, hold, ease, end=None):
    """(x, y, heading radians) per frame: arc-length along the spline with an ease profile,
    still until frame `hold`, arriving at frame `end` (default the last)."""
    import numpy as np
    dense = catmull_rom(route_px)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    out = []
    end = n - 1 if end is None else min(int(end), n - 1)
    moving = max(1, end - hold)
    for f in range(n):
        u = min(1.0, max(0.0, (f - hold) / moving))
        if ease == "inout":
            u = u * u * (3 - 2 * u)
        d = u * total
        i = int(np.searchsorted(s, d, side="right") - 1)
        i = min(max(i, 0), len(dense) - 2)
        a = (d - s[i]) / max(seg[i], 1e-6)
        x, y = dense[i] + a * (dense[i + 1] - dense[i])
        # heading from a short look-ahead/behind window (smooth turns)
        j0, j1 = max(0, i - 3), min(len(dense) - 1, i + 4)
        dx, dy = dense[j1] - dense[j0]
        out.append((float(x), float(y), math.atan2(dy, dx)))
    return out


def car_axis(mask):
    """The car's long-axis angle (radians, image coords) and centre, from its mask."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    w, v = np.linalg.eigh(cov)
    vx, vy = v[:, int(np.argmax(w))]
    return math.atan2(vy, vx), (float(cx), float(cy))


def pick_cars(mask, cars, w0, h0):
    """One mask per car: the connected piece of the SAM mask under (or nearest to) each car's
    point."""
    import cv2
    import numpy as np
    n, lab = cv2.connectedComponents((mask > 127).astype(np.uint8))
    out = []
    for c in cars:
        px, py = int(c["at"][0] * w0), int(c["at"][1] * h0)
        k = lab[min(max(py, 0), h0 - 1), min(max(px, 0), w0 - 1)]
        if k == 0:
            best = None
            for j in range(1, n):
                ys, xs = np.nonzero(lab == j)
                if len(xs) < 50:
                    continue
                d = (xs.mean() - px) ** 2 + (ys.mean() - py) ** 2
                if best is None or d < best[0]:
                    best = (d, j)
            if best is None:
                raise RuntimeError(f"no car found near {c['at']}")
            k = best[1]
        out.append(((lab == k).astype(np.uint8) * 255))
    return out


def car_masks_for(still_p, mask_dir, cars, default_noun, w0, h0):
    """Each car's own SAM pass (its noun: "white SUV", "red sedan"...), then the piece under its
    point. One pass for all cars missed the weaker one (only the best-scoring car came back)."""
    import cv2
    import numpy as np
    si = upload(still_p)
    masks = []
    for i, c in enumerate(cars):
        dest = os.path.join(mask_dir, f"mask_{i}.png")
        run(sam_graph(si, c.get("noun", default_noun), f"route_try/sam{i}", float(c.get("threshold", 0.3))), dest, ".png")
        m = cv2.resize(cv2.imread(dest, cv2.IMREAD_GRAYSCALE), (w0, h0), interpolation=cv2.INTER_NEAREST)
        masks.append(pick_cars(m, [c], w0, h0)[0])
    for i in range(len(masks)):
        for j in range(i):
            inter = ((masks[i] > 0) & (masks[j] > 0)).sum()
            if inter > 0.5 * min((masks[i] > 0).sum(), (masks[j] > 0).sum()):
                raise RuntimeError(f"cars {j} and {i} found the same object; give them distinct nouns or points")
    return masks


def build_reference(still, plate, car_masks, cars, n, w, h, out):
    """Cut-and-drag clip: each car moved along its route over the clean plate, turned to face
    its direction of travel, and the mask of where they all are."""
    import cv2
    import numpy as np
    still_c = cv2.resize(still, (w, h), interpolation=cv2.INTER_AREA)
    plate_c = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA)
    car_src = still_c.astype(np.float32)
    k21 = np.ones((21, 21), np.uint8)
    tracks = []
    for cm, c in zip(car_masks, cars):
        m = cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        ang, (cx, cy) = car_axis(m)
        route = [(cx, cy)] + [(x * w, y * h) for x, y in c["route"][1:]]
        hold = int(c.get("hold", 6))
        ps = poses(route, n, hold, c.get("ease", "inout"), c.get("end"))
        # the car's heading: its long axis, pointed the way the route starts
        h0 = ps[min(n - 1, hold + 2)][2]
        if math.cos(ang - h0) < 0:
            ang += math.pi
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)
        tracks.append((m, ang, cx, cy, hold, ps, alpha, route))
    frames, masks = [], []
    for f in range(n):
        if f == 0:
            frames.append(still_c.copy())
            u = np.zeros((h, w), np.uint8)
            for m, *_ in tracks:
                u |= m.astype(np.uint8) * 255
            masks.append(cv2.dilate(u, k21))
            continue
        img = plate_c.astype(np.float32)
        mk = np.zeros((h, w), np.uint8)
        for m, ang, cx, cy, hold, ps, alpha, _ in tracks:
            x, y, hd = ps[f]
            # rotate by the change of heading (ramped in over the first frames of moving so the
            # still car does not snap round), about the car's own centre, then move it on
            ramp = min(1.0, max(f - hold, 0) / 8.0) if hold > 0 else min(1.0, f / 8.0)
            dth = math.atan2(math.sin(hd - ang), math.cos(hd - ang)) * ramp
            M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(dth), 1.0)
            M[0, 2] += x - cx
            M[1, 2] += y - cy
            c = cv2.warpAffine(car_src, M, (w, h), flags=cv2.INTER_LINEAR)
            a = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR)[..., None]
            img = img * (1 - a) + c * a
            mk |= (a[..., 0] > 0.5).astype(np.uint8) * 255
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        masks.append(cv2.dilate(mk, k21))
    # route picture
    pic = still_c.copy()
    colours = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (255, 0, 255)]
    for i, (*_, ps, _a, route) in enumerate(tracks):
        col = colours[i % len(colours)]
        for (x0, y0, _), (x1, y1, _) in zip(ps, ps[1:]):
            cv2.line(pic, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
        for x, y in route:
            cv2.circle(pic, (int(x), int(y)), 5, (0, 255, 255), -1)
    cv2.imwrite(os.path.join(out, "route.png"), pic)
    return still_c, frames, masks


def write_video(path, frames, fps=24):
    import subprocess
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def clear_out(out):
    """The python engine's output dir sits in a shared, reused repo cache and is zipped whole
    after the run: start it empty, or every earlier spike's results ride along (700 MB zips
    timed out the upload)."""
    import shutil
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)


def main():
    import cv2
    import numpy as np
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    clear_out(out)
    still_p = os.path.join(out, "still.png")
    with httpx.stream("GET", job["still"], timeout=T, follow_redirects=True) as r:
        r.raise_for_status()
        with open(still_p, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    still = cv2.imread(still_p)
    ensure_comfy()
    H0, W0 = still.shape[:2]
    w, h, n = int(job.get("width", 832)), int(job.get("height", 480)), int(job.get("frames", 121))
    try:
        # 1. car mask(s)
        cars = job.get("cars")
        t0 = time.monotonic()
        if cars:
            car_masks = car_masks_for(still_p, out, cars, job.get("noun", "car"), W0, H0)
            mask = np.max(np.stack(car_masks), axis=0)
            cv2.imwrite(os.path.join(out, "mask.png"), mask)
        else:
            si = upload(still_p)
            run(sam_graph(si, job.get("noun", "car"), "route_try/sam"), os.path.join(out, "mask.png"), ".png")
            mask = cv2.imread(os.path.join(out, "mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
        t = time.monotonic() - t0
        log(f"sam: {t:.0f}s, car covers {float((mask > 127).mean()):.3f}")
        free()
        # 2. clean plate: Krea redraws the car's area (plus room for its shadow)
        kw, kh = graphs_clean.size_for(W0, H0)
        km = cv2.dilate((mask > 127).astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
        cv2.imwrite(os.path.join(out, "_ki.png"), cv2.resize(still, (kw, kh), interpolation=cv2.INTER_AREA))
        cv2.imwrite(os.path.join(out, "_km.png"), cv2.resize(km, (kw, kh), interpolation=cv2.INTER_NEAREST))
        t = run(graphs_clean.build(upload(os.path.join(out, "_ki.png")), upload(os.path.join(out, "_km.png")),
                                   job["scene"], 6332, "route_try/plate"), os.path.join(out, "plate_raw.png"), ".png")
        plate = cv2.resize(cv2.imread(os.path.join(out, "plate_raw.png")), (W0, H0), interpolation=cv2.INTER_CUBIC)
        a = (cv2.GaussianBlur(km, (0, 0), 4).astype(np.float32) / 255.0)[..., None]
        plate = np.clip(still * (1 - a) + plate * a, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out, "plate.png"), plate)
        log(f"plate: {t:.0f}s")
        free()
        # 3. cut-and-drag reference
        if not cars:
            cars = [{"route": job["route"], "hold": int(job.get("hold", 6)), "ease": job.get("ease", "inout")}]
            car_masks = [mask]
        first, frames, masks = build_reference(still, plate, car_masks, cars, n, w, h, out)
        cv2.imwrite(os.path.join(out, "first.png"), first)
        write_video(os.path.join(out, "reference.mp4"), frames)
        write_video(os.path.join(out, "refmask.mp4"), [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks])
        names = [upload(os.path.join(out, f)) for f in ("reference.mp4", "refmask.mp4", "first.png")]
        # 4. renders
        for rd in job.get("renders", [{"name": "ttm_full", "steps": 20, "turbo": False, "ttm": [1, 3]}]):
            t = run(ttm_graph(*names, job["prompt"], w, h, n, int(rd.get("seed", 6332)), f"route_try/{rd['name']}",
                              int(rd.get("steps", 20)), bool(rd.get("turbo", False)), rd.get("ttm")),
                    os.path.join(out, f"{rd['name']}.mp4"), ".mp4")
            log(f"render {rd['name']}: {t:.0f}s")
    finally:
        free()
        for f in ("_ki.png", "_km.png"):
            try:
                os.remove(os.path.join(out, f))
            except OSError:
                pass


if __name__ == "__main__":
    main()
