"""Spike: multi-anchor lettering_fix + a harmonise pass, on the render PC's ComfyUI.

Runs through the farm's python engine (it never touches the live worker's code). For each
test in the job JSON:

  pass 1  lettering.py prep with anchor_mode "multi": the real lettering is warped into
          8-frame runs on H3's latent grid (light-matched, motion-blurred), everything else
          on the plate is masked; H3 (production graph, graphs_inpaint.build) fills the
          masked frames; compose scores the takes and keeps the best.
  pass 2  lettering.py harmonise: mask ONLY the pasted anchor runs on the pass-1 result and
          let H3 redraw them from the now-correct frames around them, so no frame of the
          plate is a paste.

Writes per test: pass1.mp4, pass2.mp4, both proof sheets and reports, a side-by-side
compare (original | pass 1 | pass 2, follow-crop on the element) and metrics.json
(flicker inside runs vs at run hand-offs). Usage: python run.py <job.json> <out_dir>
"""
import json
import os
import shutil
import sys
import time
import uuid

import cv2
import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "worker", "lettering"))
sys.path.insert(0, os.path.join(REPO, "worker"))
import lettering  # noqa: E402
from videogen import graphs_inpaint  # noqa: E402

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)


def log(*a):
    print(*a, flush=True)


def comfy_up():
    try:
        return httpx.get(COMFY + "/system_stats", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_comfy():
    """Wait (up to 4 min) for ComfyUI; never launch it, the worker owns that."""
    for _ in range(120):
        if comfy_up():
            return
        time.sleep(2)
    raise RuntimeError("ComfyUI did not answer /system_stats within 4 min")


def upload(path, sub):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": sub, "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_graph(graph, dest, label, timeout_s=1500):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:2000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    last = 0
    while time.monotonic() - t0 < timeout_s:
        e = httpx.get(COMFY + f"/history/{pid}", timeout=15).json().get(pid)
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"{label}: execution error at {err.get('node_type')}: "
                                   f"{err.get('exception_message') or json.dumps(msgs)[:1200]}")
            for node_out in (e.get("outputs") or {}).values():
                for key in ("videos", "images", "gifs"):
                    for f in node_out.get(key, []) or []:
                        if f.get("type", "output") == "output" and f["filename"].lower().endswith(".mp4"):
                            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                                resp.raise_for_status()
                                with open(dest, "wb") as out:
                                    for chunk in resp.iter_bytes(1 << 20):
                                        out.write(chunk)
                            log(f"{label}: done in {time.monotonic() - t0:.0f}s")
                            return dest
            raise RuntimeError(f"{label}: no mp4 in outputs")
        if time.monotonic() - last > 30:     # the python engine treats silence as a hang
            log(f"{label}: running {time.monotonic() - t0:.0f}s")
            last = time.monotonic()
        time.sleep(3)
    httpx.post(COMFY + "/interrupt", timeout=10)
    raise RuntimeError(f"{label}: timed out after {timeout_s}s")


def free_vram():
    try:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


def h3_takes(work, prefix_files, prompt, seeds, steps, turbo, tag, test):
    plan_name = "plan.json" if prefix_files == "gen" else "plan2.json"
    plan = json.load(open(os.path.join(work, plan_name)))
    W, H = plan["canvas"]
    sub = f"lettering_multi/{test}"
    names = [upload(os.path.join(work, f"{prefix_files}_{k}.mp4"), sub) for k in ("src", "mask", "amask")]
    out = []
    for i, seed in enumerate(seeds):
        graph, meta = graphs_inpaint.build(*names, prompt, W, H, plan["length"], seed,
                                           f"lettering_multi/{test}_{tag}_t{i + 1}", steps=steps, turbo=turbo)
        dest = os.path.join(work, f"{tag}raw{i + 1}.mp4")
        log(f"[{test}] {tag} take {i + 1}/{len(seeds)}: {W}x{H} {plan['length']}f seed={seed} steps={meta['steps']}")
        out.append(run_graph(graph, dest, f"[{test}] {tag} take {i + 1}"))
    return out


def read(path):
    return lettering.read_clip(path)[0]


def element_crops(frames, plan, size=(480, 240)):
    """Follow-crop around element 0 for every window frame (zoomed, for the compare)."""
    t = {int(k): v for k, v in plan["tracks"][0].items()}
    lo, hi = plan["window"]
    b0 = plan["anchor_boxes"][0]
    out = {}
    for i in range(lo, hi + 1):
        b = t.get(i)
        if b is None:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        w = max(b[2] - b[0], 24) * 2.0
        h = w * size[1] / size[0]
        fh, fw = frames[0].shape[:2]
        x0 = int(min(max(0, cx - w / 2), fw - w)); y0 = int(min(max(0, cy - h / 2), fh - h))
        c = frames[i][y0:y0 + int(h), x0:x0 + int(w)]
        if c.size:
            out[i] = cv2.resize(c, size, interpolation=cv2.INTER_CUBIC)
    return out


def metrics(crops, runs):
    """Mean abs change of the element crop frame to frame, split into: inside a run, at a
    run's edges (the hand-off between pasted and generated frames), elsewhere."""
    edges = {s for s, _ in runs} | {e + 1 for _, e in runs}
    inside = {i for s, e in runs for i in range(s + 1, e + 1)}
    acc = {"handoff": [], "inside_runs": [], "elsewhere": []}
    for i in sorted(crops):
        if i - 1 not in crops:
            continue
        d = float(np.abs(crops[i].astype(np.float32) - crops[i - 1].astype(np.float32)).mean())
        acc["handoff" if i in edges else "inside_runs" if i in inside else "elsewhere"].append(d)
    return {k: {"mean": round(float(np.mean(v)), 2) if v else None, "max": round(float(np.max(v)), 2) if v else None,
                "n": len(v)} for k, v in acc.items()}


def compare(work, clip, plan, out_path):
    a, b, c = read(clip), read(os.path.join(work, "pass1.mp4")), read(os.path.join(work, "pass2.mp4"))
    ca, cb, cc = element_crops(a, plan), element_crops(b, plan), element_crops(c, plan)
    runs = plan.get("anchor_runs") or []
    anchors = {i for s, e in runs for i in range(s, e + 1)}
    frames = []
    for i in sorted(set(ca) & set(cb) & set(cc)):
        row = []
        for img, lab in ((ca[i], "original"), (cb[i], "pass 1" + (" ANCHOR" if i in anchors else "")), (cc[i], "pass 2")):
            img = img.copy()
            cv2.putText(img, f"{i} {lab}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, f"{i} {lab}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            row.append(img)
        frames.append(np.hstack(row))
    if frames:
        lettering.write_clip(out_path, frames, 24)
    return {"pass1": metrics(cb, runs), "pass2": metrics(cc, runs), "original": metrics(ca, runs)}


def run_test(t, out_dir, media):
    name = t["name"]
    work = os.path.join(out_dir, name)
    os.makedirs(work, exist_ok=True)
    spec = {"clip": os.path.join(media, t["clip"]), "reference": os.path.join(media, t["reference"]),
            "work_dir": work, "elements": t["elements"], "anchor_frame": t["anchor_frame"],
            "anchor_mode": "multi", "anchor_every": t.get("anchor_every", 1)}
    log(f"[{name}] prep")
    lettering.prep(spec)
    plan = json.load(open(os.path.join(work, "plan.json")))
    if plan.get("anchor_policy") != "multi":
        raise RuntimeError(f"[{name}] no multi-anchor runs: {plan.get('warnings')}")
    seeds = [6332 + 1009 * i for i in range(t.get("takes", 2))]
    takes = h3_takes(work, "gen", t["prompt"], seeds, t.get("steps"), t.get("turbo", True), "p1", name)
    lettering.compose(dict(spec, takes=takes, seeds=seeds, out=os.path.join(work, "pass1.mp4"),
                           proof=os.path.join(work, "pass1-proof.png"), report=os.path.join(work, "pass1.json")))
    log(f"[{name}] harmonise")
    lettering.prep_harmonise(dict(spec, base=os.path.join(work, "pass1.mp4")))
    seeds2 = [7341 + 1009 * i for i in range(t.get("takes2", 1))]
    takes2 = h3_takes(work, "gen2", t["prompt"], seeds2, t.get("steps"), t.get("turbo", True), "p2", name)
    lettering.compose(dict(spec, plan="plan2.json", base=os.path.join(work, "pass1.mp4"), tag="h_", takes=takes2,
                           seeds=seeds2, out=os.path.join(work, "pass2.mp4"),
                           proof=os.path.join(work, "pass2-proof.png"), report=os.path.join(work, "pass2.json")))
    m = compare(work, spec["clip"], plan, os.path.join(out_dir, f"{name}-compare.mp4"))
    for f in ("pass1.mp4", "pass2.mp4", "pass1-proof.png", "pass2-proof.png", "pass1.json", "pass2.json", "plan.json"):
        shutil.copy(os.path.join(work, f), os.path.join(out_dir, f"{name}-{f}"))
    log(f"[{name}] metrics {json.dumps(m)}")
    return m


def main():
    job = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    media = os.path.join(HERE, "media")
    ensure_comfy()
    results = {}
    try:
        for k, t in enumerate(job["tests"]):
            results[t["name"]] = run_test(t, out_dir, media)
            print(f"PROGRESS {int(100 * (k + 1) / len(job['tests']))}", flush=True)
            json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    finally:
        free_vram()
        # work dirs hold big lossless intermediates; the zip only needs the copies above
        for t in job["tests"]:
            shutil.rmtree(os.path.join(out_dir, t["name"]), ignore_errors=True)
    log(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
