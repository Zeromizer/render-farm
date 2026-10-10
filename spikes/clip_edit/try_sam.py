"""Test driver: SAM 3.1 text-prompted object tracking on the render PC's ComfyUI (native
SAM3 nodes, ComfyUI >= 0.37) through the farm's python engine.

params.args = [<job json string>, <out dir>]. The job JSON:
  {"clip": <signed url>, "tests": [{"name", "prompt", "detect_frame", "threshold",
   "track_threshold", "max_objects", "detect_interval"}]}
Writes <out>/object_info.json (the SAM3 nodes' real schemas), and per test
<out>/<name>/{mask.mp4, overlay.mp4, coverage.json}.
"""
import json
import os
import re
import sys
import time
import traceback
import uuid

import httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)
CKPT = "sam3.1_multiplex_fp16.safetensors"
NODES = ("SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask", "SAM3_TrackPreview", "CheckpointLoaderSimple",
         "ImageFromBatch", "MaskToImage", "CreateVideo", "SaveVideo", "GetVideoComponents", "LoadVideo")


def log(*a):
    print(*a, flush=True)


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "clip_edit_try", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_graph(graph, dest, timeout_s=1200):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:3000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        e = httpx.get(COMFY + f"/history/{pid}", timeout=15).json().get(pid)
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}\n"
                                   f"{''.join(err.get('traceback') or [])[-2000:]}")
            got = 0
            for node_out in (e.get("outputs") or {}).values():
                for key in ("videos", "images", "gifs"):
                    for f in node_out.get(key, []) or []:
                        if f.get("type", "output") == "output" and f["filename"].lower().endswith(".mp4"):
                            m = re.search(r"_obj(\d+)", f["filename"])
                            target = dest if not m else dest.replace("mask.mp4", f"obj{m.group(1)}.mp4")
                            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                                resp.raise_for_status()
                                with open(target, "wb") as out:
                                    for chunk in resp.iter_bytes(1 << 20):
                                        out.write(chunk)
                            got += 1
            if not got:
                raise RuntimeError("no mp4 in outputs")
            return time.monotonic() - t0
        time.sleep(2)
    httpx.post(COMFY + "/interrupt", timeout=10)
    raise RuntimeError(f"timed out after {timeout_s}s")


def graph(src, t, prefix):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadVideo", "inputs": {"file": src}},
        "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": t["prompt"], "clip": ["1", 1]}},
        "5": {"class_type": "ImageFromBatch", "inputs": {"image": ["3", 0], "batch_index": int(t.get("detect_frame", 0)),
                                                         "length": 1}},
        "6": {"class_type": "SAM3_Detect", "inputs": {"model": ["1", 0], "image": ["5", 0], "conditioning": ["4", 0],
                                                      "threshold": float(t.get("threshold", 0.4)),
                                                      "refine_iterations": 2, "individual_masks": False}},
        "7": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["3", 0], "model": ["1", 0], "initial_mask": ["6", 0],
                                                          "conditioning": ["4", 0],
                                                          "detection_threshold": float(t.get("track_threshold", 0.5)),
                                                          "max_objects": int(t.get("max_objects", 1)),
                                                          "detect_interval": int(t.get("detect_interval", 4))}},
        "8": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["7", 0], "object_indices": t.get("objects", "")}},
        "9": {"class_type": "MaskToImage", "inputs": {"mask": ["8", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": 24.0}},
        "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": prefix, "format": "mp4",
                                                     "codec": "h264"}},
    }
    # one mask video per object index (per_object: how many)
    for k in range(int(t.get("per_object", 0))):
        b = 100 + 10 * k
        g.update({
            str(b): {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["7", 0], "object_indices": str(k)}},
            str(b + 1): {"class_type": "MaskToImage", "inputs": {"mask": [str(b), 0]}},
            str(b + 2): {"class_type": "CreateVideo", "inputs": {"images": [str(b + 1), 0], "fps": 24.0}},
            str(b + 3): {"class_type": "SaveVideo", "inputs": {"video": [str(b + 2), 0], "filename_prefix": f"{prefix}_obj{k}",
                                                              "format": "mp4", "codec": "h264"}},
        })
    return g


def overlay(clip, mask, out_dir):
    import cv2
    import numpy as np
    a, b = cv2.VideoCapture(clip), cv2.VideoCapture(mask)
    frames, cov = [], []
    while True:
        ok1, f = a.read()
        ok2, m = b.read()
        if not (ok1 and ok2):
            break
        if m.shape[:2] != f.shape[:2]:
            m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        on = m[..., 0] > 127
        cov.append(round(float(on.mean()), 4))
        g = f.copy()
        g[on] = (0.5 * g[on] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        frames.append(g)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(os.path.join(out_dir, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for g in frames:
        vw.write(g)
    vw.release()
    json.dump({"coverage": cov, "frames_with_mask": sum(c > 0.001 for c in cov)},
              open(os.path.join(out_dir, "coverage.json"), "w"))


def main():
    job = json.loads(sys.argv[1])
    out_root = os.path.abspath(sys.argv[2])
    os.makedirs(out_root, exist_ok=True)
    info = httpx.get(COMFY + "/object_info", timeout=60).json()
    json.dump({n: info.get(n) for n in NODES}, open(os.path.join(out_root, "object_info.json"), "w"), indent=1)
    ck = (info.get("CheckpointLoaderSimple") or {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
    log(f"sam3 checkpoint listed: {CKPT in ck}")
    clip = os.path.join(out_root, "clip.mp4")
    with httpx.stream("GET", job["clip"], timeout=T, follow_redirects=True) as r:
        r.raise_for_status()
        with open(clip, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    src = upload(clip)
    failures = {}
    try:
        for t in job["tests"]:
            d = os.path.join(out_root, t["name"])
            os.makedirs(d, exist_ok=True)
            try:
                secs = run_graph(graph(src, t, f"clip_edit_try/sam_{t['name']}"), os.path.join(d, "mask.mp4"))
                log(f"{t['name']}: {secs:.0f}s")
                overlay(clip, os.path.join(d, "mask.mp4"), d)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                failures[t["name"]] = str(exc)[:3000]
    finally:
        try:
            httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
        except httpx.HTTPError:
            pass
    os.remove(clip)
    json.dump({"failures": failures}, open(os.path.join(out_root, "failures.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
