"""Test driver: top-down stills for the route spike, from Krea 2 on the render PC's ComfyUI
(the clean graph with the whole frame masked = text-to-image with known nodes).

params.args = [<job json string>, <out dir>]. Job JSON:
  {"prompt": "...", "seeds": [1, 2, 3], "width": 1344, "height": 768}
Writes <out>/still_<seed>.png.
"""
import json
import os
import sys
import time
import uuid

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))
from videogen import graphs_clean  # noqa: E402

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "route_try", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_png(graph, dest, timeout_s=900):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:2000]}")
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
            for node_out in (e.get("outputs") or {}).values():
                for f in node_out.get("images", []) or []:
                    if f.get("type", "output") == "output":
                        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                        with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                            resp.raise_for_status()
                            with open(dest, "wb") as out:
                                for chunk in resp.iter_bytes(1 << 20):
                                    out.write(chunk)
                        return time.monotonic() - t0
            raise RuntimeError("no image in outputs")
        time.sleep(2)
    raise RuntimeError("timed out")


def main():
    import cv2
    import numpy as np
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    w, h = int(job.get("width", 1344)), int(job.get("height", 768))
    grey, white = os.path.join(out, "_grey.png"), os.path.join(out, "_white.png")
    cv2.imwrite(grey, np.full((h, w, 3), 127, np.uint8))
    cv2.imwrite(white, np.full((h, w), 255, np.uint8))
    gi, wi = upload(grey), upload(white)
    try:
        for seed in job.get("seeds", [1, 2, 3]):
            secs = run_png(graphs_clean.build(gi, wi, job["prompt"], int(seed), f"route_try/still_{seed}", grow=0),
                           os.path.join(out, f"still_{seed}.png"))
            print(f"still seed {seed}: {secs:.0f}s", flush=True)
    finally:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
        for f in (grey, white):
            os.remove(f)


if __name__ == "__main__":
    main()
