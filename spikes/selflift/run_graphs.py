"""Post API graphs to the PC ComfyUI one by one, collect every saved video into OUT_DIR, log timings/errors.
usage: python run_graphs.py OUT_DIR jobs.json [label,label]"""
import json
import os
import sys
import time
import uuid

import httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
out = os.path.abspath(sys.argv[1])
os.makedirs(out, exist_ok=True)
with open(sys.argv[2], encoding="utf-8") as f:
    jobs = json.load(f)
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None
rep = []
for i, j in enumerate(jobs):
    if only and j["label"] not in only:
        continue
    t0 = time.time()
    r = httpx.post(COMFY + "/prompt", json={"prompt": j["graph"], "client_id": str(uuid.uuid4())}, timeout=120)
    if r.status_code != 200:
        rep.append({"label": j["label"], "error": r.text[:4000]})
        print(r.text[:2000], flush=True)
        continue
    pid = r.json()["prompt_id"]
    while True:
        time.sleep(5)
        e = httpx.get(COMFY + f"/history/{pid}", timeout=60).json().get(pid)
        if e and (e.get("status", {}).get("completed") or e.get("status", {}).get("status_str") == "error"):
            break
    st = e["status"]
    item = {"label": j["label"], "seconds": round(time.time() - t0), "status": st.get("status_str")}
    if st.get("status_str") == "error":
        item["messages"] = st.get("messages")
    files = []
    for node in e.get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            for it in node.get(key, []):
                if it.get("type") != "output":
                    continue
                dst = os.path.join(out, it["filename"])
                with httpx.stream("GET", COMFY + "/view", params={"filename": it["filename"], "subfolder": it.get("subfolder", ""),
                                                                 "type": "output"}, timeout=600) as resp:
                    with open(dst, "wb") as fh:
                        for c in resp.iter_bytes():
                            fh.write(c)
                files.append(it["filename"])
        if "text" in node:
            item.setdefault("text", []).extend(node["text"] if isinstance(node["text"], list) else [node["text"]])
    item["files"] = files
    rep.append(item)
    print(json.dumps(item)[:3000], flush=True)
    print(f"PROGRESS {int(100 * (i + 1) / len(jobs))}", flush=True)
with open(os.path.join(out, "report.json"), "w") as f:
    json.dump(rep, f, indent=1)
