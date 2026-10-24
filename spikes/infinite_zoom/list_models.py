"""Dump what the PC's ComfyUI can load (model folders + whether key inpaint/edit nodes exist)."""
import json
import os
import sys

import httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
out = os.path.abspath(sys.argv[1])
os.makedirs(out, exist_ok=True)
rep = {}
folders = httpx.get(COMFY + "/models", timeout=60).json()
for f in folders:
    try:
        rep[f] = httpx.get(COMFY + f"/models/{f}", timeout=60).json()
    except Exception as e:  # noqa: BLE001
        rep[f] = str(e)
info = httpx.get(COMFY + "/object_info", timeout=120).json()
rep["_nodes_of_interest"] = sorted(n for n in info if any(s in n.lower() for s in
                                   ("inpaint", "outpaint", "qwen", "edit", "fill", "kontext", "differential", "pad")))
rep["_node_count"] = len(info)
with open(os.path.join(out, "models.json"), "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps(rep)[:4000])
