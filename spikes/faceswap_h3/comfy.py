"""Local ComfyUI (the PC's headless instance on 8188): copy inputs in, run API graphs, find outputs."""
import json
import os
import shutil
import time
import urllib.request
import uuid

URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
INPUT = os.path.join(COMFY_DIR, "input")
OUTPUT = os.path.join(COMFY_DIR, "output")


def _http(path, data=None):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def put_input(path, name=None):
    """Copy a local file into ComfyUI's input dir; returns the name graphs refer to."""
    name = name or os.path.basename(path)
    shutil.copyfile(path, os.path.join(INPUT, name))
    return name


def run_outputs(graph, poll=2.0):
    """Queue an API graph and block until it finishes. Returns the raw per-node outputs from /history."""
    pid = _http("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        time.sleep(poll)
        h = _http(f"/history/{pid}")
        if pid not in h:
            continue
        e = h[pid]
        for msg in e.get("status", {}).get("messages", []):
            if msg[0] == "execution_error":
                raise RuntimeError(f"ComfyUI error in {msg[1].get('node_type')}: {msg[1].get('exception_message')}")
        return e.get("outputs", {})


def run(graph, poll=2.0):
    """Queue an API graph and block until it finishes. Returns the list of output files (absolute paths)."""
    files = []
    for o in run_outputs(graph, poll).values():
        for kind in ("images", "videos", "gifs", "video"):
            for f in o.get(kind, []) or []:
                if isinstance(f, dict) and "filename" in f and f.get("type", "output") == "output":
                    files.append(os.path.join(OUTPUT, f.get("subfolder", ""), f["filename"]))
    return files
