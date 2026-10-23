"""ComfyUI over HTTP for the fsh3 stages: upload inputs, run API graphs, fetch outputs, free memory.

Everything goes through the API, never ComfyUI's folders: a cloud pod's ComfyUI runs with --input-directory /
--output-directory outside COMFYUI_DIR, so copying files in (the spike's first version) put them where it never
looks. Inputs go into a per-job subfolder (FSH3_SUBFOLDER), so two jobs on one ComfyUI never share a name; outputs
are downloaded into FSH3_FETCH_DIR (default: a temp folder).
"""
import json
import mimetypes
import os
import tempfile
import time
import urllib.parse
import urllib.request
import uuid

URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
SUBFOLDER = os.environ.get("FSH3_SUBFOLDER", "fsh3")
FETCH_DIR = os.environ.get("FSH3_FETCH_DIR") or os.path.join(tempfile.gettempdir(), "fsh3_outputs")


def _http(path, data=None, timeout=60):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}


def put_input(path, name=None):
    """Upload a local file as a ComfyUI input (images and videos alike, /upload/image); returns the name graphs
    refer to ("<subfolder>/<name>")."""
    name = name or os.path.basename(path)
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    parts = []
    for k, v in (("subfolder", SUBFOLDER), ("type", "input"), ("overwrite", "true")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    with open(path, "rb") as f:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + f.read() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(URL + "/upload/image", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        got = json.loads(r.read())
    return f"{got['subfolder']}/{got['name']}" if got.get("subfolder") else got["name"]


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


def _fetch(f):
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"})
    dest = os.path.join(FETCH_DIR, f.get("subfolder", ""), f["filename"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(f"{URL}/view?{q}", timeout=600) as r, open(dest, "wb") as out:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
    return dest


def run(graph, poll=2.0):
    """Queue an API graph and block until it finishes. Returns its output files, downloaded (local paths that keep
    ComfyUI's subfolder/filename, so callers can still filter by name)."""
    files = []
    for o in run_outputs(graph, poll).values():
        for kind in ("images", "videos", "gifs", "video"):
            for f in o.get(kind, []) or []:
                if isinstance(f, dict) and "filename" in f and f.get("type", "output") == "output":
                    files.append(_fetch(f))
    return files


def free():
    """Unload models and free memory (SAM 3.1 / the VLM before H3, H3 before SAM 3.1 again)."""
    try:
        _http("/free", {"unload_models": True, "free_memory": True})
    except Exception:  # noqa: BLE001 - only a memory hint
        pass
