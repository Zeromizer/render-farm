"""Thin HTTP client for the headless ComfyUI that hosts MiniMax H3.

Only the routes the video_gen runner needs, all against server.py in the
ComfyUI checkout (/system_stats, /upload/image, /prompt, /history/{id},
/view, /interrupt, /free). No websocket: /history is the durable source of
truth and a 2 s poll is fine for a job that takes minutes.

ComfyUI is started lazily by ensure_server() from run-headless.bat and left
resident, so the second job of the day skips the ~minute of model paging.
free() after each job hands VRAM back to the TTS workers.
"""
import json
import os
import subprocess
import time
import uuid

import httpx

import config

_TIMEOUT = httpx.Timeout(30.0, read=600.0)


class ComfyError(RuntimeError):
    pass


def _url(path):
    return config.COMFYUI_URL.rstrip("/") + path


def is_up():
    try:
        return httpx.get(_url("/system_stats"), timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def system_stats():
    return httpx.get(_url("/system_stats"), timeout=10).json()


def ensure_server(log, wait_seconds=240):
    """Start run-headless.bat if nothing answers on COMFYUI_URL; block until it does."""
    if is_up():
        return
    bat = os.path.join(config.COMFYUI_DIR, "run-headless.bat")
    if not os.path.exists(bat):
        raise ComfyError(f"ComfyUI launcher missing: {bat}")
    log(f"comfyui not running; launching {bat}")
    # Detached: the server must outlive this job (and this worker process).
    # The bat redirects its own output to comfyui-headless.log; handing it a
    # file handle from here did not survive the detached launch.
    subprocess.Popen(["cmd", "/c", bat], cwd=config.COMFYUI_DIR,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                     | getattr(subprocess, "DETACHED_PROCESS", 0))
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if is_up():
            log("comfyui is up")
            return
        time.sleep(2)
    raise ComfyError(f"ComfyUI did not answer on {config.COMFYUI_URL} within {wait_seconds}s "
                     f"(see {os.path.join(config.COMFYUI_DIR, 'comfyui-headless.log')})")


def upload_input(local_path, subfolder="video_gen"):
    """Put a file in ComfyUI/input/<subfolder>/ and return the name a Load* node wants."""
    with open(local_path, "rb") as f:
        r = httpx.post(_url("/upload/image"),
                       files={"image": (os.path.basename(local_path), f)},
                       data={"subfolder": subfolder, "type": "input", "overwrite": "true"},
                       timeout=_TIMEOUT)
    if r.status_code != 200:
        raise ComfyError(f"upload failed HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    sub = j.get("subfolder") or ""
    return f"{sub}/{j['name']}" if sub else j["name"]


def submit(graph):
    client_id = str(uuid.uuid4())
    r = httpx.post(_url("/prompt"), json={"prompt": graph, "client_id": client_id}, timeout=_TIMEOUT)
    if r.status_code != 200:
        # 400 carries node_errors: surface them, they are the whole diagnosis.
        try:
            detail = json.dumps(r.json(), indent=1)[:2000]
        except ValueError:
            detail = r.text[:2000]
        raise ComfyError(f"/prompt rejected (HTTP {r.status_code}):\n{detail}")
    j = r.json()
    if j.get("node_errors"):
        raise ComfyError(f"/prompt node errors: {json.dumps(j['node_errors'])[:2000]}")
    return j["prompt_id"]


def interrupt():
    try:
        httpx.post(_url("/interrupt"), timeout=10)
    except httpx.HTTPError:
        pass


def free():
    try:
        httpx.post(_url("/free"), json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


def _queue_position(prompt_id):
    try:
        q = httpx.get(_url("/queue"), timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return None
    for item in q.get("queue_running", []):
        if item[1] == prompt_id:
            return 0
    for i, item in enumerate(q.get("queue_pending", [])):
        if item[1] == prompt_id:
            return i + 1
    return None


def wait(prompt_id, on_status, cancel_check, timeout_seconds, poll=2.0):
    """Block until /history has the prompt. on_status(phase, progress_0_100) is
    called as things change; progress here is coarse (queue → running), the
    fine-grained sampler steps come from the /progress websocket which we skip.
    """
    started = time.monotonic()
    last_phase = None
    unreachable_since = None
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            interrupt()
            raise ComfyError(f"generation exceeded {timeout_seconds}s")
        if cancel_check():
            interrupt()
            raise _CanceledSignal()
        try:
            h = httpx.get(_url(f"/history/{prompt_id}"), timeout=10).json()
            unreachable_since = None
        except (httpx.HTTPError, ValueError):
            # A server that stops answering mid-job has almost certainly died
            # (the 243-frame SeedVR2 decode took the process to 82 GB virtual
            # and Windows killed it). Fail fast instead of polling to timeout.
            h = {}
            unreachable_since = unreachable_since or time.monotonic()
            if time.monotonic() - unreachable_since > 45:
                raise ComfyError("comfyui stopped answering mid-job (crashed? see comfyui-headless.log "
                                 "and the Windows Application log for a low-memory event)")
        entry = h.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error" or not status.get("completed", True):
                msgs = status.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), None)
                detail = (err or {}).get("exception_message") or json.dumps(msgs)[:1500]
                node = (err or {}).get("node_type")
                raise ComfyError(f"comfyui execution error at {node}: {detail}")
            return entry.get("outputs") or {}
        pos = _queue_position(prompt_id)
        if pos is None:
            phase = "running"
        elif pos == 0:
            phase = "running"
        else:
            phase = f"queued ({pos} ahead)"
        if phase != last_phase:
            on_status(phase, 10 if phase.startswith("queued") else 15)
            last_phase = phase
        time.sleep(poll)


class _CanceledSignal(Exception):
    """Translated to proc.Canceled by the runner (keeps this module free of proc)."""


def fetch_output(outputs, dest_path):
    """Find the saved video in a /history outputs dict and download it.

    Only type=="output" entries count: LoadVideo reports its *input* file in the
    same outputs dict as a UI preview (type "input"), and taking that returned
    the source clip instead of the upscale once.
    """
    for node_out in outputs.values():
        for key in ("videos", "images", "gifs"):
            for f in node_out.get(key, []) or []:
                name = f.get("filename", "")
                if f.get("type", "output") != "output":
                    continue
                if not name.lower().endswith((".mp4", ".mkv", ".webm")):
                    continue
                params = {"filename": name, "subfolder": f.get("subfolder", ""), "type": "output"}
                with httpx.stream("GET", _url("/view"), params=params, timeout=_TIMEOUT) as r:
                    if r.status_code != 200:
                        raise ComfyError(f"/view failed HTTP {r.status_code} for {name}")
                    with open(dest_path, "wb") as out:
                        for chunk in r.iter_bytes(1 << 20):
                            out.write(chunk)
                return dest_path
    raise ComfyError(f"no video in comfyui outputs: {json.dumps(outputs)[:800]}")
