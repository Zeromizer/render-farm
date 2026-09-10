"""Thin HTTP client for the headless ComfyUI that hosts MiniMax H3.

Only the routes the video_gen runner needs, all against server.py in the
ComfyUI checkout (/system_stats, /upload/image, /prompt, /history/{id},
/view, /interrupt, /free). No websocket: /history is the durable source of
truth and a 2 s poll is fine for a job that takes minutes.

ComfyUI is started lazily by ensure_server() from run-headless.bat and left
resident, so the second job of the day skips the ~minute of model paging.
free() after each job hands VRAM back to the TTS workers.
"""
import glob
import json
import re
import shutil
from datetime import datetime
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


def object_info():
    """Every registered node class with its input schema (several MB). Used by
    videogen/h3_preflight to fail loudly before /prompt when a node pack or a
    model file is missing."""
    r = httpx.get(_url("/object_info"), timeout=60)
    if r.status_code != 200:
        raise ComfyError(f"/object_info failed HTTP {r.status_code}")
    return r.json()


def ensure_server(log, wait_seconds=240):
    """Start run-headless.bat if nothing answers on COMFYUI_URL; block until it does."""
    if is_up():
        return
    bat = os.path.join(config.COMFYUI_DIR, "run-headless.bat")
    if not os.path.exists(bat):
        raise ComfyError(f"ComfyUI launcher missing: {bat}")
    log(f"comfyui not running; launching {bat}")
    # CREATE_NO_WINDOW only: cmd gets a hidden console that python.exe inherits.
    # Adding DETACHED_PROCESS (as this once did) gave cmd no console at all, so
    # python allocated its own, and Windows 11 hands a fresh console to Windows
    # Terminal: a visible tab titled ".venv\Scripts\python.exe" on every launch.
    # The server outlives the worker regardless (no job object ties them). ComfyUI
    # writes its own log to comfyui-headless.log (see the bat).
    subprocess.Popen(["cmd", "/c", bat], cwd=config.COMFYUI_DIR,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
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


_TQDM = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)<[^,]*,\s*([\d.]+)(s/it|it/s)\]")


def sampling_progress(since_iso):
    """(step, total, seconds_per_step) from the newest tqdm line ComfyUI logged
    after since_iso (its /internal/logs/raw buffer keeps the last 300 lines,
    each stamped with a local ISO time), or None before sampling starts. This
    replaces the /ws progress socket, which would need a websocket client."""
    try:
        entries = httpx.get(_url("/internal/logs/raw"), timeout=5).json().get("entries") or []
    except (httpx.HTTPError, ValueError):
        return None
    for e in reversed(entries):
        if e.get("t", "") < since_iso:
            break
        m = _TQDM.search(e.get("m") or "")
        if m:
            rate = float(m.group(5))
            if m.group(6) == "it/s":
                rate = 1.0 / rate if rate else 0.0
            return int(m.group(1)), int(m.group(2)), rate
    return None


def wait(prompt_id, on_status, cancel_check, timeout_seconds, poll=2.0, hint=None, since_iso=None):
    """Block until /history has the prompt. on_status(phase, fraction, eta_seconds)
    is called as things change: fraction is 0..1 through this prompt's work
    (queue -> load -> sampling steps -> decode), eta_seconds the estimated time
    left for it. `hint` = videogen.estimate.sampling_hint(...) seeds the ETA
    before the first step; once ComfyUI logs a step the measured s/it wins.
    """
    started = time.monotonic()
    since_iso = since_iso or datetime.now().isoformat()
    hint = hint or {}
    steps_hint = int(hint.get("steps") or 8)
    step_s = float(hint.get("step_seconds") or 30.0)
    load_s = float(hint.get("load_seconds") or 45.0)
    tail_s = float(hint.get("tail_seconds") or 20.0)
    total_est = load_s + steps_hint * step_s + tail_s
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
        if pos:
            phase, frac, eta = f"queued ({pos} ahead)", 0.0, total_est
        else:
            prog = sampling_progress(since_iso)
            if prog is None:
                # Loading the checkpoint (streamed from disk) and the text encoder.
                frac = min(0.08, 0.08 * elapsed / max(load_s, 1))
                eta = max(total_est - elapsed, tail_s + steps_hint * step_s)
                phase = "loading model"
            else:
                i, n, rate = prog
                rate = rate or step_s
                if i >= n:
                    phase, frac, eta = "decoding", 0.9, tail_s
                else:
                    phase = f"sampling {i}/{n}"
                    frac = 0.1 + 0.8 * i / max(n, 1)
                    eta = (n - i) * rate + tail_s
        key = (phase, int(frac * 100))
        if key != last_phase:
            on_status(phase, frac, eta)
            last_phase = key
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


def fetch_file_output(outputs, dest, key="mmh3_saved", ext=".mmh3", prefix_glob=None):
    """Retrieve a non-video file a custom save node reported in /history.

    MMH3Save puts {"file": "output::sub/name.mmh3", "path": "<absolute>", ...}
    under ui.mmh3_saved; the worker shares the machine with ComfyUI, so the
    absolute path is copied directly. Fallbacks: /view from the output::
    selector, then the newest <COMFYUI_DIR>/output/<prefix_glob>*.mmh3.
    """
    entries = []
    for node_out in outputs.values():
        for e in node_out.get(key, []) or []:
            if isinstance(e, dict):
                entries.append(e)
    for e in entries:
        p = e.get("path")
        if p and os.path.exists(p):
            shutil.copyfile(p, dest)
            return dest
    for e in entries:
        sel = e.get("file") or ""
        if "::" in sel:
            kind, rel = sel.split("::", 1)
            rel = rel.replace("\\", "/")
            sub, name = (rel.rsplit("/", 1) + [""])[:2] if "/" in rel else ("", rel)
            params = {"filename": name, "subfolder": sub, "type": kind}
            with httpx.stream("GET", _url("/view"), params=params, timeout=_TIMEOUT) as r:
                if r.status_code == 200:
                    with open(dest, "wb") as out:
                        for chunk in r.iter_bytes(1 << 20):
                            out.write(chunk)
                    return dest
    if prefix_glob:
        cands = glob.glob(os.path.join(config.COMFYUI_DIR, "output", prefix_glob + "*" + ext))
        if cands:
            newest = max(cands, key=os.path.getmtime)
            shutil.copyfile(newest, dest)
            return dest
    raise ComfyError(f"no {ext} file in comfyui outputs (key {key}): {json.dumps(outputs)[:800]}")


def fetch_texts(outputs):
    """{node_id: [text...]} for output nodes that report ui.text (PreviewAny,
    MMH3Save's saved path, ...)."""
    texts = {}
    for node_id, node_out in outputs.items():
        t = node_out.get("text")
        if isinstance(t, list) and t:
            texts[node_id] = [str(x) for x in t]
    return texts
