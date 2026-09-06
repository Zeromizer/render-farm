"""H3 Studio: a local web UI for the video_gen engine.

  worker\\start-studio.bat  ->  http://127.0.0.1:8790

- Generate: prompt MiniMax H3 (t2v / i2v / r2v) with every knob the runner
  accepts, plus the optional resize pass (lanczos default, SeedVR2 opt-in).
- Library: every result with a thumbnail, inline player, its full params,
  and actions: upscale (lanczos / SeedVR2 + blend), 60 fps (RIFE), reuse,
  download, delete, cancel.
- Turntable: the anchored 2 x 180-degree car flow, filed as a single farm job
  (video_gen mode "turntable", run by the worker; studio/turntable.py).

Jobs go through the normal farm queue (farm_render_jobs, engine video_gen),
so the worker, TTS pause and ComfyUI lifecycle are untouched; the studio
only inserts rows, polls them and downloads outputs. The 60 fps action (RIFE)
runs in a thread inside this process; everything else is a farm job. Binds 127.0.0.1 only: the
service key never leaves the PC. Stdlib http.server: no new dependencies.
"""
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.dirname(HERE)
sys.path.insert(0, WORKER)

import config  # noqa: E402
import db  # noqa: E402
from videogen import graphs, segments  # noqa: E402
from studio import post, turntable  # noqa: E402

ROOT = config.STUDIO_DIR
DIRS = {k: os.path.join(ROOT, k) for k in ("items", "videos", "thumbs", "uploads", "work", "tmp")}
ITEMS = {}
LOCK = threading.RLock()
_IMAGE = (".png", ".jpg", ".jpeg", ".webp")
_VIDEO = (".mp4", ".mov", ".webm", ".mkv")
_AUDIO = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
_TERMINAL = ("done", "failed", "canceled")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- items ----
def _load_items():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    for f in os.listdir(DIRS["items"]):
        if f.endswith(".json"):
            try:
                it = json.load(open(os.path.join(DIRS["items"], f), encoding="utf-8"))
                ITEMS[it["id"]] = it
            except Exception as e:  # noqa: BLE001
                log(f"skip {f}: {e}")
    # Anything that was mid-flight inside this process when it last died cannot resume.
    for it in ITEMS.values():
        if it["kind"] == "interpolate" and it["status"] not in _TERMINAL or (
                it["kind"] == "turntable" and not it.get("job_id") and it["status"] not in _TERMINAL):
            it["status"], it["error"] = "failed", "studio restarted while this was running"
            _save(it)


def _save(it):
    with LOCK:
        tmp = os.path.join(DIRS["items"], it["id"] + ".json.tmp")
        json.dump(it, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, os.path.join(DIRS["items"], it["id"] + ".json"))


def _new_item(kind, **fields):
    it = {"id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4], "kind": kind, "status": "pending",
          "phase": "queued", "progress": 0, "error": None, "created": time.time(), "finished": None,
          "job_id": None, "jobs": [], "video": None, "thumb": None, "info": None, "remote": None, "label": "",
          "prompt": "", "params": {}, "parent": None, "pieces": [], "log": []}
    it.update(fields)
    with LOCK:
        ITEMS[it["id"]] = it
    _save(it)
    return it


def _set(it, **fields):
    with LOCK:
        it.update(fields)
    _save(it)


def _note(it, msg):
    log(f"{it['id']}: {msg}")
    with LOCK:
        it["log"] = (it.get("log") or [])[-60:] + [f"{time.strftime('%H:%M:%S')} {msg}"]
    _save(it)


def _finalize(it, local_video, remote=None):
    """Move a finished video into the library and fill in its metadata."""
    dest = os.path.join(DIRS["videos"], it["id"] + ".mp4")
    if os.path.abspath(local_video) != os.path.abspath(dest):
        shutil.copyfile(local_video, dest)
    inf = post.info(dest)
    thumb = os.path.join(DIRS["thumbs"], it["id"] + ".jpg")
    post.thumbnail(dest, thumb, t=min(1.0, max(0.0, inf["duration"] / 4)))
    _set(it, status="done", phase="done", progress=100, finished=time.time(), video=f"videos/{it['id']}.mp4",
         thumb=f"thumbs/{it['id']}.jpg", info=inf, remote=remote or it.get("remote"))


# ------------------------------------------------------------ farm queue ----
def submit_job(vg, priority=50, timeout_minutes=None):
    row = {"status": "pending", "engine": "video_gen", "repo_url": "-", "git_ref": "main",
           "params": {"video_gen": vg}, "priority": int(priority),
           "timeout_minutes": int(timeout_minutes or config.VIDEO_GEN_DEFAULT_TIMEOUT_MINUTES)}
    return db.sb.table("farm_render_jobs").insert(row).execute().data[0]["id"]


def job_rows(ids):
    if not ids:
        return {}
    rows = (db.sb.table("farm_render_jobs").select("id,status,phase,progress,error,output_path")
            .in_("id", list(ids)).execute().data)
    return {r["id"]: r for r in rows}


def download_output(remote, dest):
    data = db.sb.storage.from_(config.BUCKET).download(remote)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def upload_input(local):
    """Content-named copy of a local file in the renders bucket, for job inputs."""
    h = hashlib.sha256(open(local, "rb").read()).hexdigest()[:12]
    base = re.sub(r"^[0-9a-f]{12}_", "", os.path.basename(local))   # uploads/ files already carry the hash
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[-60:]
    remote = f"inputs/studio/{h}_{safe}"
    ctype = mimetypes.guess_type(local)[0] or "application/octet-stream"
    db.upload_file(remote, local, ctype)
    return {"bucket": config.BUCKET, "path": remote}


def _fetch_sidecars(it, jid):
    """The turntable runner leaves its 24 fps pieces next to the result; pull the ones that exist."""
    pieces = []
    for name in [f"piece{i}" for i in range(1, 9)] + ["joined24"]:
        rel = f"videos/{it['id']}_{name}.mp4"
        try:
            download_output(f"outputs/{jid}-{name}.mp4", os.path.join(ROOT, rel))
            pieces.append(rel)
        except Exception:  # noqa: BLE001 - not there
            if name != "joined24":
                break
    _set(it, pieces=pieces)


def _poll_farm():
    """Track items that are farm jobs; download finished outputs into the library."""
    while True:
        try:
            with LOCK:
                watch = {it["job_id"]: it for it in ITEMS.values()
                         if it.get("job_id") and it["kind"] in ("generate", "upscale", "turntable")
                         and it["status"] not in _TERMINAL}
            if watch:
                for jid, row in job_rows(watch).items():
                    it = watch[jid]
                    st = row["status"]
                    if st == "done":
                        _note(it, "downloading")
                        tmp = os.path.join(DIRS["tmp"], f"{it['id']}.mp4")
                        download_output(row["output_path"], tmp)
                        if it["kind"] == "turntable":
                            _fetch_sidecars(it, jid)
                        _finalize(it, tmp, remote=row["output_path"])
                        os.remove(tmp)
                    elif st in ("failed", "canceled"):
                        _set(it, status=st, phase=st, error=row.get("error"), finished=time.time())
                    else:
                        _set(it, status=st, phase=row.get("phase") or st, progress=row.get("progress") or 0)
        except Exception as e:  # noqa: BLE001
            log(f"poll: {e}")
        time.sleep(4)


# ------------------------------------------------------------ local flows ----
def _run_interpolate(it, src, fps, shorter_size):
    try:
        _set(it, status="processing", phase=f"RIFE to {fps} fps", progress=10)
        out = os.path.join(DIRS["tmp"], f"{it['id']}.mp4")
        post.interpolate(src, out, fps=fps, shorter_size=shorter_size, tmp_root=DIRS["tmp"], log=lambda m: _note(it, m))
        _finalize(it, out)
        os.remove(out)
    except Exception as e:  # noqa: BLE001
        _set(it, status="failed", phase="failed", error=str(e), finished=time.time())
        log(traceback.format_exc())


# ------------------------------------------------------------------ http ----
def _ref(x):
    return {"bucket": x["bucket"], "path": x["path"]} if isinstance(x, dict) and x.get("path") else None


def _build_vg(p):
    vg = {"prompt": (p.get("prompt") or "").strip()}
    if p.get("mode") and p["mode"] != "auto":
        vg["mode"] = p["mode"]
    for k in ("duration_s", "resolution", "ratio", "seed", "steps", "ref_image_size"):
        if p.get(k) not in (None, ""):
            vg[k] = p[k]
    if "turbo" in p:
        vg["turbo"] = bool(p["turbo"])
    for k in ("first_frame", "last_frame"):
        if _ref(p.get(k)):
            vg[k] = _ref(p[k])
    for k in ("ref_images", "ref_videos", "ref_audios"):
        refs = [_ref(x) for x in (p.get(k) or []) if _ref(x)]
        if refs:
            vg[k] = refs
    up = _build_upscale(p.get("upscale"))
    if up:
        vg["upscale"] = up
    return vg


def _build_upscale(u):
    if not u or u.get("method") in (None, "", "none"):
        return None
    out = {"method": u["method"]}
    if u.get("shorter_size"):
        out["shorter_size"] = int(u["shorter_size"])
    elif u.get("factor"):
        out["factor"] = float(u["factor"])
    if u["method"] == "seedvr2":
        for k in ("blend", "color_correction", "frames_per_chunk", "temporal_overlap", "seed"):
            if u.get(k) not in (None, ""):
                out[k] = u[k]
    return out


def _item_remote(it):
    """Bucket path for an item's video, uploading local-only results on demand."""
    if it.get("remote"):
        return {"bucket": config.BUCKET, "path": it["remote"]}
    if not it.get("video"):
        raise ValueError("item has no video yet")
    ref = upload_input(os.path.join(ROOT, it["video"]))
    _set(it, remote=ref["path"])
    return ref


class Handler(BaseHTTPRequestHandler):
    server_version = "H3Studio/1.0"

    def log_message(self, fmt, *args):  # quiet
        if self.path.startswith("/api/") and not self.path.startswith("/api/state"):
            log(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}")

    # helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _file(self, path, ctype=None):
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if (b and a) else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    # routes
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
        if u.path == "/api/state":
            with LOCK:
                items = sorted(ITEMS.values(), key=lambda x: x["created"], reverse=True)
                items = [{k: v for k, v in it.items() if k != "log"} | {"log": (it.get("log") or [])[-8:]} for it in items]
            return self._json({"items": items, "config": {
                "resolutions": sorted(graphs.SHORT_EDGE), "ratios": sorted(graphs.RATIOS),
                "colors": list(graphs.UPSCALE_COLOR_METHODS), "studio_dir": ROOT,
                "rife": os.path.exists(os.path.join(config.RIFE_DIR, "rife-ncnn-vulkan.exe")),
                "turntable_defaults": turntable.DEFAULTS}})
        if u.path.startswith("/media/"):
            rel = unquote(u.path[len("/media/"):]).replace("\\", "/")
            full = os.path.normpath(os.path.join(ROOT, rel))
            if not full.startswith(os.path.normpath(ROOT)):
                return self._json({"error": "forbidden"}, 403)
            return self._file(full)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/upload":
                return self._upload()
            p = json.loads(self._body() or b"{}")
            fn = {"/api/generate": self._generate, "/api/upscale": self._upscale, "/api/interpolate": self._interpolate,
                  "/api/turntable": self._turntable, "/api/cancel": self._cancel, "/api/delete": self._delete}.get(u.path)
            if not fn:
                return self._json({"error": "not found"}, 404)
            return self._json(fn(p))
        except (ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc())
            return self._json({"error": str(e)}, 500)

    def _upload(self):
        name = unquote(self.headers.get("X-Filename") or "upload.bin")
        data = self._body()
        if not data:
            return self._json({"error": "empty upload"}, 400)
        ext = os.path.splitext(name)[1].lower()
        kind = "image" if ext in _IMAGE else "video" if ext in _VIDEO else "audio" if ext in _AUDIO else None
        if not kind:
            return self._json({"error": f"unsupported file type {ext}"}, 400)
        h = hashlib.sha256(data).hexdigest()[:12]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[-60:]
        local = os.path.join(DIRS["uploads"], f"{h}_{safe}")
        if not os.path.exists(local):
            with open(local, "wb") as f:
                f.write(data)
        ref = upload_input(local)
        return self._json({**ref, "kind": kind, "name": name, "local": local, "url": f"/media/uploads/{os.path.basename(local)}"})

    def _generate(self, p):
        vg = _build_vg(p)
        if not vg["prompt"]:
            raise ValueError("prompt is required")
        it = _new_item("generate", label=p.get("label") or "", prompt=vg["prompt"], params=vg,
                       priority=int(p.get("priority") or 50))
        jid = submit_job(vg, priority=it["priority"], timeout_minutes=p.get("timeout_minutes"))
        _set(it, job_id=jid, jobs=[{"id": jid, "label": "generate", "status": "pending"}])
        return {"item": it}

    def _upscale(self, p):
        src = ITEMS[p["item_id"]]
        up = _build_upscale(p.get("upscale"))
        if not up:
            raise ValueError("choose an upscale method")
        vg = {"mode": "upscale", "source": _item_remote(src), "upscale": up}
        it = _new_item("upscale", label=p.get("label") or f"{up['method']} of {src.get('label') or src['id']}",
                       prompt=src.get("prompt", ""), params=vg, parent=src["id"], priority=int(p.get("priority") or 50))
        jid = submit_job(vg, priority=it["priority"])
        _set(it, job_id=jid, jobs=[{"id": jid, "label": "upscale", "status": "pending"}])
        return {"item": it}

    def _interpolate(self, p):
        src = ITEMS[p["item_id"]]
        if not src.get("video"):
            raise ValueError("item has no video yet")
        fps = int(p.get("fps") or 60)
        short = int(p["shorter_size"]) if p.get("shorter_size") else None
        it = _new_item("interpolate", label=p.get("label") or f"{fps} fps of {src.get('label') or src['id']}",
                       prompt=src.get("prompt", ""), params={"fps": fps, "shorter_size": short}, parent=src["id"])
        threading.Thread(target=_run_interpolate, args=(it, os.path.join(ROOT, src["video"]), fps, short), daemon=True).start()
        return {"item": it}

    def _turntable(self, p):
        for k in ("front", "rear"):
            if not (p.get(k) or {}).get("path"):
                raise ValueError(f"upload the {k} photo first")
        if not (p.get("car") or "").strip():
            raise ValueError("describe the car in a few words")
        tt = {"front": _ref(p["front"]), "rear": _ref(p["rear"]), "car": p["car"].strip(),
              "details": (p.get("details") or "").strip()}
        for k in ("resolution", "ratio", "seconds_per_half", "seed", "fps", "shorter_size", "density"):
            if p.get(k) not in (None, ""):
                tt[k] = p[k]
        vg = {"mode": "turntable", "turntable": tt}
        it = _new_item("turntable", label=p.get("label") or f"turntable: {tt['car'][:40]}", prompt=tt["car"],
                       params=vg, priority=int(p.get("priority") or 50))
        jid = submit_job(vg, priority=it["priority"], timeout_minutes=p.get("timeout_minutes") or 150)
        _set(it, job_id=jid, jobs=[{"id": jid, "label": "turntable", "status": "pending"}])
        return {"item": it}

    def _cancel(self, p):
        it = ITEMS[p["item_id"]]
        with LOCK:
            it["cancel"] = True
        for j in it.get("jobs") or []:
            if j.get("status") not in _TERMINAL:
                db.update_job(j["id"], {"cancel_requested": True})
        if it["kind"] in ("generate", "upscale"):
            _set(it, phase="cancel requested")
        return {"ok": True}

    def _delete(self, p):
        it = ITEMS.pop(p["item_id"], None)
        if not it:
            return {"ok": True}
        for rel in [it.get("video"), it.get("thumb")] + list(it.get("pieces") or []):
            if rel:
                try:
                    os.remove(os.path.join(ROOT, rel))
                except OSError:
                    pass
        try:
            os.remove(os.path.join(DIRS["items"], it["id"] + ".json"))
        except OSError:
            pass
        return {"ok": True}


def main():
    _load_items()
    threading.Thread(target=_poll_farm, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", config.STUDIO_PORT), Handler)
    srv.daemon_threads = True
    log(f"H3 Studio on http://127.0.0.1:{config.STUDIO_PORT}  library: {ROOT}  items: {len(ITEMS)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
