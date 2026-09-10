"""Provenance for the H3 latent path: what exactly produced an upscaled clip.

outputs/<jid>-upscale.json is the record the brief asks for (source render,
latent packet, recipe, upscaler weights, refine settings, seed, ComfyUI/node
versions), written beside the mp4 like the turntable manifest. Pure helpers
here (hashing, git heads, packet reading) plus a VRAM sampler that polls
/system_stats while a prompt runs, which is the only peak-VRAM signal the
worker can get over HTTP.
"""
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import zipfile


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(directory):
    """Commit hash of a checkout (reads .git/HEAD, follows refs), or None."""
    try:
        git = os.path.join(directory, ".git")
        if os.path.isfile(git):   # worktree pointer
            with open(git, encoding="utf-8") as f:
                line = f.read().strip()
            if line.startswith("gitdir:"):
                git = line.split(":", 1)[1].strip()
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        p = os.path.join(git, ref)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        packed = os.path.join(git, "packed-refs")
        if os.path.exists(packed):
            with open(packed, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
    except OSError:
        pass
    return None


def node_revisions(comfy_dir):
    from videogen import graphs_h3
    out = {}
    for label, sub in (("mmh3_media", graphs_h3.MMH3_NODE_DIR), ("latent_upscaler", graphs_h3.UPSCALER_NODE_DIR)):
        d = os.path.join(comfy_dir, "custom_nodes", sub)
        out[label] = {"dir": d, "present": os.path.isdir(d), "git_head": git_head(d) if os.path.isdir(d) else None}
    return out


def weight_sha256(comfy_dir, name):
    from videogen import graphs_h3
    p = os.path.join(comfy_dir, "models", graphs_h3.UPSCALER_FOLDER, name)
    if not os.path.exists(p):
        return None
    return sha256(p)


def packet_manifest(mmh3_path):
    """packet.json of a .mmh3 archive (ZIP), or raise."""
    with zipfile.ZipFile(mmh3_path) as z:
        with z.open("packet.json") as f:
            return json.load(f)


def is_packet(path):
    try:
        with zipfile.ZipFile(path) as z:
            return "packet.json" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def packet_summary(manifest):
    """The few fields worth copying into the provenance record."""
    ext = ((manifest.get("extensions") or {}).get("minimax_h3") or {})
    gen = manifest.get("generation") or {}
    return {"id": manifest.get("id"), "name": manifest.get("name"), "schema_version": manifest.get("schema_version"),
            "created_at": manifest.get("created_at"), "generation": gen if isinstance(gen, dict) else None,
            "h3": {k: v for k, v in ext.items() if k != "latent"} if isinstance(ext, dict) else None,
            "history_len": len(manifest.get("history") or [])}


def packet_geometry(manifest):
    """(width, height, frames) recorded for the packet's generation, best effort."""
    gen = manifest.get("generation") or {}
    for src in (gen, gen.get("settings") or {}, (manifest.get("extensions") or {}).get("minimax_h3") or {}):
        if not isinstance(src, dict):
            continue
        w, h, n = src.get("width"), src.get("height"), src.get("frames")
        if w and h and n:
            return int(w), int(h), int(n)
    return None


def packet_last_process(manifest):
    ext = (manifest.get("extensions") or {}).get("mmh3_media") or {}
    lp = ext.get("last_process") if isinstance(ext, dict) else None
    if lp:
        return lp
    hist = manifest.get("history") or []
    return hist[-1] if hist else None


def nvidia_smi_used_mb():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - telemetry only
        return None


class VramSampler:
    """Poll /system_stats every `interval` s on a thread; peak used VRAM is
    total minus the lowest free value seen. Also samples nvidia-smi when on
    PATH (process-agnostic, catches what ComfyUI's own counter misses)."""

    def __init__(self, stats_fn, interval=2.0):
        self.stats_fn, self.interval = stats_fn, interval
        self.total = None
        self.min_free = None
        self.min_torch_free = None
        self.max_smi_used = None
        self.samples = 0
        self._stop = threading.Event()
        self._t = None

    def _sample(self):
        try:
            d = (self.stats_fn().get("devices") or [{}])[0]
            total, free, tfree = d.get("vram_total"), d.get("vram_free"), d.get("torch_vram_free")
            if total:
                self.total = int(total) // (1 << 20)
            if free is not None:
                self.min_free = int(free) // (1 << 20) if self.min_free is None else min(self.min_free, int(free) // (1 << 20))
            if tfree is not None:
                v = int(tfree) // (1 << 20)
                self.min_torch_free = v if self.min_torch_free is None else min(self.min_torch_free, v)
            self.samples += 1
        except Exception:  # noqa: BLE001
            pass
        used = nvidia_smi_used_mb()
        if used is not None:
            self.max_smi_used = used if self.max_smi_used is None else max(self.max_smi_used, used)

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self):
        self._sample()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=self.interval + 2)
        self._sample()
        return self.result()

    def result(self):
        return {"vram_total_mb": self.total, "min_free_mb": self.min_free, "min_torch_free_mb": self.min_torch_free,
                "peak_used_estimate_mb": (self.total - self.min_free) if self.total and self.min_free is not None else None,
                "nvidia_smi_peak_used_mb": self.max_smi_used, "samples": self.samples,
                "method": f"/system_stats poll every {self.interval:g} s (+ nvidia-smi when on PATH)"}


class Timer:
    def __init__(self):
        self.marks = {}
        self._t0 = time.monotonic()
        self._last = self._t0

    def lap(self, name):
        now = time.monotonic()
        self.marks[name] = round(self.marks.get(name, 0.0) + now - self._last, 1)
        self._last = now

    def total(self):
        return round(time.monotonic() - self._t0, 1)


def build_upscale_json(*, job_id, meta, u, source, latent, comfy_stats, comfy_dir, timing, vram, fidelity,
                       outputs, refine_actual=None, tile_plan=None, warnings=None):
    """Assemble outputs/<jid>-upscale.json. Every input is plain data."""
    from videogen import graphs_h3
    sysinfo = (comfy_stats or {}).get("system") or {}
    dev = ((comfy_stats or {}).get("devices") or [{}])[0]
    refine = {"denoise": u["denoise"], "steps_override": u["steps_override"], "seed": u["seed"],
              "source": "worker params"}
    if refine_actual:
        refine.update({k: refine_actual.get(k) for k in ("task_family", "steps", "video_shift", "audio_shift",
                                                          "sampler_name", "scheduler", "denoise") if k in refine_actual})
        refine["source"] = refine_actual.get("_source", "MMH3H3UpscaleRefineSampling probe")
    return {"job_id": job_id, "method": graphs_h3.METHOD, "variant": meta["variant"], "recipe": meta["recipe"],
            "operation": meta["operation"], "source": source, "latent": latent,
            "upscaler": {"class": graphs_h3.UPSCALER_CLASS, "weight": meta["upscaler"],
                         "weight_sha256": weight_sha256(comfy_dir, meta["upscaler"]) if comfy_dir else None,
                         "precision": "bf16", "align": graphs_h3.ALIGN, "mode": graphs_h3.UPSCALER_MODE_DIMS},
            "target": {"requested": meta["requested"], "refine": meta["refine"], "crop": meta["crop"],
                       "scale": meta["scale"]},
            "refine": refine, "tiles": meta.get("tiles"), "tile_plan": tile_plan,
            "force_unload": u["force_unload"], "attention": u["attention"], "fp16_accumulation": u["fp16_accumulation"],
            "comfyui": {"version": sysinfo.get("comfyui_version"), "python": sysinfo.get("python_version"),
                        "pytorch": sysinfo.get("pytorch_version"), "device": dev.get("name"),
                        "vram_total_mb": (int(dev["vram_total"]) // (1 << 20)) if dev.get("vram_total") else None},
            "nodes": node_revisions(comfy_dir) if comfy_dir else None,
            "timing_s": timing, "vram_mb": vram, "fidelity": fidelity, "outputs": outputs,
            "warnings": warnings or [], "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
