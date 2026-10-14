"""Per-job resource sampler: what a job actually cost the box in RAM, VRAM,
GPU and CPU, so we can tell which engines could share it (jobs still run one
at a time; this only measures).

A thread samples every SAMPLE_SECONDS while the job runs. Readings are
system-wide (nvidia-smi, psutil) because the work happens in children and in
the long-lived ComfyUI, not in this process; the baseline taken at claim time
is what was already held (TTS workers, an idle ComfyUI), so peak - baseline is
the job's own share. Separately: the peak RSS of this worker's process tree
(node/chrome/ffmpeg/blender children) and of the ComfyUI server.

With the light lane on (worker/light_lane.py) two jobs can overlap; the
system-wide readings then cover both, so each record carries its lane and
max_concurrent (2 = it shared the box for at least one sample) and the report
can split clean single-job numbers from overlapped ones.

Results go to the log ("stats <id> ...") and one JSON line per job in
worker/job_stats.jsonl. Best effort: a failed reading never touches the job.
"""
import json
import os
import subprocess
import threading
import time

import psutil

import config

SAMPLE_SECONDS = 2
STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_stats.jsonl")
_GB = 1 << 30
_active_lock = threading.Lock()
_active = [0]   # jobs currently inside a JobStats, across lanes


def _gpu():
    """(vram_used_mib, gpu_util_pct) for GPU 0, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
        used, util = (int(x) for x in out.split(","))
        return used, util
    except Exception:
        return None


def _comfy_proc():
    """The process listening on the ComfyUI port, or None."""
    try:
        port = int(config.COMFYUI_URL.rsplit(":", 1)[-1].split("/")[0])
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                return psutil.Process(c.pid)
    except Exception:
        pass
    return None


def _tree_rss(root):
    total = 0
    try:
        procs = [root] + root.children(recursive=True)
    except Exception:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except Exception:
            pass
    return total


class JobStats:
    def __init__(self, job_id, engine, log, lane="main"):
        self.job_id = job_id
        self.engine = engine
        self.lane = lane
        self.max_concurrent = 1
        self.log = log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._me = psutil.Process()
        self._comfy = None
        self._comfy_looked = 0.0
        self.samples = 0
        self.vram = []          # MiB used, system-wide
        self.util = []          # GPU %
        self.ram_used = []      # bytes, system-wide (total - available)
        self.cpu = []           # % of all cores
        self.tree_peak = 0      # bytes, this worker + children
        self.comfy_peak = 0     # bytes, ComfyUI server RSS
        self.baseline = {}

    def _comfy_rss(self):
        now = time.monotonic()
        if self._comfy is None or not self._comfy.is_running():
            if now - self._comfy_looked < 30:
                return 0
            self._comfy_looked = now
            self._comfy = _comfy_proc()
            if self._comfy is None:
                return 0
        try:
            return self._comfy.memory_info().rss
        except Exception:
            self._comfy = None
            return 0

    def _sample(self):
        g = _gpu()
        if g:
            self.vram.append(g[0])
            self.util.append(g[1])
        self.ram_used.append(psutil.virtual_memory().total - psutil.virtual_memory().available)
        self.cpu.append(psutil.cpu_percent(interval=None))
        self.tree_peak = max(self.tree_peak, _tree_rss(self._me))
        self.comfy_peak = max(self.comfy_peak, self._comfy_rss())
        self.max_concurrent = max(self.max_concurrent, _active[0])
        self.samples += 1

    def _loop(self):
        while not self._stop.wait(SAMPLE_SECONDS):
            try:
                self._sample()
            except Exception:
                pass

    def __enter__(self):
        with _active_lock:
            _active[0] += 1
            self.max_concurrent = _active[0]
        try:
            g = _gpu()
            vm = psutil.virtual_memory()
            psutil.cpu_percent(interval=None)  # primes the delta for the first sample
            self.baseline = {"vram_mib": g[0] if g else None,
                             "ram_used_gb": round((vm.total - vm.available) / _GB, 2)}
        except Exception:
            pass
        self._t0 = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self._stop.set()
        self._thread.join(timeout=15)
        with _active_lock:
            _active[0] -= 1
        try:
            self._report("done" if exc_type is None else exc_type.__name__)
        except Exception as e:  # noqa: BLE001 - stats must never fail a job
            self.log(f"stats {self.job_id}: not recorded ({str(e)[:120]})")

    def _report(self, outcome):
        def avg(xs):
            return round(sum(xs) / len(xs), 1) if xs else None
        b_vram = self.baseline.get("vram_mib")
        b_ram = self.baseline.get("ram_used_gb")
        peak_vram = max(self.vram) if self.vram else None
        peak_ram = round(max(self.ram_used) / _GB, 2) if self.ram_used else None
        rec = {
            "job_id": self.job_id, "engine": self.engine, "lane": self.lane,
            "max_concurrent": self.max_concurrent, "outcome": outcome,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.monotonic() - self._t0, 1), "samples": self.samples,
            "vram_baseline_mib": b_vram, "vram_peak_mib": peak_vram,
            "vram_job_peak_mib": (peak_vram - b_vram) if peak_vram is not None and b_vram is not None else None,
            "gpu_util_avg": avg(self.util), "gpu_util_peak": max(self.util) if self.util else None,
            "ram_baseline_gb": b_ram, "ram_peak_gb": peak_ram,
            "ram_job_peak_gb": round(peak_ram - b_ram, 2) if peak_ram is not None and b_ram is not None else None,
            "worker_tree_peak_gb": round(self.tree_peak / _GB, 2),
            "comfyui_peak_gb": round(self.comfy_peak / _GB, 2) if self.comfy_peak else None,
            "cpu_avg": avg(self.cpu), "cpu_peak": max(self.cpu) if self.cpu else None,
        }
        with open(STATS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.log(f"stats {self.job_id} {self.engine} {rec['seconds']}s"
                 f"{' (overlapped)' if self.max_concurrent > 1 else ''}: "
                 f"vram peak {peak_vram} MiB (+{rec['vram_job_peak_mib']}), "
                 f"gpu avg {rec['gpu_util_avg']}% peak {rec['gpu_util_peak']}%, "
                 f"ram peak {peak_ram} GB (+{rec['ram_job_peak_gb']}), "
                 f"tree {rec['worker_tree_peak_gb']} GB, comfyui {rec['comfyui_peak_gb']} GB, "
                 f"cpu avg {rec['cpu_avg']}%")
