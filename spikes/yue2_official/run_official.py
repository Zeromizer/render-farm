"""Run m-a-p's OFFICIAL YuE2 instrumental skill on the render PC, unchanged, and report.

Phase 1/2 of the "replicate the authors first" rule: skill/ is the upstream bundle
(skills/yue2-music/instrumental at the commit in UPSTREAM_COMMIT), byte-identical
(skill/scripts/verify_bundle.py checks its manifest). This driver only installs nothing,
calls their own scripts as documented, and collects results. It does not patch them.

params.args = [<job json path, repo-relative>, <out dir>]. Job JSON:
  {"run_id": "p1",
   "checks": true,                       # their CPU tests + doctor before any GPU work
   "takes": [{"name": "smoke", "kind": "smoke"},          # their documented smoke command
             {"name": "funk_11", "kind": "style", "style": "Instrumental, ...", "seed": 11}]}

Deviations from their setup.md, all forced by this box (logged in results.json):
  * Windows, not Linux; the farm's cached venv (requirements.txt here) instead of their
    `setup_runtime.py install` (same pinned wheel + sha; torch 2.10.0 from the cu128 index,
    since PyPI's Windows torch is CPU-only).
  * 16 GB RTX 4080 SUPER, not a 24 GB card. Their pipeline caps itself at
    min(budget-2, total-2) GiB, so it runs at ~14 GiB here; their doctor's 24 GB gate is
    recorded, not enforced.
  * PYTHONUTF8=1 (their scripts read text with the locale encoding; cp1252 breaks one test).
  * ComfyUI's loaded models are freed first (POST /free) so the card is empty.

Full work dirs stay on the PC under ~/yue2-official/runs/<run_id>/; <out dir> gets a slim copy
(audio, scores, prompt, summaries, log) for the laptop.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.join(HERE, "skill")
SCRIPTS = os.path.join(SKILL, "scripts")
HOME = os.path.join(os.path.expanduser("~"), "yue2-official")
MODELS = os.environ.get("YUE2_OFFICIAL_MODELS", os.path.join(HOME, "models"))
COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")

T0 = time.monotonic()
_state = {"phase": "start", "peak": {}, "vram": None}
_lock = threading.Lock()


def log(msg):
    print(msg, flush=True)


def elapsed():
    return int(time.monotonic() - T0)


def nvidia_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        used, total = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def monitor():
    """Sample VRAM every 3 s (peak per phase) and print a heartbeat every 15 s, so the
    farm's cancel/timeout checks always have a line to act on."""
    last = 0.0
    while True:
        v = nvidia_used()
        with _lock:
            phase = _state["phase"]
            if v:
                _state["vram"] = v
                _state["peak"][phase] = max(_state["peak"].get(phase, 0), v[0])
        if time.monotonic() - last >= 15:
            last = time.monotonic()
            log(f"alive t={elapsed()}s phase={phase} vram={v[0] if v else '?'}/{v[1] if v else '?'} MiB")
        time.sleep(3)


def set_phase(name):
    with _lock:
        _state["phase"] = name


def stream(cmd, logfile=None, cwd=None):
    """Run a child with stderr merged; relay its lines (tqdm's \\r updates throttled to one
    every 5 s) and return (exit code, seconds)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    t = time.monotonic()
    log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=cwd or SKILL, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        return _relay(p, logfile, t)
    finally:
        if p.poll() is None:
            p.kill()


def _relay(p, logfile, t):
    fh = open(logfile, "a", encoding="utf-8") if logfile else None
    buf, last_cr = b"", 0.0
    while True:
        chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            m = re.search(rb"[\r\n]", buf)
            if not m:
                break
            line, sep, buf = buf[:m.start()], buf[m.start():m.start() + 1], buf[m.end():]
            text = line.decode("utf-8", "replace").rstrip()
            if not text:
                continue
            if fh:
                fh.write(text + "\n")
            if sep == b"\r":
                if time.monotonic() - last_cr < 5:
                    continue
                last_cr = time.monotonic()
            log("  | " + text)
    if buf.strip():
        text = buf.decode("utf-8", "replace").rstrip()
        log("  | " + text)
        if fh:
            fh.write(text + "\n")
    code = p.wait()
    if fh:
        fh.close()
    return code, round(time.monotonic() - t, 1)


PROBE = r"""
import json, torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
d = torch.device("cuda")
q = torch.randn(1, 16, 1, 128, device=d, dtype=torch.bfloat16)
k = torch.randn(1, 8, 64, 128, device=d, dtype=torch.bfloat16)
mask = torch.ones(1, 1, 1, 64, device=d, dtype=torch.bool)
out = {}
try:
    qq = torch.randn(1, 16, 128, device=d, dtype=torch.bfloat16)
    kk = torch.randn(64, 8, 128, device=d, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1], device=d, dtype=torch.int32); cu_k = torch.tensor([0, 64], device=d, dtype=torch.int32)
    torch.ops.aten._flash_attention_forward(qq[0:1].reshape(1, 16, 128), kk, kk, cu_q, cu_k, 1, 64, 0.0, False, False,
                                            seqused_k=torch.tensor([64], device=d, dtype=torch.int32))
    torch.cuda.synchronize(); out["flash"] = "ok"
except Exception as e:
    out["flash"] = f"{type(e).__name__}: {e}"[:200]
try:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        F.scaled_dot_product_attention(q, k, k, attn_mask=mask, is_causal=False, enable_gqa=True)
    torch.cuda.synchronize(); out["cudnn"] = "ok"
except Exception as e:
    out["cudnn"] = f"{type(e).__name__}: {e}"[:200]
print("PROBE " + json.dumps(out))
"""


def probe_attention(py):
    """Which of GraphAR's backends actually run here, in the authors' order."""
    r = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=300,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    found = {}
    for line in r.stdout.splitlines():
        if line.startswith("PROBE "):
            found = json.loads(line[6:])
    if not found:
        found = {"error": (r.stderr or r.stdout)[-500:]}
    choice = "flash" if found.get("flash") == "ok" else "cudnn" if found.get("cudnn") == "ok" else "sdpa"
    return choice, found


def comfy_free():
    """Empty the card: wait for any running ComfyUI prompt (a live music bed), then /free."""
    try:
        urllib.request.urlopen(COMFY + "/system_stats", timeout=5).read()
    except Exception:
        return "comfyui not running"
    waited = 0
    while True:
        try:
            q = json.loads(urllib.request.urlopen(COMFY + "/queue", timeout=10).read())
        except Exception as e:
            return f"queue read failed: {e}"
        if not q.get("queue_running") and not q.get("queue_pending"):
            break
        if waited >= 1200:
            return "gave up waiting for ComfyUI's queue after 20 min; not freed"
        log(f"ComfyUI busy ({len(q.get('queue_running', []))} running, {len(q.get('queue_pending', []))} pending); waiting")
        time.sleep(15)
        waited += 15
    req = urllib.request.Request(COMFY + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=30).read()
    time.sleep(5)
    return f"freed after waiting {waited}s"


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def copy_if(src, dst):
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    return False


def collect(work, slim):
    """Pull the facts and the small files out of one take's work dir."""
    gen = os.path.join(work, "generation")
    summary = read_json(os.path.join(gen, "summary.json"))
    native = read_json(os.path.join(gen, "native", "result.json"))
    transfer = read_json(os.path.join(work, "conversion", "transfer.json"))
    failure = (read_json(os.path.join(gen, "failure.json")) or read_json(os.path.join(work, "planning", "failure.json")))
    for src, name in ((os.path.join(gen, "native", "audio.flac"), "audio.flac"),
                      (os.path.join(gen, "listening.mp3"), "listening.mp3"),
                      (os.path.join(work, "planning", "score.abc"), "score_model_plan.abc"),
                      (os.path.join(work, "conversion", "score.abc"), "score_instrumental.abc"),
                      (os.path.join(work, "conversion", "transfer.json"), "transfer.json"),
                      (os.path.join(work, "prepared", "prompt.txt"), "prompt.txt"),
                      (os.path.join(work, "workflow.json"), "workflow.json"),
                      (os.path.join(gen, "summary.json"), "summary.json"),
                      (os.path.join(gen, "native", "result.json"), "native_result.json"),
                      (os.path.join(gen, "native", "config.json"), "native_config.json"),
                      (os.path.join(work, "planning", "provenance.json"), "planning_provenance.json")):
        copy_if(src, os.path.join(slim, name))
    return {
        "status": (summary or {}).get("status") or (native or {}).get("status"),
        "audio_seconds": (summary or native or {}).get("audio_seconds"),
        "nominal_seconds": (transfer or {}).get("nominal_seconds"),
        "truncated": (summary or native or {}).get("truncated"),
        "timing": (native or {}).get("timing"),
        "peak": (summary or {}).get("peak"), "rms": (summary or {}).get("rms"),
        "failure": failure,
    }


def main():
    # The farm pipes stdout with the locale codec (cp1252); tqdm's block characters crash it.
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    job_path, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(os.getcwd(), job_path) if not os.path.isabs(job_path) else job_path, encoding="utf-8") as f:
        job = json.load(f)
    run_id = job["run_id"]
    runs = os.path.join(HOME, "runs", run_id)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    os.makedirs(runs, exist_ok=True)
    threading.Thread(target=monitor, daemon=True).start()
    py = sys.executable
    results = {"run_id": run_id, "python": sys.version, "skill_commit": open(os.path.join(HERE, "UPSTREAM_COMMIT")).read().strip(),
               "models_root": MODELS, "pc_runs_dir": runs, "deviations": [
                   "Windows + farm venv (torch 2.10.0+cu128) instead of setup_runtime.py install on Linux",
                   "16 GB GPU: pipeline self-caps at ~14 GiB; doctor's 24 GB gate recorded only",
                   "PYTHONUTF8=1", "ComfyUI models freed before GPU work"], "steps": {}, "takes": []}

    def save():
        results["vram_peak_mib"] = dict(_state["peak"])
        results["elapsed_s"] = elapsed()
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        v = subprocess.run([py, "-c", "import importlib.metadata as m, json\n"
                            "def ver(p):\n try: return m.version(p)\n except Exception: return None\n"
                            "print(json.dumps({p: ver(p) for p in ('yue2-infer','torch','transformers','numpy','huggingface-hub')}))"],
                           capture_output=True, text=True, timeout=120)
        results["versions"] = v.stdout.strip() or v.stderr.strip()[-500:]
        log(f"versions: {results['versions']}")

        if job.get("checks", True):
            set_phase("cpu-checks")
            for name, cmd in (("verify_bundle", [py, "-B", os.path.join(SCRIPTS, "verify_bundle.py")]),
                              ("test_tools", [py, "-B", os.path.join(SCRIPTS, "test_tools.py")])):
                code, secs = stream(cmd, os.path.join(out, f"{name}.log"))
                results["steps"][name] = {"exit": code, "seconds": secs}
            save()

        set_phase("models")
        log("PROGRESS 2")
        if not os.path.isfile(os.path.join(MODELS, "models-verified.json")):
            code, secs = stream([py, "-B", os.path.join(SCRIPTS, "setup_runtime.py"), "models", "--output", MODELS],
                                os.path.join(out, "models.log"))
            results["steps"]["models"] = {"exit": code, "seconds": secs}
            if code:
                raise RuntimeError("model download/verification failed; see models.log")
        else:
            results["steps"]["models"] = {"exit": 0, "cached": True}
        save()

        set_phase("comfy-free")
        results["steps"]["comfy_free"] = comfy_free()
        log(f"comfy: {results['steps']['comfy_free']}")
        results["vram_before_takes"] = nvidia_used()

        set_phase("attention-probe")
        choice, found = probe_attention(py)
        results["attention"] = {"probe": found, "used": choice}
        log(f"attention probe: {found} -> {choice}")
        if choice != "flash":
            # Their auto-pick would choose flash (the op is registered) and crash; pass the
            # first working backend in their order through the sitecustomize shim.
            os.environ["YUE2_ATTENTION"] = choice
            os.environ["PYTHONPATH"] = os.path.join(HERE, "shim") + os.pathsep + os.environ.get("PYTHONPATH", "")
            results["deviations"].append(f"GraphAR attention forced to '{choice}' (Windows torch has no FlashAttention kernel)")
        save()

        set_phase("doctor")
        code, secs = stream([py, "-B", os.path.join(SCRIPTS, "setup_runtime.py"), "doctor", "--models-root", MODELS,
                             "--output", os.path.join(out, "doctor.json")], os.path.join(out, "doctor.log"))
        results["steps"]["doctor"] = {"exit": code, "seconds": secs}
        save()

        takes = job["takes"]
        for i, take in enumerate(takes):
            name = take["name"]
            set_phase(name)
            log(f"PROGRESS {5 + int(90 * i / len(takes))}")
            log(f"=== take {i + 1}/{len(takes)}: {name}")
            work = os.path.join(runs, name)
            if os.path.exists(work):
                shutil.rmtree(work)
            slim = os.path.join(out, "takes", name)
            os.makedirs(slim, exist_ok=True)
            cmd = [py, "-B", os.path.join(SCRIPTS, "instrumental.py"), "run"]
            if take["kind"] == "smoke":
                a = os.path.join(SKILL, "assets")
                cmd += ["--abc", os.path.join(a, "example-ins.abc"), "--source", os.path.join(a, "example-original.abc"),
                        "--style-file", os.path.join(a, "example-style.txt"), "--repairs", os.path.join(a, "example-repairs.json")]
            else:
                cmd += ["--style", take["style"]]
                if take.get("planning_lyrics"):
                    # Their documented --planning-lyrics-file: planner-only section tags, never sung.
                    pl = os.path.join(runs, f"{name}.planning-lyrics.txt")
                    with open(pl, "w", encoding="utf-8", newline="\n") as f:
                        f.write(take["planning_lyrics"])
                    cmd += ["--planning-lyrics-file", pl]
            if "seed" in take:
                cmd += ["--seed", str(take["seed"])]
            tail = ["--id", re.sub(r"[^A-Za-z0-9_.-]", "_", name), "--models-root", MODELS, "--offline"]
            cut_info, plan_secs = None, 0.0
            if take["kind"] == "cut":
                # 1) their planner + Vocal->Ins transfer, stopping before audio (no GPU audio, no OOM);
                # 2) cut the instrumental score to ad length (trim_score.py, YuE2's own bars only);
                # 3) their run on the cut score as a provided ABC: prepare, generate, share.
                plan_dir = work + ".plan"
                if os.path.exists(plan_dir):
                    shutil.rmtree(plan_dir)
                code, plan_secs = stream(cmd + tail + ["--prepare-only", "--output", plan_dir], os.path.join(slim, "log.txt"))
                src = os.path.join(plan_dir, "conversion", "score.abc")
                copy_if(os.path.join(plan_dir, "planning", "score.abc"), os.path.join(slim, "score_model_plan.abc"))
                copy_if(src, os.path.join(slim, "score_plan_instrumental.abc"))
                if code == 0 and os.path.isfile(src):
                    cut_abc = os.path.join(runs, f"{name}.cut.abc")
                    r = subprocess.run([py, "-B", os.path.join(HERE, "trim_score.py"), src, cut_abc,
                                        "--seconds", str(take.get("seconds", 30))] +
                                       (["--intro-bars", str(take["intro_bars"])] if "intro_bars" in take else []),
                                       capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"), timeout=120)
                    log("cut: " + (r.stdout.strip() or r.stderr.strip()[-400:]))
                    if r.returncode == 0:
                        cut_info = json.loads(r.stdout.strip().splitlines()[-1])
                        copy_if(cut_abc, os.path.join(slim, "score_cut.abc"))
                        cmd = [py, "-B", os.path.join(SCRIPTS, "instrumental.py"), "run", "--abc", cut_abc,
                               "--style", take["style"], "--seed", str(take["seed"])]
                    else:
                        code = 2
                if code:
                    cmd = None
            if cmd:
                code, secs = stream(cmd + tail + ["--output", work], os.path.join(slim, "log.txt"))
            else:
                secs = 0.0
            row = {"name": name, "kind": take["kind"], "style": take.get("style"), "seed": take.get("seed"),
                   "exit": code, "wall_s": round(secs + plan_secs, 1), "plan_s": plan_secs or None, "cut": cut_info,
                   "vram_peak_mib": _state["peak"].get(name)}
            row.update(collect(work, slim))
            if code in (0, 1) and os.path.isdir(os.path.join(work, "generation")):
                vcode, _ = stream([py, "-B", os.path.join(SCRIPTS, "instrumental.py"), "verify",
                                   os.path.join(work, "generation")], os.path.join(slim, "verify.log"))
                row["verify_exit"] = vcode
            results["takes"].append(row)
            log(f"take {name}: exit={code} wall={secs}s audio={row['audio_seconds']} nominal={row['nominal_seconds']} "
                f"truncated={row['truncated']} vram_peak={row['vram_peak_mib']}")
            save()
            if code == 2 and take.get("stop_on_fail"):
                log("stop_on_fail: stopping here")
                break
        log("PROGRESS 99")
    except Exception as e:
        results["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR {results['error']}")
    finally:
        save()
    log(json.dumps({k: results[k] for k in ("run_id", "elapsed_s", "vram_peak_mib") if k in results}))
    if not results["takes"] and "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
