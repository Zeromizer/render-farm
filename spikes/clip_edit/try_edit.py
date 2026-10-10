"""Test driver: run clip_edit (worker/edit/clip_edit.py) on the render PC through the
farm's python engine, before the clip_edit engine exists.

params.args = [<job json string>, <out dir>]. The job JSON:
  {"tests": [{"name": "...", "mode": "...", "inputs": {"clip": <url>, "clip_b": <url>},
              ...clip_edit spec keys..., "prompt": "...", "takes": 2, "seed": 6332}]}
Inputs are fetched from (signed) URLs, so client clips never go into the repo.
Per test: <out>/<name>/{best.mp4, take<N>.mp4, proof.png, report.json}.

Same ComfyUI graph as the engine will use (videogen/graphs_inpaint.py). ComfyUI must
already be up: the worker owns its lifecycle, this only waits for it.
"""
import json
import os
import shutil
import sys
import time
import traceback
import uuid

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))
sys.path.insert(0, os.path.join(ROOT, "worker", "edit"))
from videogen import graphs_inpaint  # noqa: E402
import clip_edit  # noqa: E402

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)


def log(*a):
    print(*a, flush=True)


def ensure_comfy():
    for _ in range(120):
        try:
            if httpx.get(COMFY + "/system_stats", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI did not answer /system_stats within 4 min (the worker starts it)")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "clip_edit_try", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_graph(graph, dest, timeout_s=1800):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:2000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        e = httpx.get(COMFY + f"/history/{pid}", timeout=15).json().get(pid)
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}")
            for node_out in (e.get("outputs") or {}).values():
                for key in ("videos", "images", "gifs"):
                    for f in node_out.get(key, []) or []:
                        if f.get("type", "output") == "output" and f["filename"].lower().endswith(".mp4"):
                            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                                resp.raise_for_status()
                                with open(dest, "wb") as out:
                                    for chunk in resp.iter_bytes(1 << 20):
                                        out.write(chunk)
                            return time.monotonic() - t0
            raise RuntimeError("no mp4 in outputs")
        time.sleep(3)
    httpx.post(COMFY + "/interrupt", timeout=10)
    raise RuntimeError(f"timed out after {timeout_s}s")


def fetch(url, dest):
    with httpx.stream("GET", url, timeout=T, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    return dest


def run_test(t, out_root, work_root):
    name = t["name"]
    work = os.path.join(work_root, name)
    out = os.path.join(out_root, name)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    spec = {k: v for k, v in t.items() if k not in ("name", "inputs", "prompt", "takes", "seed", "steps")}
    for key, url in t["inputs"].items():
        spec[key] = fetch(url, os.path.join(work, f"{key}.mp4"))
    spec["work_dir"] = work
    log(f"== {name}: prep {json.dumps({k: v for k, v in spec.items() if k not in ('clip', 'clip_b')})[:400]}")
    clip_edit.prep(spec)
    plan = json.load(open(os.path.join(work, "plan.json")))
    W, H = plan["canvas"]
    names = [upload(os.path.join(work, f)) for f in ("gen_src.mp4", "gen_mask.mp4", "gen_amask.mp4")]
    takes, seeds = [], []
    for i in range(int(t.get("takes", 2))):
        seed = int(t.get("seed", 6332)) + 1009 * i
        graph, meta = graphs_inpaint.build(*names, t["prompt"], W, H, plan["length"], seed,
                                           f"clip_edit_try/{name}_t{i + 1}", steps=t.get("steps"))
        dest = os.path.join(work, f"raw{i + 1}.mp4")
        secs = run_graph(graph, dest)
        log(f"   take {i + 1}: {W}x{H} {plan['length']}f seed {seed} -> {secs:.0f}s")
        takes.append(dest)
        seeds.append(seed)
    clip_edit.compose({"work_dir": work, "takes": takes, "seeds": seeds, "out": os.path.join(out, "best.mp4"),
                       "proof": os.path.join(out, "proof.png"), "report": os.path.join(out, "report.json")})
    for i in range(len(takes)):
        shutil.copy(os.path.join(work, f"take{i + 1}.mp4"), os.path.join(out, f"take{i + 1}.mp4"))


def main():
    job = json.loads(sys.argv[1])
    out_root = os.path.abspath(sys.argv[2])
    work_root = os.path.join(os.environ.get("RENDER_WORK_DIR", out_root), "clip_edit_work")
    os.makedirs(out_root, exist_ok=True)
    ensure_comfy()
    info = httpx.get(COMFY + "/object_info", timeout=60).json()
    missing = [n for n in graphs_inpaint.REQUIRED_NODES if n not in info]
    if missing:
        raise RuntimeError(f"ComfyUI is missing nodes {missing}")
    failures = {}
    try:
        for k, t in enumerate(job["tests"]):
            print(f"PROGRESS {int(100 * k / len(job['tests']))}", flush=True)
            try:
                run_test(t, out_root, work_root)
            except Exception as exc:  # noqa: BLE001 - one bad test must not sink the rest
                traceback.print_exc()
                failures[t["name"]] = str(exc)
    finally:
        try:
            httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
        except httpx.HTTPError:
            pass
    json.dump({"failures": failures}, open(os.path.join(out_root, "failures.json"), "w"), indent=1)
    if len(failures) == len(job["tests"]):
        sys.exit(f"every test failed: {failures}")


if __name__ == "__main__":
    main()
