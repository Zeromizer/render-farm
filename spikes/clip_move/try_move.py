"""GPU check of clip_edit's move mode through the farm's python engine, before the worker
runs it: the production pixel code (worker/edit/clip_move.py) and graphs (graphs_sam3,
graphs_ttm), in the order the runner uses them (_run_move), against the PC's ComfyUI.

params.args = [<job json string>, <out dir>]. Job JSON = the runner's params.clip_edit for
mode move, with "clip" a signed URL instead of source.
Writes <out>/{edit.mp4, edit-proof.png, edit-report.json, take*.mp4, route.png, ...}.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))
sys.path.insert(0, os.path.join(ROOT, "worker", "edit"))
from videogen import graphs_sam3, graphs_ttm  # noqa: E402

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)
SEED_STEP = 1009


def log(*a):
    print(*a, flush=True)


def ensure_comfy(wait_s=600):
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        try:
            if httpx.get(COMFY + "/system_stats", timeout=10).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI did not answer")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "move_try", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_outputs(graph, timeout_s=1800):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:3000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            e = httpx.get(COMFY + f"/history/{pid}", timeout=60).json().get(pid)
        except httpx.TimeoutException:
            continue
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}")
            return e.get("outputs") or {}
        time.sleep(2)
    raise RuntimeError("timed out")


def fetch_all(outputs, dest_dir, exts=(".mp4",)):
    paths = []
    for node_out in outputs.values():
        for key in ("videos", "images", "gifs"):
            for f in node_out.get(key, []) or []:
                name = f.get("filename", "")
                if f.get("type", "output") != "output" or not name.lower().endswith(exts):
                    continue
                dest = os.path.join(dest_dir, os.path.basename(name))
                params = {"filename": name, "subfolder": f.get("subfolder", ""), "type": "output"}
                with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as out:
                        for chunk in resp.iter_bytes(1 << 20):
                            out.write(chunk)
                paths.append(dest)
    return paths


def free():
    try:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


def sam(noun, clip, work, tag):
    d = os.path.join(work, tag)
    os.makedirs(d, exist_ok=True)
    outs = run_outputs(graphs_sam3.build(upload(clip), noun, f"move_try/{tag}"))
    num = lambda q: int(m.group(1)) if (m := re.search(r"_obj(\d+)", os.path.basename(q))) else 99  # noqa: E731
    paths = sorted(fetch_all(outs, d), key=num)
    free()
    log(f"sam {tag}: {len(paths)} track(s)")
    return paths


def stream(sub, spec, work):
    path = os.path.join(work, f"edit-{sub}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-u", os.path.join(ROOT, "worker", "edit", "clip_move.py"), sub, path],
                   check=True, cwd=os.path.join(ROOT, "worker", "edit"))
    log(f"{sub}: {time.monotonic() - t0:.0f}s")


def main():
    p = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)
    work = os.path.join(out, "work")
    os.makedirs(work)
    clip = os.path.join(work, "source.mp4")
    with httpx.stream("GET", p["clip"], timeout=T, follow_redirects=True) as r:
        r.raise_for_status()
        with open(clip, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    ensure_comfy()
    t_all = time.monotonic()
    stream("prep", {"clip": clip, "work_dir": work}, work)
    plan = json.load(open(os.path.join(work, "plan.json")))
    W, H = plan["canvas"]
    noun = p["object"].strip()
    t0 = time.monotonic()
    src_masks = sam(noun, os.path.join(work, "move_src.mp4"), work, "sam_src")
    log(f"sam source: {time.monotonic() - t0:.0f}s")
    spec = {k: p[k] for k in ("object", "point_norm", "route", "start_s", "start_frame", "hold_s", "arrive_s",
                              "arrive_frame", "ease", "turn") if p.get(k) is not None}
    spec.update(work_dir=work, sam_masks=src_masks)
    stream("build", spec, work)
    names = [upload(os.path.join(work, f)) for f in ("move_ref.mp4", "move_refmask.mp4", "first.png")]
    seeds = [int(p.get("seed", 6332)) + SEED_STEP * i for i in range(int(p.get("takes", 1)))]
    takes, take_masks = [], []
    for i, seed in enumerate(seeds):
        graph, meta = graphs_ttm.build(*names, p["prompt"], W, H, plan["length"], seed, f"move_try/t{i + 1}",
                                       ttm=p.get("ttm") or [1, 3])
        t0 = time.monotonic()
        outs = run_outputs(graph)
        raw = fetch_all(outs, work)
        dest = os.path.join(work, f"raw{i + 1}.mp4")
        os.replace(raw[0], dest)
        free()
        log(f"take {i + 1}: {W}x{H} {plan['length']}f {meta} in {time.monotonic() - t0:.0f}s")
        take_masks.append(sam(noun, dest, work, f"sam_t{i + 1}"))
        takes.append(dest)
    stream("compose", {"work_dir": work, "clip": clip, "takes": takes, "take_masks": take_masks, "seeds": seeds,
                       "out": os.path.join(out, "edit.mp4"), "proof": os.path.join(out, "edit-proof.png"),
                       "report": os.path.join(out, "edit-report.json")}, work)
    log(f"total {time.monotonic() - t_all:.0f}s")
    for f in ("route.png", "plate.png"):
        shutil.copy(os.path.join(work, f), out)
    for f in os.listdir(work):
        if f.startswith("take") and f.endswith(".mp4"):
            shutil.copy(os.path.join(work, f), out)
    shutil.rmtree(work, ignore_errors=True)  # keep the zip small


if __name__ == "__main__":
    main()
