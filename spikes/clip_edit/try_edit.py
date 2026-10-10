"""Test driver: run clip_edit (worker/edit/clip_edit.py) on the render PC through the
farm's python engine, before the clip_edit engine exists.

params.args = [<job json string>, <out dir>]. The job JSON:
  {"tests": [{"name": "...", "mode": "...", "inputs": {"clip": <url>, "clip_b": <url>},
              ...clip_edit spec keys..., "prompt": "...", "takes": 2, "seed": 6332}]}
Inputs are fetched from (signed) URLs, so client clips never go into the repo.
Per test: <out>/<name>/{best.mp4, take<N>.mp4, proof.png, report.json}.

"krea_clean": {"frame", "prompt", "seed", "grow"} first makes cleaned anchor frames with
Krea 2 (masked img2img in the same ComfyUI) over the region's boxes, the frames that
clip_edit.clean_inputs picks, saves each as <out>/<name>/krea_<frame>.png, and passes them
to clip_edit as anchors (with the test's anchor_every).

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
from videogen import graphs_clean, graphs_inpaint, graphs_sam3  # noqa: E402
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


def run_graph(graph, dest, timeout_s=1800, ext=".mp4"):
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
                        if f.get("type", "output") == "output" and f["filename"].lower().endswith(ext):
                            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                                resp.raise_for_status()
                                with open(dest, "wb") as out:
                                    for chunk in resp.iter_bytes(1 << 20):
                                        out.write(chunk)
                            return time.monotonic() - t0
            raise RuntimeError(f"no {ext} in outputs")
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


def run_graph_all(graph, dest_dir, timeout_s=1800):
    """Run a graph and download every saved mp4 into dest_dir; returns the paths."""
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:2000]}")
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
            paths = []
            for node_out in (e.get("outputs") or {}).values():
                for key in ("videos", "images", "gifs"):
                    for f in node_out.get(key, []) or []:
                        if f.get("type", "output") == "output" and f["filename"].lower().endswith(".mp4"):
                            dest = os.path.join(dest_dir, os.path.basename(f["filename"]))
                            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"}
                            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                                resp.raise_for_status()
                                with open(dest, "wb") as out:
                                    for chunk in resp.iter_bytes(1 << 20):
                                        out.write(chunk)
                            paths.append(dest)
            return paths
        time.sleep(2)
    raise RuntimeError(f"timed out after {timeout_s}s")


def track_objects(t, spec, work, out=None):
    """Object regions: SAM 3.1 per-object mask videos, as the runner does."""
    import re
    regions = spec.get("regions") or []
    nouns = sorted({r["object"].strip() for r in regions if r.get("object")})
    if not nouns:
        return
    src = upload(spec["clip"])
    found = {}
    for k, noun in enumerate(nouns):
        d = os.path.join(work, f"sam{k}")
        os.makedirs(d, exist_ok=True)
        t0 = time.monotonic()
        paths = run_graph_all(graphs_sam3.build(src, noun, f"clip_edit_try/{t['name']}_sam{k}"), d)
        found[noun] = sorted(paths, key=lambda q: int(re.search(r"_obj(\d+)", q).group(1)))
        log(f"   sam {noun!r}: {len(paths)} objects in {time.monotonic() - t0:.0f}s")
        if out:   # the per-object mask videos, to see what SAM tracked
            for q in paths:
                shutil.copy(q, os.path.join(out, f"sam{k}_" + re.search(r"_obj\d+", q).group(0)[1:] + ".mp4"))
    httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    for r in regions:
        if r.get("object"):
            r["object_masks"] = found[r["object"].strip()]


def krea_clean(t, spec, work, out):
    """Cleaned anchor frames from Krea 2, the production path: a first prep for the boxes,
    clip_edit.clean_inputs for the frames + masks, videogen/graphs_clean for the graph."""
    kc = t["krea_clean"]
    pre = os.path.join(work, "pre")
    os.makedirs(pre, exist_ok=True)
    clip_edit.prep({**json.loads(json.dumps({k: v for k, v in spec.items() if k != "work_dir"})), "work_dir": pre})
    plan = json.load(open(os.path.join(pre, "plan.json")))
    kw, kh = graphs_clean.size_for(plan["width"], plan["height"])
    clip_edit.clean_inputs({"work_dir": pre, "clip": spec["clip"], "size": [kw, kh], "frame": kc.get("frame"),
                            "grow": kc.get("grow", 24), "out_prefix": os.path.join(work, "krea_in_")})
    cj = json.load(open(os.path.join(pre, "clean.json")))
    seed = int(kc.get("seed", 6332))
    anchors = []
    for f in cj["frames"]:
        img_n, mask_n = upload(f["image"]), upload(f["mask"])
        dest = os.path.join(out, f"krea_{f['frame']}.png")
        secs = run_graph(graphs_clean.build(img_n, mask_n, kc["prompt"], seed, f"clip_edit_try/{t['name']}_krea{f['frame']}"),
                         dest, timeout_s=900, ext=".png")
        log(f"   krea frame {f['frame']}: {kw}x{kh} seed {seed} -> {secs:.0f}s")
        anchors.append({"frame": f["frame"], "image": dest})
    httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    return anchors


def run_test(t, out_root, work_root):
    name = t["name"]
    work = os.path.join(work_root, name)
    out = os.path.join(out_root, name)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    spec = {k: v for k, v in t.items() if k not in ("name", "inputs", "prompt", "takes", "seed", "steps",
                                                    "family", "edges", "krea_clean")}
    for key, url in t["inputs"].items():
        spec[key] = fetch(url, os.path.join(work, f"{key}.mp4"))
    spec["work_dir"] = work
    spec["regions"] = json.loads(json.dumps(spec.get("regions") or [])) or spec.get("regions")
    if not spec["regions"]:
        spec.pop("regions")
    track_objects(t, spec, work, out)
    if t.get("krea_clean"):
        spec["anchors"] = krea_clean(t, spec, work, out)
    log(f"== {name}: prep {json.dumps({k: v for k, v in spec.items() if k not in ('clip', 'clip_b')})[:400]}")
    clip_edit.prep(spec)
    plan = json.load(open(os.path.join(work, "plan.json")))
    W, H = plan["canvas"]
    names = [upload(os.path.join(work, f)) for f in ("gen_src.mp4", "gen_mask.mp4", "gen_amask.mp4")]
    family = t.get("family", "ref2va")
    edges = {}
    if t.get("edges"):
        edges = {"first_frame": upload(os.path.join(work, "edge_first.png")),
                 "last_frame": upload(os.path.join(work, "edge_last.png"))}
    takes, seeds = [], []
    for i in range(int(t.get("takes", 2))):
        seed = int(t.get("seed", 6332)) + 1009 * i
        graph, meta = graphs_inpaint.build(*names, t["prompt"], W, H, plan["length"], seed,
                                           f"clip_edit_try/{name}_t{i + 1}", steps=t.get("steps"),
                                           family=family, **edges)
        dest = os.path.join(work, f"raw{i + 1}.mp4")
        secs = run_graph(graph, dest)
        log(f"   take {i + 1}: {family}{' +edges' if edges else ''} {W}x{H} {plan['length']}f seed {seed} "
            f"steps {meta['steps']} -> {secs:.0f}s")
        takes.append(dest)
        seeds.append(seed)
    clip_edit.compose({"work_dir": work, "takes": takes, "seeds": seeds, "out": os.path.join(out, "best.mp4"),
                       "proof": os.path.join(out, "proof.png"), "report": os.path.join(out, "report.json")})
    for i in range(len(takes)):
        shutil.copy(os.path.join(work, f"take{i + 1}.mp4"), os.path.join(out, f"take{i + 1}.mp4"))


def clear_out(out):
    """The python engine's output dir sits in a shared, reused repo cache and is zipped whole
    after the run: start it empty, or every earlier spike's results ride along (700 MB zips
    timed out the upload)."""
    import shutil
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)


def main():
    job = json.loads(sys.argv[1])
    out_root = os.path.abspath(sys.argv[2])
    clear_out(out_root)
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
