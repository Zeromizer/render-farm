"""Test driver: run the turntable flow (worker/studio/turntable.py) with bridged joins on the
render PC through the farm's python engine, from a branch, before it is deployed.

params.args = [<job json string>, <out dir>]. The job JSON:
  {"segments": {"front_to_left": <url>, ...},   # ready native 24 fps segment clips (signed URLs)
   "turntable": {...turntable params without ready_segments...}}
Writes <out>/{turntable.mp4, joined24.mp4, result.json, bridges/<k>_<r>/(take*.mp4, edit-report.json)}.

Uses the real video_gen _TurntableApi (ComfyUI, clip_edit bridges) with the farm database
stubbed out: nothing is written to farm_render_jobs or storage.
"""
import json
import os
import shutil
import sys
import time
import types
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))
os.environ.setdefault("SUPABASE_URL", "http://unused.invalid")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "unused.unused.unused")   # create_client wants a JWT shape

import db  # noqa: E402

db.set_phase = lambda *a, **k: None
db.upload_file = lambda *a, **k: "(not uploaded)"

from runners import video_gen  # noqa: E402
from studio import turntable  # noqa: E402
from videogen import comfy_client  # noqa: E402


def log(*a):
    print(*a, flush=True)


def main():
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    work = os.path.join(os.environ.get("RENDER_WORK_DIR", out), "tt_work")
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    ready_local, ready_refs = {}, {}
    for name, url in job["segments"].items():
        dest = os.path.join(work, f"ready_{name}.mp4")
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        ready_local[name] = dest
        ready_refs[name] = {"bucket": "renders", "path": f"test/{name}.mp4"}
    tt = dict(job["turntable"], ready_segments=ready_refs)
    pl = turntable.plan(tt)
    log(f"plan: {pl['variant']} reuse {pl['reuse']} bridge {pl['bridge']}")
    comfy_client.ensure_server(log)
    hb = types.SimpleNamespace(progress=0)
    api = video_gen._TurntableApi("tt-bridge-test", work, hb, lambda: False, time.monotonic() + 3 * 3600, log)
    t0 = time.monotonic()
    try:
        res = turntable.run(dict(tt, anchor_local={}, ready_local=ready_local), api, log)
    finally:
        comfy_client.free()
    log(f"turntable done in {time.monotonic() - t0:.0f} s: joins {res['joins']}, {res['frames']} frames, "
        f"defects {res['defects'] or 'none'}")
    shutil.copy(res["video"], os.path.join(out, "turntable.mp4"))
    shutil.copy(res["joined"], os.path.join(out, "joined24.mp4"))
    keep = {k: v for k, v in res.items() if k not in ("segments", "profile_before")}
    keep["seconds_total"] = round(time.monotonic() - t0)
    json.dump(keep, open(os.path.join(out, "result.json"), "w"), indent=1, default=str)
    bdir = os.path.join(api.work_dir, "bridges")
    if os.path.isdir(bdir):
        for sub in sorted(os.listdir(bdir)):
            src = os.path.join(bdir, sub)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(out, "bridges", sub)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                if f.startswith("take") or f in ("edit-report.json", "edit-proof.png"):
                    shutil.copy(os.path.join(src, f), dst)


if __name__ == "__main__":
    main()
