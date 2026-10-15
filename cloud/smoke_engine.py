"""Run one farm engine on the pod without the queue, to prove the box before it
takes real jobs (the PC would claim a test row first while it runs old code).

    /workspace/venvs/worker/bin/python cloud/smoke_engine.py video_gen '{"prompt": "...", "duration_s": 3}'

Supabase is stubbed: phases print, uploads are skipped, and the output stays
in /workspace/cache/work/smoke-<engine>/. Only no-clone engines that take
their inputs from params (no storage downloads) can run this way.
"""
import json
import os
import sys
import time

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "stub")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "worker"))

import db  # noqa: E402

db.set_phase = lambda jid, phase, progress=None: print(f"  phase: {phase} ({progress})", flush=True)
db.update_job = lambda *a, **k: None
db.upload_file = lambda remote, local, ct: print(f"  (upload skipped: {remote})", flush=True) or remote

import config  # noqa: E402
from runners import audio_gen, video_gen  # noqa: E402
from videogen import comfy_client, graphs_ltx_plate  # noqa: E402

# {"local": "/path"} in place of a {bucket, path} input: a file already on the pod.
_fetch = video_gen._fetch
video_gen._fetch = lambda ref, *a, **k: ref["local"] if isinstance(ref, dict) and "local" in ref else _fetch(ref, *a, **k)


def ltx_plate(job, repo, work, hb, log, cancel_check, timeout):
    """The LTX 2.5 clean-plate graph (clip_edit's int8_convrot model) on a local clip:
    params {"source": path, "width": 1024, "height": 576, "frames": 97}."""
    p = job["params"]["ltx_plate"]
    src = comfy_client.upload_input(p["source"], subfolder="clip_edit")
    graph, _ = graphs_ltx_plate.build(src, p.get("noun", "car"), p.get("width", 1024), p.get("height", 576),
                                      p.get("frames", 97), 6332, f"clip_edit/{job['id']}_plate")
    pid = comfy_client.submit(graph)
    outputs = comfy_client.wait(pid, lambda *a, **k: None, cancel_check, timeout)
    dest = os.path.join(work, "plate.mp4")
    comfy_client.fetch_output(outputs, dest)
    return dest, "mp4", "video/mp4"


RUNNERS = {"video_gen": video_gen.run, "audio_gen": audio_gen.run, "ltx_plate": ltx_plate}


class _Heartbeat:
    progress = 0


def main():
    engine, params = sys.argv[1], json.loads(sys.argv[2])
    work = os.path.join(config.WORK_DIR, f"smoke-{engine}")
    os.makedirs(work, exist_ok=True)
    job = {"id": f"smoke-{engine}-{int(time.time())}", "engine": engine, "params": {engine: params}}
    t = time.monotonic()
    out, ext, ctype = RUNNERS[engine](job, None, work, _Heartbeat(), lambda m: print(m, flush=True),
                                      lambda: False, 3600)
    print(f"OK {engine}: {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {time.monotonic() - t:.0f} s")


if __name__ == "__main__":
    main()
