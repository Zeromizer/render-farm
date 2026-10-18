"""E2E of THIS checkout's video_gen turntable runner (orbit method) inside a python farm job, so a branch
can be proven on the PC before the live worker runs it. Each case is a params.video_gen.turntable object;
the runner is called exactly as render_worker would, under a throwaway job id (its phase writes match no
row; its sidecars land under renders/outputs/<fake id>-*).

params.args = [<json {"cases": {name: turntable}}>, <out dir>]
Writes <out>/<case>.mp4 (master), <case>-joined24.mp4, <case>-manifest.json, <case>-log.txt, summary.json.
"""
import json
import os
import shutil
import sys
import time
import traceback
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "worker"))

from runners import video_gen  # noqa: E402


class Beat:
    progress = 0


def main():
    cases = json.loads(sys.argv[1])["cases"]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    summary = {}
    for k, (name, tt) in enumerate(cases.items()):
        print(f"PROGRESS {int(100 * k / len(cases))}", flush=True)
        jid = str(uuid.uuid4())
        work = os.path.join(out, f"_work_{name}")
        os.makedirs(work, exist_ok=True)
        lines = []

        def log(msg, _l=lines):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            _l.append(line)
            print(f"{name}: {line}", flush=True)

        t0 = time.monotonic()
        rec = {"job_id": jid}
        try:
            job = {"id": jid, "params": {"video_gen": {"mode": "turntable", "turntable": tt}}}
            path, ext, _ = video_gen.run(job, None, work, Beat(), log, lambda: False, 5400)
            shutil.copyfile(path, os.path.join(out, f"{name}.mp4"))
            tdir = os.path.join(work, "turntable")
            for src, dst in (("joined_24fps.mp4", "joined24.mp4"), ("manifest.json", "manifest.json")):
                if os.path.exists(os.path.join(tdir, src)):
                    shutil.copyfile(os.path.join(tdir, src), os.path.join(out, f"{name}-{dst}"))
            rec["ok"] = True
        except Exception as exc:  # noqa: BLE001 - report every case
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            log(traceback.format_exc())
        rec["seconds"] = round(time.monotonic() - t0, 1)
        with open(os.path.join(out, f"{name}-log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        shutil.rmtree(work, ignore_errors=True)
        summary[name] = rec
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    print("PROGRESS 100", flush=True)


if __name__ == "__main__":
    main()
