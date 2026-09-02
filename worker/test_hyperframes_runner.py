"""Exercise runners/hyperframes.py against a local HyperFrames project without
Supabase: fakes the heartbeat/db plumbing and calls run() the way
render_worker.run_job does.

usage: python test_hyperframes_runner.py <project_dir> [still]
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runners import hyperframes  # noqa: E402


class HB:
    progress = 0


def main():
    repo = os.path.abspath(sys.argv[1])
    still = len(sys.argv) > 2 and sys.argv[2] == "still"
    params = {"quality": "draft", "fps": 15}
    if still:
        params = {"output_kind": "still", "at": 8.2}
    job = {"id": "local-test", "params": params}
    work = tempfile.mkdtemp(prefix="hf-runner-")
    hb = HB()
    t0 = time.time()
    out, ext, ctype = hyperframes.run(job, repo, work, hb, print, lambda: False, 600)
    print(f"OK {out} ext={ext} type={ctype} size={os.path.getsize(out)} progress={hb.progress} seconds={time.time()-t0:.0f}")


if __name__ == "__main__":
    main()
