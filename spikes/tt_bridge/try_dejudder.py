"""Test driver: the turntable output step (RIFE to 60 fps) with and without the de-judder retime,
on the render PC (RIFE lives there). params.args = [<job json>, <out dir>]; job JSON:
  {"bridged": <url of a bridged joined24.mp4>, "anchor": <url of an anchor-joined joined24.mp4>}
Writes bridged_uniform.mp4 (the live path before this change), bridged_dejudder.mp4 (after) and
anchor_remap_dejudder.mp4 (remap_constant_speed with the judder weights) at 1080p60.
"""
import json
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))
os.environ.setdefault("SUPABASE_URL", "http://unused.invalid")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "unused.unused.unused")

from studio import post  # noqa: E402


def main():
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    work = os.path.join(os.environ.get("RENDER_WORK_DIR", out), "dj_work")
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    src = {}
    for k, url in job.items():
        src[k] = os.path.join(work, f"{k}.mp4")
        with urllib.request.urlopen(url, timeout=120) as r, open(src[k], "wb") as f:
            shutil.copyfileobj(r, f)
    fps = 60
    m = post.motion_profile(src["bridged"])
    n_out = int(round(len(m) * fps / 24))
    for name, times in (("bridged_uniform", [k * (len(m) - 1) / (n_out - 1) for k in range(n_out)]),
                        ("bridged_dejudder", post.dejudder_times(m, n_out))):
        fd = os.path.join(work, name)
        post.retime_frames(src["bridged"], fd, times, density=10, tmp_root=work, log=print)
        post.encode_frames(fd, os.path.join(out, f"{name}.mp4"), fps, 1080)
        print(name, "done", flush=True)
    fd = os.path.join(work, "anchor")
    n, _, plateau = post.remap_constant_speed(src["anchor"], fd, fps=fps, density=10, tmp_root=work, log=print)
    post.encode_frames(fd, os.path.join(out, "anchor_remap_dejudder.mp4"), fps, 1080)
    print("anchor done", n, plateau, flush=True)


if __name__ == "__main__":
    main()
