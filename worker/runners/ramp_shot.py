"""Speed-ramp copy runner: fx-library tools/ramp_copy.py end to end, so the
platform's Builder can copy a reference speed-ramp video shot by shot.

Clones fx-library like remotion; the reference, the takes and the hold still
arrive as params.assets (synced into the checkout at the paths given).

params (jsonb):
  mode          analyze | shot
  reference     repo-relative path of the reference video (an asset)
  shots         analyze only: optional list of shot numbers (default all)
  shot          shot only: the reference shot to copy (1-based)
  takes         shot only, orbit/dolly shots: repo-relative paths of H3 takes
                of it (made from its hold still); the best one is used
  hold          shot only, still/slide shots: repo-relative path of the hold
                still drawn from analysis hold_NN.png
  requirements  repo-relative requirements.txt (default tools/ramp_requirements.txt)

analyze -> output analysis.json (per shot: class, hold, prompts); each
hold_NN.png is uploaded beside it as outputs/<job>-hold_NN.png.
shot -> output the finished shot (mp4 at the reference shot's exact frame
count, 30 fps): fit (or still), the TIME-01 / CAM-13 render in this same
checkout, then the cut-blur pass; outputs/<job>-report.json holds the fit
report (which take, gates, scale, size). When no take is good enough the job
fails with ramp_copy's advice (make faster/longer takes) as its error.
"""
import json
import os
import shutil

import db
import proc
from runners import remotion
from venvs import venv_python


def _py(repo, params, log, run_kw):
    req = os.path.join(repo, params.get("requirements") or "tools/ramp_requirements.txt")
    if not os.path.exists(req):
        raise RuntimeError(f"requirements not found in repo: {os.path.relpath(req, repo)}")
    return venv_python(req, log, run_kw)


def _tool(py, repo, args, log, run_kw, heartbeat, lo, hi):
    def on_line(line):
        if line.startswith("PROGRESS "):
            try:
                heartbeat.progress = lo + int(int(line.split()[1]) * (hi - lo) / 100)
            except ValueError:
                pass
        log(f"  {line}")
    out = []
    proc.run_streaming([py, "-u", os.path.join(repo, "tools", "ramp_copy.py"), *[str(a) for a in args]], cwd=repo,
                       on_line=lambda l: (out.append(l), on_line(l)),
                       cancel_check=run_kw["cancel_check"], timeout_seconds=run_kw["timeout_seconds"])
    return out


def _path(repo, rel, what):
    p = os.path.join(repo, rel or "")
    if not rel or not os.path.exists(p):
        raise RuntimeError(f"{what} not found in the checkout: {rel!r} (pass it in params.assets)")
    return p


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    params = job.get("params") or {}
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}
    py = _py(repo, params, log, run_kw)
    heartbeat.progress = 4
    ref = _path(repo, params.get("reference"), "reference")
    mode = params.get("mode")

    if mode == "analyze":
        out = os.path.join(work_dir, "analysis")
        args = ["analyze", ref, out]
        if params.get("shots"):
            args += ["--shots", ",".join(str(int(s)) for s in params["shots"])]
        _tool(py, repo, args, log, run_kw, heartbeat, 5, 95)
        a = json.load(open(os.path.join(out, "analysis.json"), encoding="utf-8"))
        for s in a["shots"]:
            p = os.path.join(out, s["hold_png"])
            s["hold_png_path"] = db.upload_file(f"outputs/{jid}-{s['hold_png']}", p, "image/png")
        res = os.path.join(work_dir, "analysis.json")
        with open(res, "w", encoding="utf-8") as f:
            json.dump(a, f)
        return res, "json", "application/json"

    if mode != "shot":
        raise RuntimeError("ramp_shot params.mode must be analyze or shot")
    shot = int(params.get("shot") or 0)
    if shot < 1:
        raise RuntimeError("ramp_shot shot mode needs params.shot (1-based)")
    out = os.path.join(work_dir, "fit")
    takes = [_path(repo, t, "take") for t in (params.get("takes") or [])]
    if takes:
        _tool(py, repo, ["fit", ref, shot, out, *takes], log, run_kw, heartbeat, 5, 55)
        rep = json.load(open(os.path.join(out, "fit.json"), encoding="utf-8"))
        if not rep.get("best"):
            db.upload_file(f"outputs/{jid}-report.json", os.path.join(out, "fit.json"), "application/json")
            raise RuntimeError(f"no take of shot {shot} is usable: {rep.get('advice')}")
        composition, src, public = "TIME-01", "plate.mp4", os.path.join(out, "plate.mp4")
    else:
        hold = _path(repo, params.get("hold"), "hold still")
        _tool(py, repo, ["still", ref, shot, hold, out], log, run_kw, heartbeat, 5, 30)
        rep = json.load(open(os.path.join(out, "still.json"), encoding="utf-8"))
        composition, src, public = "CAM-13", "master.png", os.path.join(out, "master.png")
    props = json.load(open(os.path.join(out, "props.json"), encoding="utf-8"))
    pub = os.path.join(repo, "remotion", "public", src)
    os.makedirs(os.path.dirname(pub), exist_ok=True)
    shutil.copy2(public, pub)

    class _HB:  # remotion.run reports 5..98: fold it into 55..90 of this job
        def __setattr__(self, k, v):
            if k == "progress":
                heartbeat.progress = 55 + int(v * 0.35)
    rjob = {"id": jid, "params": {"composition": composition, "project_dir": "remotion", "quality": "final", "props": props}}
    rendered, _, _ = remotion.run(rjob, repo, work_dir, _HB(), log, cancel_check, timeout_seconds)
    heartbeat.progress = 90

    final = os.path.join(work_dir, "shot.mp4")
    _tool(py, repo, ["blur", ref, shot, rendered, final], log, run_kw, heartbeat, 90, 98)
    rep["composition"] = composition
    rp = os.path.join(work_dir, "report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rep, f)
    db.upload_file(f"outputs/{jid}-report.json", rp, "application/json")
    return final, "mp4", "video/mp4"
