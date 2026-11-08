"""remake runner (cloud 5090, gated gpu-cloud): the render half of a Viggle Remake person swap. Takes the session a
remake_paint job packed, stores the agent's six-section prompts through Viggle's lint, renders every window with their
ComfyUI 0.42 worker (cloud/setup-remake.sh installed it beside the farm's ComfyUI 0.37), then their review strip and
finish. The farm's own ComfyUI server is stopped for the job (the int8 model needs the whole 32 GB card) and comes
back on the next job that needs it.

params.remake:
  org_id / job_id    uuids
  session            {bucket, path}: a remake_paint paint/repaint job's outputs/<id>.tgz
  prompts            {"0": "<six sections>", ...}: one per window the paint report lists (not passthrough ones)
  consent            {confirmed: true, ...} (carried from the paint stage; refused without it; not for mode object)
  mode               person (default) | object (an edit session: a car or other object replaced)
  seeds              [0] (a fix round: [1])
Output: outputs/<job_id>.mp4 (final, source soundtrack), -sbs.mp4 (source | result), -strip.jpg (their review strip),
-report.json (strip text with cut-timing / edge-band flags, picks, worker log tail).
"""
import json
import os
import signal

import config
import db
import proc
from runners import gate_common
from venvs import venv_python
from videogen import comfy_client

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RM_DIR = os.path.join(_WORKER_DIR, "remake")
REQUIREMENTS = os.path.join(RM_DIR, "requirements.txt")


def validate(p):
    if not (p.get("org_id") and p.get("job_id") and isinstance(p.get("session"), dict) and p["session"].get("path")):
        raise RuntimeError("remake needs params.remake.{org_id, job_id, session, prompts} (+ consent for a person)")
    c = p.get("consent")
    if p.get("mode") not in (None, "person", "face", "whole", "object"):
        raise RuntimeError("mode is face or whole (a person) or object")
    if p.get("mode") != "object" and not (isinstance(c, dict) and c.get("confirmed") is True):
        raise RuntimeError("remake refuses without consent.confirmed")
    pr = p.get("prompts")
    if not (isinstance(pr, dict) and pr and all(str(k).isdigit() and isinstance(v, str) and len(v) > 200 for k, v in pr.items())):
        raise RuntimeError("prompts: {window index: six-section prompt}, one per window to render")
    seeds = p.get("seeds") or [0]
    if not (isinstance(seeds, list) and len(seeds) == 1 and 0 <= int(seeds[0]) < 1000):
        raise RuntimeError("seeds: one seed, e.g. [0] or [1] for a fix round")
    return p


def stop_farm_comfy(log):
    """The int8 transformer, activations and 32 GB card leave no room for the 0.37 server's CUDA context."""
    try:
        comfy_client.free()
        sp = comfy_client.server_process()
        if sp and os.name != "nt":
            os.kill(sp["pid"], signal.SIGTERM)
            log(f"stopped the farm ComfyUI (pid {sp['pid']}) for the Viggle worker")
    except Exception as exc:  # noqa: BLE001
        log(f"could not stop the farm ComfyUI ({exc}); continuing")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = validate(dict((job.get("params") or {}).get("remake") or {}))
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}
    if not os.path.exists(os.path.join(config.REMAKE_COMFY_WORKER_DIR, "ComfyUI", "venv", "bin", "python")):
        raise RuntimeError(f"no Viggle render stack at {config.REMAKE_COMFY_WORKER_DIR}: run cloud/setup-remake.sh")
    db.set_phase(jid, "downloading", 1)
    tar_in = gate_common.download(p["session"]["bucket"], p["session"]["path"], work_dir, "session_in", log)
    import tarfile
    with tarfile.open(tar_in) as t:
        sid = t.getnames()[0].split("/")[0]
    out = {k: os.path.join(work_dir, n) for k, n in (("final_out", "final.mp4"), ("sbs_out", "sbs.mp4"),
                                                     ("strip_out", "strip.jpg"), ("report_out", "report.json"))}
    spec = {"work_dir": work_dir, "home": os.path.join(work_dir, "home"), "sid": sid, "tar_in": tar_in,
            "prompts": {str(k): v for k, v in p["prompts"].items()}, "seeds": [int(s) for s in p.get("seeds") or [0]],
            "comfy_worker_dir": config.REMAKE_COMFY_WORKER_DIR, "worker_timeout_s": max(600, timeout_seconds - 300), **out}
    spec_path = os.path.join(work_dir, "rm-spec.json")
    json.dump(spec, open(spec_path, "w", encoding="utf-8"), indent=1)
    db.set_phase(jid, "building venv", 2)
    py = venv_python(REQUIREMENTS, log, run_kw)

    def on_line(line):
        if line.startswith("PHASE "):
            db.set_phase(jid, line[6:].strip()[:80])
        elif line.startswith("PROGRESS "):
            try:
                heartbeat.progress = int(3 + 92 * min(100, int(line.split()[1])) / 100)
            except ValueError:
                pass
        else:
            log(f"  {line}")

    stop_farm_comfy(log)
    proc.run_streaming([py, "-u", os.path.join(RM_DIR, "driver.py"), "render", spec_path], cwd=RM_DIR,
                       on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)
    db.set_phase(jid, "uploading", 96)
    for f, remote, mime in ((out["sbs_out"], f"outputs/{jid}-sbs.mp4", "video/mp4"),
                            (out["strip_out"], f"outputs/{jid}-strip.jpg", "image/jpeg"),
                            (out["report_out"], f"outputs/{jid}-report.json", "application/json")):
        if os.path.exists(f):
            try:
                log(f"{os.path.basename(f)} -> {db.upload_file(remote, f, mime)}")
            except Exception as exc:  # noqa: BLE001 - the video is the deliverable
                log(f"upload failed for {remote}: {exc}")
    report = json.load(open(out["report_out"], encoding="utf-8"))
    log(f"remake done: flags {report.get('flags')}")
    heartbeat.progress = 97
    return out["final_out"], "mp4", "video/mp4"
