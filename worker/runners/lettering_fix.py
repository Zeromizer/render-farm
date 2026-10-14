"""lettering_fix runner: correct number plates and badges on a generated clip with MiniMax
H3, propagating the REAL lettering from a reference photo (worker/lettering/lettering.py
has the why and the pixel work).

params (jsonb):
  lettering_fix:
    org_id / job_id   uuids, scoping and logging only
    source            {bucket, path}  the clip to fix (an H3 output, typically)
    reference         {bucket, path}  a clean photo of the same car whose pose matches one
                      frame of the clip: the i2v last_frame / first_frame, the dealer ad the
                      clip arrives at, a press still of the end pose
    anchor_frame      clip frame that matches the reference (default -1 = last frame)
    elements          [{name, box: [x0, y0, x1, y1]}], 1-4, boxes in REFERENCE pixels,
                      each around one lettered element (plate, badge, wordmark)
    prompt            H3 prompt for the crop: the car, what each element reads, the light
    takes             how many seeds to generate and score (default 2, max 4)
    seed              first seed (default 6332); takes use seed, seed+1009, ...
    steps / turbo     sampling (default turbo 4-step ref2v)
    settle_px         how still an element must hold to count as an anchor frame (2.5)

Output: the best take at outputs/<job_id>.mp4 like every engine, plus siblings
outputs/<job_id>-proof.png (before/fixed per element), outputs/<job_id>-lettering.json
(takes, scores, anchor run, warnings) and outputs/<job_id>-take<N>.mp4 for every take.

GPU: one H3 pass per take (~2 min each for 5-7 s at the follow-crop canvas), in the
same headless ComfyUI as video_gen, with its RAM gate, TTS pause and /free.
"""
import json
import os
import time

import config
import db
import proc
from venvs import venv_python
from runners import gate_common
from runners.video_gen import _run_prompt
from videogen import comfy_client, graphs_inpaint, media_type, ram_gate, tts_guard

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETTERING_DIR = os.path.join(_WORKER_DIR, "lettering")
# Same requirements file as planar_patch -> same cached venv (opencv-headless + numpy).
REQUIREMENTS = os.path.join(_WORKER_DIR, "patch", "requirements.txt")
MAX_TAKES = 4
SEED_STEP = 1009


def validate(p):
    """Raise RuntimeError with an actionable message; return the normalised params."""
    src, ref = p.get("source") or {}, p.get("reference") or {}
    if not (p.get("org_id") and p.get("job_id") and src.get("bucket") and src.get("path")
            and ref.get("bucket") and ref.get("path")):
        raise RuntimeError("lettering_fix needs params.lettering_fix.{org_id, job_id, source, reference, elements, prompt}")
    els = p.get("elements") or []
    if not 1 <= len(els) <= 4:
        raise RuntimeError("lettering_fix needs 1-4 elements")
    names = set()
    for e in els:
        b = e.get("box")
        if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
                and b[2] > b[0] + 4 and b[3] > b[1] + 4):
            raise RuntimeError(f"element {e.get('name')!r}: box must be [x0, y0, x1, y1] in reference pixels, x1>x0, y1>y0")
        name = str(e.get("name") or f"element{len(names)}")
        if name in names:
            raise RuntimeError(f"duplicate element name {name!r}")
        names.add(name)
        e["name"] = name
    if not (p.get("prompt") or "").strip():
        raise RuntimeError("lettering_fix needs a prompt describing the car and what each element reads")
    takes = int(p.get("takes") or 2)
    if not 1 <= takes <= MAX_TAKES:
        raise RuntimeError(f"takes must be 1-{MAX_TAKES}")
    return dict(p, takes=takes, seed=int(p.get("seed") if p.get("seed") is not None else 6332))


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = validate(dict((job.get("params") or {}).get("lettering_fix") or {}))
    deadline = time.monotonic() + timeout_seconds
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}

    db.set_phase(jid, "downloading", 1)
    clip = media_type.ensure_extension(gate_common.download(p["source"]["bucket"], p["source"]["path"], work_dir, "source", log),
                                       ("video",), name="source clip", probe=True)
    ref = media_type.ensure_extension(gate_common.download(p["reference"]["bucket"], p["reference"]["path"], work_dir, "reference", log),
                                      ("image",), name="reference", probe=False)
    heartbeat.progress = 2

    db.set_phase(jid, "building venv", 2)
    py = venv_python(REQUIREMENTS, log, run_kw)

    def stream(sub, spec, lo, hi):
        path = os.path.join(work_dir, f"{sub}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)

        def on_line(line):
            if line.startswith("PHASE "):
                db.set_phase(jid, f"{sub}: {line[6:].strip()}"[:80])
            elif line.startswith("PROGRESS "):
                try:
                    heartbeat.progress = int(lo + (hi - lo) * min(100, int(line.split()[1])) / 100)
                except ValueError:
                    pass
            else:
                log(f"  {line}")
        proc.run_streaming([py, "-u", os.path.join(LETTERING_DIR, "lettering.py"), sub, path], cwd=LETTERING_DIR,
                           on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    base = {"clip": clip, "reference": ref, "work_dir": work_dir, "elements": p["elements"],
            "anchor_frame": int(p.get("anchor_frame", -1)), "settle_px": float(p.get("settle_px", 2.5))}
    stream("prep", base, 3, 10)
    plan = json.load(open(os.path.join(work_dir, "plan.json")))
    W, H = plan["canvas"]
    length = plan["length"]
    for w in plan.get("warnings") or []:
        log(f"warning: {w}")

    db.set_phase(jid, "starting comfyui", 10)
    comfy_client.ensure_server(log)
    seeds = [p["seed"] + SEED_STEP * i for i in range(p["takes"])]
    takes = []
    with tts_guard.paused(log):
        missing = [n for n in graphs_inpaint.REQUIRED_NODES if n not in comfy_client.object_info()]
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {missing}: install PxTicks/ComfyUI-vlo @ 8a7092a in "
                               f"C:\\ComfyUI\\custom_nodes and restart ComfyUI")
        need = config.VIDEO_GEN_MIN_AVAIL_RAM_GB
        waited = ram_gate.wait_for_ram(
            need, config.VIDEO_GEN_RAM_WAIT_MAX_MINUTES * 60, cancel_check,
            lambda gb: db.set_phase(jid, f"waiting for memory: {gb:.1f} GB free, needs {need:g} GB", 11), log)
        deadline += waited
        db.set_phase(jid, "uploading inputs", 12)
        names = [comfy_client.upload_input(os.path.join(work_dir, f), subfolder="lettering_fix")
                 for f in ("gen_src.mp4", "gen_mask.mp4", "gen_amask.mp4")]
        try:
            span = (88 - 12) / len(seeds)
            for i, seed in enumerate(seeds):
                graph, meta = graphs_inpaint.build(*names, p["prompt"], W, H, length, seed, f"lettering_fix/{jid}_t{i + 1}",
                                                   steps=p.get("steps"), turbo=p.get("turbo", True) is not False)
                log(f"take {i + 1}/{len(seeds)}: {W}x{H} {length}f seed={seed} steps={meta['steps']} turbo={meta['turbo']}")
                dest = os.path.join(work_dir, f"raw{i + 1}.mp4")
                lo = int(12 + span * i)
                _run_prompt(jid, graph, f"take {i + 1}/{len(seeds)}", heartbeat, cancel_check, deadline, dest, log,
                            prange=(lo, int(lo + span)))
                takes.append(dest)
        finally:
            comfy_client.free()

    out_local = os.path.join(work_dir, "lettering.mp4")
    proof_local = os.path.join(work_dir, "lettering-proof.png")
    report_local = os.path.join(work_dir, "lettering-report.json")
    stream("compose", dict(base, takes=takes, seeds=seeds, out=out_local, proof=proof_local, report=report_local),
           88, 94)
    if not os.path.exists(out_local):
        raise RuntimeError("lettering compose finished but wrote no output")

    db.set_phase(jid, "uploading", 94)
    report = json.load(open(report_local))
    for f, remote, mime in ((proof_local, f"outputs/{jid}-proof.png", "image/png"),
                            (report_local, f"outputs/{jid}-lettering.json", "application/json")):
        try:
            log(f"{os.path.basename(f)} -> {db.upload_file(remote, f, mime)}")
        except Exception as exc:  # noqa: BLE001 - the fixed clip is the deliverable
            log(f"upload failed for {remote} (clip itself is fine): {exc}")
    for t in report.get("takes", []):
        local = os.path.join(work_dir, f"take{t['take']}.mp4")
        if os.path.exists(local):
            try:
                db.upload_file(f"outputs/{jid}-take{t['take']}.mp4", local, "video/mp4")
            except Exception as exc:  # noqa: BLE001
                log(f"take {t['take']} upload failed: {exc}")
    log(f"lettering_fix done: best take {report.get('best_take')} of {len(takes)}; "
        + ", ".join(f"#{t['take']} {t['score']}" for t in report.get("takes", [])))
    heartbeat.progress = 95
    return out_local, "mp4", "video/mp4"
