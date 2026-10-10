"""clip_edit runner: surgical MiniMax H3 edits of a generated clip (worker/edit/clip_edit.py
has the modes and the pixel work).

params (jsonb):
  clip_edit:
    org_id / job_id   uuids, scoping and logging only
    mode              region | extend | prepend | bridge | audio | move
    source            {bucket, path}  the clip to edit
    source_b          {bucket, path}  bridge only: the clip that follows
    prompt            region (crop): what should be inside the window around the boxes;
                      otherwise the whole shot as it should look, incl. sound
    regions           region: [{box: [x0, y0, x1, y1] in source px | box_norm: the same in
                      0-1 of the frame, frame | at_s (where the box is drawn), track
                      (default true; false for overlays that do not move),
                      start_frame | start_s, end_frame | end_s} or {keys: [{at_s | frame,
                      box | box_norm}, ...] (2-12, linear in between), end_frame | end_s}], 1-4
    crop              region: auto | crop | full (default auto)
    seconds           extend / prepend / bridge: how much to generate (snapped to H3's grid)
    context_s         extend / prepend (2.0) / bridge (1.5): original footage H3 sees
    start_frame / end_frame (or start_s / end_s)   audio: range whose sound is regenerated
    move              send a car along a new path (worker/edit/clip_move.py; still or moving
                      camera, up to ~15 s): object (a noun SAM 3.1 tracks: "red car"),
                      point_norm [x, y] on that car at the start, route 1-12 points its centre
                      passes through after where it is: [[x, y], ...] in 0-1 of the START frame
                      (-0.5..1.5 to drive out of shot), or timed [{at_s | frame, x, y}, ...],
                      each in 0-1 of the frame AT that time (a moving camera: the car is there
                      then), start_s | start_frame (when it leaves its old path, default 0),
                      hold_s (wait before moving, default 0.25), arrive_s | arrive_frame (plain
                      routes: reaches the last point, default the end), ease inout | linear,
                      turn (default true: the car turns to face where it goes; false keeps its
                      orientation), ttm [start, end] steps (default [1, 3]). The road without
                      the car comes from the LTX 2.5 clean-plate IC-LoRA (graphs_ltx_plate);
                      without its models: a median plate, locked-off shots only
    takes             seeds to generate and score (default 2, max 4; move: default 1)
    seed              first seed (default 6332); takes use seed, seed+1009, ...
    steps / turbo     sampling (default turbo 4-step ref2v)

Output: the best take at outputs/<job_id>.mp4 like every engine, plus
outputs/<job_id>-proof.png, outputs/<job_id>-edit.json (takes, metrics, window,
warnings) and outputs/<job_id>-take<N>.mp4.

GPU: one H3 pass per take, in the same headless ComfyUI as video_gen, with its RAM
gate, TTS pause and /free.
"""
import json
import os
import re
import time

import config
import db
import proc
from venvs import venv_python
from runners import gate_common
from runners.video_gen import _run_prompt
from videogen import (comfy_client, graphs_inpaint, graphs_ltx_plate, graphs_sam3, graphs_ttm, media_type, ram_gate,
                      tts_guard)

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT_DIR = os.path.join(_WORKER_DIR, "edit")
# Same requirements file as planar_patch / lettering_fix -> same cached venv.
REQUIREMENTS = os.path.join(_WORKER_DIR, "patch", "requirements.txt")
MODES = ("region", "extend", "prepend", "bridge", "audio", "move")
MAX_TAKES = 4
MAX_FRAME = 362
SEED_STEP = 1009


def _src_ok(s):
    return isinstance(s, dict) and s.get("bucket") and s.get("path")


def validate(p):
    """Raise RuntimeError with an actionable message; return the normalised params."""
    if not (p.get("org_id") and p.get("job_id") and _src_ok(p.get("source"))):
        raise RuntimeError("clip_edit needs params.clip_edit.{org_id, job_id, source, mode, prompt}")
    mode = p.get("mode")
    if mode not in MODES:
        raise RuntimeError(f"mode must be one of {MODES}")
    if not (p.get("prompt") or "").strip():
        raise RuntimeError("clip_edit needs a prompt describing the result")
    if mode == "region":
        regs = p.get("regions") or []
        if not 1 <= len(regs) <= 4:
            raise RuntimeError("region mode needs 1-4 regions")
        def box_ok(o):
            b, norm = (o.get("box_norm"), True) if o.get("box_norm") is not None else (o.get("box"), False)
            gap = 0.002 if norm else 4
            return (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
                    and b[2] > b[0] + gap and b[3] > b[1] + gap and (not norm or all(0 <= v <= 1 for v in b)))
        for r in regs:
            keys = r.get("keys")
            if keys is not None:
                if not (isinstance(keys, list) and 2 <= len(keys) <= 12 and all(
                        box_ok(k) and (k.get("at_s") is not None or k.get("frame") is not None) for k in keys)):
                    raise RuntimeError("region keys must be 2-12 of {at_s | frame, box | box_norm}")
            elif not box_ok(r):
                raise RuntimeError("region needs box [x0, y0, x1, y1] in source pixels or box_norm in 0-1 of the "
                                   "frame, with x1>x0 and y1>y0 (or keys)")
        if p.get("crop", "auto") not in ("auto", "crop", "full"):
            raise RuntimeError("crop must be auto, crop or full")
    if mode in ("extend", "prepend", "bridge"):
        sec = float(p.get("seconds") or 0)
        if not 0.25 <= sec <= 12:
            raise RuntimeError("seconds must be 0.25-12")
    if mode == "bridge" and not _src_ok(p.get("source_b")):
        raise RuntimeError("bridge needs source_b, the clip that follows")
    if mode == "move":
        _validate_move(p)
    takes = int(p.get("takes") or (1 if mode == "move" else 2))
    if not 1 <= takes <= MAX_TAKES:
        raise RuntimeError(f"takes must be 1-{MAX_TAKES}")
    return dict(p, takes=takes, seed=int(p.get("seed") if p.get("seed") is not None else 6332))


def _validate_move(p):
    if not (isinstance(p.get("object"), str) and 0 < len(p["object"].strip()) <= 60):
        raise RuntimeError('move needs object: a short noun for the car to move ("red car")')
    pt = p.get("point_norm")
    if not (isinstance(pt, (list, tuple)) and len(pt) == 2
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in pt)):
        raise RuntimeError("move needs point_norm [x, y] in 0-1 of the frame, on the car at the start")
    route = p.get("route")
    coord = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and -0.5 <= v <= 1.5  # noqa: E731

    def plain(q):
        return isinstance(q, (list, tuple)) and len(q) == 2 and all(coord(v) for v in q)

    def timed(q):
        t = q.get("frame") if q.get("frame") is not None else q.get("at_s")
        return (isinstance(t, (int, float)) and not isinstance(t, bool) and 0 <= t <= (MAX_FRAME if q.get("frame")
                is not None else 20) and coord(q.get("x")) and coord(q.get("y")))
    ok = isinstance(route, list) and 1 <= len(route) <= 12 and (
        all(plain(q) for q in route) or all(isinstance(q, dict) and timed(q) for q in route))
    if not ok:
        raise RuntimeError("move needs route: 1-12 points the car's centre passes through, either [x, y] in 0-1 of "
                           "the start frame (-0.5..1.5 to leave the shot) or, all of them, timed {at_s, x, y} with "
                           "x, y in 0-1 of the frame at that time")
    if p.get("ease", "inout") not in ("inout", "linear"):
        raise RuntimeError("ease must be inout or linear")
    ttm = p.get("ttm") or [1, 3]
    if not (isinstance(ttm, (list, tuple)) and len(ttm) == 2 and 1 <= int(ttm[0]) < int(ttm[1]) <= 8):
        raise RuntimeError("ttm must be [start, end] steps with 1 <= start < end <= 8")
    for k in ("start_s", "hold_s", "arrive_s"):
        if p.get(k) is not None and not (isinstance(p[k], (int, float)) and 0 <= p[k] <= 20):
            raise RuntimeError(f"{k} must be seconds, 0-20")
    if p.get("start_s") is not None and p.get("arrive_s") is not None and p["arrive_s"] <= p["start_s"]:
        raise RuntimeError("arrive_s must come after start_s")


def _sam_tracks(jid, noun, clip, work_dir, tag, cancel_check, deadline, log):
    """SAM 3.1 tracks every instance of noun through a clip: one mask video per object,
    in object order."""
    missing = [n for n in graphs_sam3.REQUIRED_NODES if n not in comfy_client.object_info()]
    if missing:
        raise RuntimeError(f"ComfyUI is missing nodes {missing} for object tracking (needs ComfyUI >= 0.37)")
    d = os.path.join(work_dir, tag)
    os.makedirs(d, exist_ok=True)
    src = comfy_client.upload_input(clip, subfolder="clip_edit")
    pid = comfy_client.submit(graphs_sam3.build(src, noun, f"clip_edit/{jid}_{tag}"))
    try:
        outputs = comfy_client.wait(pid, lambda *_: None, cancel_check, max(60, int(deadline - time.monotonic())))
    except comfy_client._CanceledSignal:
        raise proc.Canceled()

    def num(q):
        m = re.search(r"_obj(\d+)", os.path.basename(q))
        return int(m.group(1)) if m else 99
    paths = sorted(comfy_client.fetch_outputs(outputs, d), key=num)
    log(f"objects: SAM 3.1 tracked {noun!r} in {tag}: {len(paths)} mask video(s)")
    return paths


def _ltx_ready(log):
    """True when the PC has every node and model file the LTX clean plate needs."""
    info = comfy_client.object_info()
    nodes = [n for n in graphs_ltx_plate.REQUIRED_NODES if n not in info]
    files = graphs_ltx_plate.missing_models(info)
    if nodes or files:
        log(f"LTX clean plate unavailable (missing nodes {nodes}, files {files}): median plate")
        return False
    return True


def _ltx_plate(jid, noun, plan, work_dir, heartbeat, cancel_check, deadline, log, prange):
    """The LTX clean plate, one ComfyUI pass per window (clip_move's prep wrote them).
    Returns the window videos, in order, and the seconds spent waiting for memory."""
    ltx = plan["ltx"]
    lw, lh = ltx["canvas"]
    need = config.VIDEO_GEN_MIN_AVAIL_RAM_GB
    waited = ram_gate.wait_for_ram(
        need, config.VIDEO_GEN_RAM_WAIT_MAX_MINUTES * 60, cancel_check,
        lambda gb: db.set_phase(jid, f"waiting for memory: {gb:.1f} GB free, needs {need:g} GB", prange[0]), log)
    deadline += waited
    out = []
    span = (prange[1] - prange[0]) / max(1, len(ltx["files"]))
    for i, (name, (s0, c)) in enumerate(zip(ltx["files"], ltx["windows"])):
        src = comfy_client.upload_input(os.path.join(work_dir, name), subfolder="clip_edit")
        graph, _ = graphs_ltx_plate.build(src, noun, lw, lh, c, 6332, f"clip_edit/{jid}_plate{i}")
        dest = os.path.join(work_dir, f"plate_{i}.mp4")
        lo = int(prange[0] + span * i)
        log(f"clean plate {i + 1}/{len(ltx['files'])}: LTX {lw}x{lh}, frames {s0}-{s0 + c - 1}")
        _run_prompt(jid, graph, f"clean plate {i + 1}/{len(ltx['files'])}", heartbeat, cancel_check, deadline, dest,
                    log, prange=(lo, int(lo + span)))
        out.append(dest)
    return out, waited


def _run_move(jid, p, clip, work_dir, stream, heartbeat, cancel_check, deadline, log):
    """Move mode up to the takes: prep (canvas, camera track, plate windows) -> SAM on the
    source -> the LTX clean plate (or, without its models and only on a locked-off shot, SAM on
    all traffic for a median plate) -> build (cut-and-drag reference) -> per take, H3 +
    Time-to-Move, then SAM finds the car in it. Returns (takes, their mask videos, seeds)."""
    stream("prep", {"clip": clip, "work_dir": work_dir}, 3, 6)
    plan = json.load(open(os.path.join(work_dir, "plan.json")))
    W, H = plan["canvas"]
    for w in plan.get("warnings") or []:
        log(f"warning: {w}")
    noun = p["object"].strip()
    db.set_phase(jid, "starting comfyui", 6)
    comfy_client.ensure_server(log)
    plate_windows, other_masks = None, None
    with tts_guard.paused(log):
        db.set_phase(jid, f"tracking {noun}", 7)
        try:
            src_masks = _sam_tracks(jid, noun, os.path.join(work_dir, "move_src.mp4"), work_dir, "sam_src",
                                    cancel_check, deadline, log)
        finally:
            comfy_client.free()
        if not src_masks:
            raise RuntimeError(f'SAM 3.1 found no {noun!r} in the clip; name the car the way it looks ("red car")')
        if _ltx_ready(log):
            db.set_phase(jid, "clean plate", 8)
            try:
                plate_windows, waited = _ltx_plate(jid, noun, plan, work_dir, heartbeat, cancel_check, deadline, log,
                                                   (8, 16))
                deadline += waited
            finally:
                comfy_client.free()
        elif plan.get("camera_moving"):
            raise RuntimeError("the camera moves in this clip and the LTX clean-plate models are not on the render PC "
                               "(see graphs_ltx_plate.MODEL_FILES); without them move edits need a locked-off shot")
        else:
            # every car, so other traffic stays out of the median plate
            db.set_phase(jid, "tracking other traffic", 8)
            try:
                other_masks = _sam_tracks(jid, "car", os.path.join(work_dir, "move_src.mp4"), work_dir, "sam_all",
                                          cancel_check, deadline, log)
            finally:
                comfy_client.free()
    spec = {k: p[k] for k in ("object", "point_norm", "route", "start_s", "start_frame", "hold_s", "arrive_s",
                              "arrive_frame", "ease", "turn") if p.get(k) is not None}
    spec.update(work_dir=work_dir, sam_masks=src_masks)
    if plate_windows:
        spec["plate_windows"] = plate_windows
    else:
        spec["other_masks"] = other_masks
    stream("build", spec, 16, 19)

    seeds = [p["seed"] + SEED_STEP * i for i in range(p["takes"])]
    ttm = p.get("ttm") or [1, 3]
    takes, take_masks = [], []
    db.set_phase(jid, "starting comfyui", 19)
    comfy_client.ensure_server(log)
    with tts_guard.paused(log):
        missing = [n for n in graphs_ttm.REQUIRED_NODES if n not in comfy_client.object_info()]
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {missing}: install PxTicks/ComfyUI-vlo @ 8a7092a in "
                               f"C:\\ComfyUI\\custom_nodes and restart ComfyUI")
        need = config.VIDEO_GEN_MIN_AVAIL_RAM_GB
        waited = ram_gate.wait_for_ram(
            need, config.VIDEO_GEN_RAM_WAIT_MAX_MINUTES * 60, cancel_check,
            lambda gb: db.set_phase(jid, f"waiting for memory: {gb:.1f} GB free, needs {need:g} GB", 20), log)
        deadline += waited
        db.set_phase(jid, "uploading inputs", 21)
        names = [comfy_client.upload_input(os.path.join(work_dir, f), subfolder="clip_edit")
                 for f in ("move_ref.mp4", "move_refmask.mp4", "first.png")]
        try:
            span = (80 - 21) / len(seeds)
            for i, seed in enumerate(seeds):
                graph, meta = graphs_ttm.build(*names, p["prompt"], W, H, plan["length"], seed,
                                               f"clip_edit/{jid}_t{i + 1}", ttm=ttm)
                log(f"take {i + 1}/{len(seeds)}: move {W}x{H} {plan['length']}f seed={seed} "
                    f"steps={meta['steps']} ttm={meta['ttm']}")
                dest = os.path.join(work_dir, f"raw{i + 1}.mp4")
                lo = int(21 + span * i)
                _run_prompt(jid, graph, f"take {i + 1}/{len(seeds)}", heartbeat, cancel_check, deadline, dest, log,
                            prange=(lo, int(lo + span * 0.85)))
                comfy_client.free()
                db.set_phase(jid, f"finding the {noun} in take {i + 1}", int(lo + span * 0.85))
                take_masks.append(_sam_tracks(jid, noun, dest, work_dir, f"sam_t{i + 1}", cancel_check, deadline,
                                              log))
                takes.append(dest)
        finally:
            comfy_client.free()
    return takes, take_masks, seeds


def _upload_extras(jid, work_dir, proof_local, report_local, log):
    report = json.load(open(report_local))
    for f, remote, mime in ((proof_local, f"outputs/{jid}-proof.png", "image/png"),
                            (report_local, f"outputs/{jid}-edit.json", "application/json")):
        try:
            log(f"{os.path.basename(f)} -> {db.upload_file(remote, f, mime)}")
        except Exception as exc:  # noqa: BLE001 - the edited clip is the deliverable
            log(f"upload failed for {remote} (clip itself is fine): {exc}")
    for t in report.get("takes", []):
        local = os.path.join(work_dir, f"take{t['take']}.mp4")
        if os.path.exists(local):
            try:
                db.upload_file(f"outputs/{jid}-take{t['take']}.mp4", local, "video/mp4")
            except Exception as exc:  # noqa: BLE001
                log(f"take {t['take']} upload failed: {exc}")
    return report


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = validate(dict((job.get("params") or {}).get("clip_edit") or {}))
    deadline = time.monotonic() + timeout_seconds
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}

    db.set_phase(jid, "downloading", 1)
    clips = {}
    for key, name in (("source", "clip"), ("source_b", "clip_b")):
        if p.get(key):
            clips[name] = media_type.ensure_extension(
                gate_common.download(p[key]["bucket"], p[key]["path"], work_dir, key, log),
                ("video",), name=key.replace("_", " "), probe=True)
    heartbeat.progress = 2

    db.set_phase(jid, "building venv", 2)
    py = venv_python(REQUIREMENTS, log, run_kw)

    def stream(sub, spec, lo, hi, script="clip_edit.py"):
        path = os.path.join(work_dir, f"edit-{sub}.json")
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
        proc.run_streaming([py, "-u", os.path.join(EDIT_DIR, script), sub, path], cwd=EDIT_DIR,
                           on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    out_local = os.path.join(work_dir, "edit.mp4")
    proof_local = os.path.join(work_dir, "edit-proof.png")
    report_local = os.path.join(work_dir, "edit-report.json")
    if p["mode"] == "move":
        takes, take_masks, seeds = _run_move(jid, p, clips["clip"], work_dir,
                                             lambda sub, spec, lo, hi: stream(sub, spec, lo, hi, "clip_move.py"),
                                             heartbeat, cancel_check, deadline, log)
        stream("compose", {"work_dir": work_dir, "clip": clips["clip"], "takes": takes, "take_masks": take_masks,
                           "seeds": seeds, "out": out_local, "proof": proof_local, "report": report_local},
               80, 94, "clip_move.py")
        if not os.path.exists(out_local):
            raise RuntimeError("clip_edit move compose finished but wrote no output")
        db.set_phase(jid, "uploading", 94)
        report = _upload_extras(jid, work_dir, proof_local, report_local, log)
        log(f"clip_edit done: move, best take {report.get('best_take')} of {len(takes)}")
        heartbeat.progress = 95
        return out_local, "mp4", "video/mp4"

    spec = {k: p[k] for k in ("mode", "regions", "crop", "seconds", "context_s", "start_frame", "end_frame",
                              "start_s", "end_s") if p.get(k) is not None}
    spec.update(clips, work_dir=work_dir)
    stream("prep", spec, 3, 10)
    plan = json.load(open(os.path.join(work_dir, "plan.json")))
    W, H = plan["canvas"]
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
        names = [comfy_client.upload_input(os.path.join(work_dir, f), subfolder="clip_edit")
                 for f in ("gen_src.mp4", "gen_mask.mp4", "gen_amask.mp4")]
        try:
            span = (88 - 12) / len(seeds)
            for i, seed in enumerate(seeds):
                graph, meta = graphs_inpaint.build(*names, p["prompt"], W, H, plan["length"], seed,
                                                   f"clip_edit/{jid}_t{i + 1}", steps=p.get("steps"),
                                                   turbo=p.get("turbo", True) is not False)
                log(f"take {i + 1}/{len(seeds)}: {plan['mode']} {W}x{H} {plan['length']}f seed={seed} "
                    f"steps={meta['steps']} turbo={meta['turbo']}")
                dest = os.path.join(work_dir, f"raw{i + 1}.mp4")
                lo = int(12 + span * i)
                _run_prompt(jid, graph, f"take {i + 1}/{len(seeds)}", heartbeat, cancel_check, deadline, dest, log,
                            prange=(lo, int(lo + span)))
                takes.append(dest)
        finally:
            comfy_client.free()

    stream("compose", {"work_dir": work_dir, "takes": takes, "seeds": seeds, "out": out_local,
                       "proof": proof_local, "report": report_local}, 88, 94)
    if not os.path.exists(out_local):
        raise RuntimeError("clip_edit compose finished but wrote no output")

    db.set_phase(jid, "uploading", 94)
    report = _upload_extras(jid, work_dir, proof_local, report_local, log)
    log(f"clip_edit done: {plan['mode']}, best take {report.get('best_take')} of {len(takes)}")
    heartbeat.progress = 95
    return out_local, "mp4", "video/mp4"
