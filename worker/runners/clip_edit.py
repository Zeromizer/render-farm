"""clip_edit runner: surgical MiniMax H3 edits of a generated clip (worker/edit/clip_edit.py
has the modes and the pixel work).

params (jsonb):
  clip_edit:
    org_id / job_id   uuids, scoping and logging only
    mode              region | extend | prepend | bridge | audio
    source            {bucket, path}  the clip to edit
    source_b          {bucket, path}  bridge only: the clip that follows
    prompt            region (crop): what should be inside the window around the boxes;
                      otherwise the whole shot as it should look, incl. sound
    regions           region: [{box: [x0, y0, x1, y1] in source px | box_norm: the same in
                      0-1 of the frame, frame | at_s (where the box is drawn), track
                      (default true; false for overlays that do not move),
                      or {object: a noun SAM 3.1 tracks ("car"), box | box_norm | point |
                      point_norm on that object at frame | at_s} (the object's outline, not a
                      box, is redrawn; start/end as above),
                      start_frame | start_s, end_frame | end_s} or {keys: [{at_s | frame,
                      box | box_norm}, ...] (2-12, linear in between), end_frame | end_s}], 1-4
    crop              region: auto | crop | full (default auto)
    clean             region, for removing an object the scene implies (H3 alone redraws
                      it): {prompt: what the area shows once it is gone, frame | at_s
                      (default: where the boxes cover the most), every (frames between
                      anchors, rounded to H3's 17-frame latent clips; default 17)}. Krea 2 cleans that frame inside the boxes,
                      clip_edit carries it along the camera move every `every` frames and
                      H3 keeps those frames, filling around them. The first and last
                      boxed key frames are cleaned too (entries and exits). The main
                      cleaned frame is also uploaded as outputs/<job_id>-clean.png.
    anchors           region, instead of clean: cleaned frames made elsewhere (an image
                      model's copy of a frame with the object gone), [{image: {bucket,
                      path}, frame | at_s}], 1-4; anchor_every (0 = only the key frame
                      nearest each, or 4-68, default 17)
    seconds           extend / prepend / bridge: how much to generate (snapped to H3's grid)
    context_s         extend / prepend (2.0) / bridge (1.5): original footage H3 sees
    start_frame / end_frame (or start_s / end_s)   audio: range whose sound is regenerated
    takes             seeds to generate and score (default 2, max 4)
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
from videogen import comfy_client, graphs_clean, graphs_inpaint, graphs_sam3, media_type, ram_gate, tts_guard

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT_DIR = os.path.join(_WORKER_DIR, "edit")
# Same requirements file as planar_patch / lettering_fix -> same cached venv.
REQUIREMENTS = os.path.join(_WORKER_DIR, "patch", "requirements.txt")
MODES = ("region", "extend", "prepend", "bridge", "audio")
MAX_TAKES = 4
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
        def point_ok(o):
            pt, norm = (o.get("point_norm"), True) if o.get("point_norm") is not None else (o.get("point"), False)
            return (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)
                    and (not norm or all(0 <= v <= 1 for v in pt)))
        for r in regs:
            keys = r.get("keys")
            if r.get("object") is not None:
                if not (isinstance(r["object"], str) and 0 < len(r["object"].strip()) <= 60):
                    raise RuntimeError("object must be a short noun for what to track (\"car\", \"person\")")
                if keys is not None or not (box_ok(r) if (r.get("box") or r.get("box_norm")) else point_ok(r)):
                    raise RuntimeError("an object region needs a hint on the object: box/box_norm or point/point_norm "
                                       "at frame | at_s (no keys)")
                continue
            if keys is not None:
                if not (isinstance(keys, list) and 2 <= len(keys) <= 12 and all(
                        box_ok(k) and (k.get("at_s") is not None or k.get("frame") is not None) for k in keys)):
                    raise RuntimeError("region keys must be 2-12 of {at_s | frame, box | box_norm}")
            elif not box_ok(r):
                raise RuntimeError("region needs box [x0, y0, x1, y1] in source pixels or box_norm in 0-1 of the "
                                   "frame, with x1>x0 and y1>y0 (or keys)")
        if p.get("crop", "auto") not in ("auto", "crop", "full"):
            raise RuntimeError("crop must be auto, crop or full")
        clean, anchors = p.get("clean"), p.get("anchors")
        if clean is not None and anchors:
            raise RuntimeError("pass clean or anchors, not both")
        if clean is not None:
            if not (isinstance(clean, dict) and (clean.get("prompt") or "").strip()):
                raise RuntimeError("clean needs a prompt describing what the area shows once the object is gone")
            if not 4 <= int(clean.get("every") or 17) <= 68:
                raise RuntimeError("clean.every must be 4-68 frames")
        if anchors:
            if not (isinstance(anchors, list) and 1 <= len(anchors) <= 4 and all(
                    isinstance(a, dict) and _src_ok(a.get("image")) for a in anchors)):
                raise RuntimeError("anchors must be 1-4 of {image: {bucket, path}, frame | at_s}")
            ev = int(p.get("anchor_every") if p.get("anchor_every") is not None else 17)
            if ev and not 4 <= ev <= 68:
                raise RuntimeError("anchor_every must be 0 or 4-68 frames")
    elif p.get("clean") is not None or p.get("anchors"):
        raise RuntimeError("clean / anchors only apply to region edits")
    if mode in ("extend", "prepend", "bridge"):
        sec = float(p.get("seconds") or 0)
        if not 0.25 <= sec <= 12:
            raise RuntimeError("seconds must be 0.25-12")
    if mode == "bridge" and not _src_ok(p.get("source_b")):
        raise RuntimeError("bridge needs source_b, the clip that follows")
    takes = int(p.get("takes") or 2)
    if not 1 <= takes <= MAX_TAKES:
        raise RuntimeError(f"takes must be 1-{MAX_TAKES}")
    return dict(p, takes=takes, seed=int(p.get("seed") if p.get("seed") is not None else 6332))


def _track_objects(jid, regions, clip, work_dir, cancel_check, deadline, log):
    """Object regions: SAM 3.1 tracks every instance of each noun through the clip, one mask
    video per object; clip_edit keeps the one under the region's hint. Sets object_masks."""
    nouns = sorted({r["object"].strip() for r in regions if r.get("object")})
    if not nouns:
        return
    db.set_phase(jid, f"tracking {', '.join(nouns)}", 2)
    comfy_client.ensure_server(log)
    found = {}
    with tts_guard.paused(log):
        missing = [n for n in graphs_sam3.REQUIRED_NODES if n not in comfy_client.object_info()]
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {missing} for object tracking (needs ComfyUI >= 0.37)")
        src = comfy_client.upload_input(clip, subfolder="clip_edit")
        try:
            for k, noun in enumerate(nouns):
                d = os.path.join(work_dir, f"sam{k}")
                os.makedirs(d, exist_ok=True)
                pid = comfy_client.submit(graphs_sam3.build(src, noun, f"clip_edit/{jid}_sam{k}"))
                try:
                    outputs = comfy_client.wait(pid, lambda *_: None, cancel_check,
                                                max(60, int(deadline - time.monotonic())))
                except comfy_client._CanceledSignal:
                    raise proc.Canceled()
                paths = sorted(comfy_client.fetch_outputs(outputs, d),
                               key=lambda q: int(re.search(r"_obj(\d+)", os.path.basename(q)).group(1))
                               if re.search(r"_obj(\d+)", os.path.basename(q)) else 99)
                found[noun] = paths
                log(f"objects: SAM 3.1 tracked {noun!r}: {len(paths)} mask video(s)")
        finally:
            comfy_client.free()
    for r in regions:
        if r.get("object"):
            r["object_masks"] = found.get(r["object"].strip(), [])


def _clean_anchors(jid, p, spec, clip, work_dir, stream, cancel_check, deadline, log):
    """Anchor removal's cleaned frames: a first prep finds the boxes, clip_edit picks the
    frames (the object fully in view, plus the first and last boxed key frames) and writes
    them with their masks, Krea 2 redraws the boxed area from the prompt on each. Returns
    the anchors and the main cleaned frame's path."""
    c = p["clean"]
    stream("prep", spec, 3, 5)
    plan = json.load(open(os.path.join(work_dir, "plan.json")))
    kw, kh = graphs_clean.size_for(plan["width"], plan["height"])
    frame = c.get("frame")
    if frame is None and c.get("at_s") is not None:
        frame = int(round(float(c["at_s"]) * graphs_inpaint.FPS))
    stream("clean_inputs", {"work_dir": work_dir, "clip": clip, "size": [kw, kh], "frame": frame,
                            "out_prefix": os.path.join(work_dir, "clean_in_")}, 5, 6)
    cj = json.load(open(os.path.join(work_dir, "clean.json")))
    db.set_phase(jid, f"cleaning frame{'s' if len(cj['frames']) > 1 else ''} "
                      f"{', '.join(str(f['frame']) for f in cj['frames'])}", 6)
    comfy_client.ensure_server(log)
    anchors, main = [], None
    with tts_guard.paused(log):
        missing = [n for n in graphs_clean.REQUIRED_NODES if n not in comfy_client.object_info()]
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {missing} for cleaning the anchor frame")
        try:
            for f in cj["frames"]:
                names = [comfy_client.upload_input(x, subfolder="clip_edit") for x in (f["image"], f["mask"])]
                graph = graphs_clean.build(*names, c["prompt"], p["seed"], f"clip_edit/{jid}_clean{f['frame']}")
                log(f"clean: Krea 2 on frame {f['frame']} at {kw}x{kh}, seed {p['seed']}")
                pid = comfy_client.submit(graph)
                try:
                    outputs = comfy_client.wait(pid, lambda *_: None, cancel_check,
                                                max(60, int(deadline - time.monotonic())))
                except comfy_client._CanceledSignal:
                    raise proc.Canceled()
                dest = os.path.join(work_dir, f"clean_{f['frame']}.png")
                comfy_client.fetch_output(outputs, dest, exts=(".png",))
                anchors.append({"frame": f["frame"], "image": dest})
                if f["frame"] == cj["main"]:
                    main = dest
        finally:
            comfy_client.free()
    return anchors, main


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

    def stream(sub, spec, lo, hi):
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
        proc.run_streaming([py, "-u", os.path.join(EDIT_DIR, "clip_edit.py"), sub, path], cwd=EDIT_DIR,
                           on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    spec = {k: p[k] for k in ("mode", "regions", "crop", "seconds", "context_s", "start_frame", "end_frame",
                              "start_s", "end_s") if p.get(k) is not None}
    spec.update(clips, work_dir=work_dir)
    if spec.get("regions"):
        spec["regions"] = [dict(r) for r in spec["regions"]]
        _track_objects(jid, spec["regions"], clips["clip"], work_dir, cancel_check, deadline, log)
    clean_local = None
    if p.get("anchors"):
        anchors = []
        for k, a in enumerate(p["anchors"]):
            local = media_type.ensure_extension(
                gate_common.download(a["image"]["bucket"], a["image"]["path"], work_dir, f"anchor{k + 1}", log),
                ("image",), name=f"anchor {k + 1}", probe=True)
            anchors.append(dict({x: a[x] for x in ("frame", "at_s") if a.get(x) is not None}, image=local))
        spec["anchors"] = anchors
        spec["anchor_every"] = int(p.get("anchor_every") if p.get("anchor_every") is not None else 17)
    elif p.get("clean") is not None:
        spec["anchors"], clean_local = _clean_anchors(jid, p, spec, clips["clip"], work_dir, stream, cancel_check,
                                                      deadline, log)
        spec["anchor_every"] = int(p["clean"].get("every") or 17)
    stream("prep", spec, 3 if clean_local is None else 8, 10)
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

    out_local = os.path.join(work_dir, "edit.mp4")
    proof_local = os.path.join(work_dir, "edit-proof.png")
    report_local = os.path.join(work_dir, "edit-report.json")
    stream("compose", {"work_dir": work_dir, "takes": takes, "seeds": seeds, "out": out_local,
                       "proof": proof_local, "report": report_local}, 88, 94)
    if not os.path.exists(out_local):
        raise RuntimeError("clip_edit compose finished but wrote no output")

    db.set_phase(jid, "uploading", 94)
    report = json.load(open(report_local))
    uploads = [(proof_local, f"outputs/{jid}-proof.png", "image/png"),
               (report_local, f"outputs/{jid}-edit.json", "application/json")]
    if clean_local:
        uploads.append((clean_local, f"outputs/{jid}-clean.png", "image/png"))
    for f, remote, mime in uploads:
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
    log(f"clip_edit done: {plan['mode']}, best take {report.get('best_take')} of {len(takes)}")
    heartbeat.progress = 95
    return out_local, "mp4", "video/mp4"
