"""face_swap runner: put the person from a photo into a video's performance with MiniMax
H3 reference-to-video (worker/faceswap/face_swap.py has the pixel work and why).

params (jsonb):
  face_swap:
    org_id / job_id   uuids, scoping and logging only
    target            {bucket, path}  the video whose performance is kept
    face              {bucket, path}  a clear, front-on photo of the person to put in
    consent           {confirmed: true, ...} the platform's record that the person agreed
                      (refused without it; the farm never swaps a face nobody vouched for)
    subject           man | woman | person (wording of the prompt)
    person            what the person in the photo looks like, positive words only:
                      face shape, skin, hair (H3 takes the hair from the photo)
    scene             the video's clothes and setting, as what the person wears ("the same
                      light blue t-shirt with a yellow print, in the same dim room under a
                      grey ceiling"); positive words only
    props             up to 3 things the person in the video wears that must stay: each
                      {object: a short noun SAM tracks ("headphones"), description: how it
                      looks and sits, for the prompt ("light grey over-ear headphones, the
                      headband arching over the top of his hair")} or just the noun. Each is
                      cut out of the video and passed as its own reference picture; without
                      that the model drops it (the headphones vanished, 2026-10-23)
    hair              from_photo (default: the video's hair is painted out) | from_video
    start_s / end_s   the range of the target to use (default all of it, at most 60 s)
    tries             renders per piece before the best is kept (default 2, max 4); a piece
                      is re-rendered only when its motion drifts from the video
    seed              first seed (default 43); tries use seed, seed+1009, ...
    chin_clean        fill small dark specks on the chin (default true)
    destubble         lift a stubble shadow on the chin (default false; never on a beard)

Output: outputs/<job_id>.mp4, outputs/<job_id>-proof.png (video over result) and
outputs/<job_id>-swap.json (pieces, seeds, sync scores, joins, warnings).

GPU: SAM 3.1 tracking (face, hair, hands, each prop), then one H3 r2v render per piece
and try, then SAM 3.1 on each kept render (the chin clean-up needs the new face). Gated
to the cloud lane like video_gen: a 5090 renders a 10 s piece in ~80 s. On the PC
(cloud fallback) pieces are 5 s, because 10 s at 448x768 has never fitted its 32 GB.
"""
import json
import os
import time

import config
import db
import proc
from venvs import venv_python
from runners import gate_common
from runners.clip_edit import _sam_tracks
from runners.video_gen import _run_prompt
from videogen import comfy_client, graphs, graphs_sam3, media_type, ram_gate, tts_guard

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_DIR = os.path.join(_WORKER_DIR, "faceswap")
REQUIREMENTS = os.path.join(_WORKER_DIR, "patch", "requirements.txt")   # opencv + numpy, shared venv
SUBJECTS = {"man": ("man", "he", "his"), "woman": ("woman", "she", "her"), "person": ("person", "they", "their")}
MAX_PROPS = 3
MAX_TRIES = 4
MAX_SECONDS = 60
SEED_STEP = 1009
# Re-render a piece whose motion follows the video less than this (Pearson of motion
# energy over 3 s windows; ~0.8 is normal). From one video (2026-10-23): provisional.
GOOD_MEAN, GOOD_MIN = 0.45, 0.30


def _src_ok(s):
    return isinstance(s, dict) and s.get("bucket") and s.get("path")


def _text(v, lo, hi):
    return isinstance(v, str) and lo <= len(v.strip()) <= hi


def validate(p):
    """Raise RuntimeError with an actionable message; return the normalised params."""
    if not (p.get("org_id") and p.get("job_id") and _src_ok(p.get("target")) and _src_ok(p.get("face"))):
        raise RuntimeError("face_swap needs params.face_swap.{org_id, job_id, target, face, consent, subject, person, "
                           "scene}")
    consent = p.get("consent")
    if not (isinstance(consent, dict) and consent.get("confirmed") is True):
        raise RuntimeError("face_swap refuses without consent.confirmed: the person in the photo must have agreed")
    if p.get("subject", "person") not in SUBJECTS:
        raise RuntimeError(f"subject must be one of {sorted(SUBJECTS)}")
    if not _text(p.get("person"), 10, 600):
        raise RuntimeError("face_swap needs person: what the person in the photo looks like (10-600 chars)")
    if p.get("scene") is not None and not _text(p.get("scene"), 0, 600):
        raise RuntimeError("scene must be up to 600 chars")
    props = [x if isinstance(x, dict) else {"object": x} for x in (p.get("props") or [])]
    if not (len(props) <= MAX_PROPS and all(_text(x.get("object"), 2, 40) and (
            x.get("description") is None or _text(x["description"], 3, 200)) for x in props)):
        raise RuntimeError(f"props must be up to {MAX_PROPS} of {{object: a short noun (\"headphones\"), "
                           f"description: how it looks}}")
    if p.get("hair", "from_photo") not in ("from_photo", "from_video"):
        raise RuntimeError("hair must be from_photo or from_video")
    s0, s1 = p.get("start_s"), p.get("end_s")
    for k, v in (("start_s", s0), ("end_s", s1)):
        if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 3600):
            raise RuntimeError(f"{k} must be seconds")
    if s1 is not None and s1 <= (s0 or 0):
        raise RuntimeError("end_s must come after start_s")
    if s1 is not None and s1 - (s0 or 0) > MAX_SECONDS:
        raise RuntimeError(f"one face_swap covers at most {MAX_SECONDS} s; split the video")
    tries = int(p.get("tries") or 2)
    if not 1 <= tries <= MAX_TRIES:
        raise RuntimeError(f"tries must be 1-{MAX_TRIES}")
    props = [{"object": x["object"].strip(), "description": (x.get("description") or x["object"]).strip()}
             for x in props]
    return dict(p, subject=p.get("subject", "person"), props=props, tries=tries,
                seed=int(p.get("seed") if p.get("seed") is not None else 43))


def build_prompt(p, props):
    """The r2v prompt that worked (2026-10-23), with the person, props and scene filled in.
    Positive words only: naming what should not be there tends to add it."""
    noun, he, his = SUBJECTS[p["subject"]]
    He = he.capitalize()
    wears = "wear" if he == "they" else "wears"
    parts = [f"The {noun} from <Picture 1> performs the exact performance of the person in <Video 1>.",
             f"Keep {his} identity from <Picture 1>: {his} face, {p['person'].strip().rstrip('.')}."]
    for i, prop in enumerate(props):
        parts.append(f"{He} {wears} the {prop['description'].rstrip('.')} from <Picture {i + 2}>, exactly as worn in "
                     f"<Video 1>.")
    parts.append("Copy <Video 1> exactly: the same head movements, facial expressions, lip movements and speech "
                 "timing, the same hand gestures, the same framing and camera motion.")
    scene = (p.get("scene") or "").strip().rstrip(".")
    if scene:
        # "the same light blue t-shirt with a yellow print, in the same dim room under a grey ceiling"
        parts.append(f"{He} {wears} {scene}.")
    return " ".join(parts)


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = validate(dict((job.get("params") or {}).get("face_swap") or {}))
    deadline = time.monotonic() + timeout_seconds
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}
    cloud = bool(config.WORKER_ENGINES)

    db.set_phase(jid, "downloading", 1)
    target = media_type.ensure_extension(gate_common.download(p["target"]["bucket"], p["target"]["path"], work_dir,
                                                              "target", log), ("video",), name="target", probe=True)
    face = media_type.ensure_extension(gate_common.download(p["face"]["bucket"], p["face"]["path"], work_dir,
                                                            "face", log), ("image",), name="face photo", probe=True)
    heartbeat.progress = 2
    db.set_phase(jid, "building venv", 2)
    py = venv_python(REQUIREMENTS, log, run_kw)

    def stream(sub, spec, lo, hi):
        path = os.path.join(work_dir, f"fs-{sub}.json")
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
        proc.run_streaming([py, "-u", os.path.join(FS_DIR, "face_swap.py"), sub, path], cwd=FS_DIR,
                           on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    stream("prep", {"work_dir": work_dir, "target": target, "start_s": p.get("start_s"), "end_s": p.get("end_s"),
                    "cloud": cloud}, 3, 6)
    plan = json.load(open(os.path.join(work_dir, "plan.json")))
    W, H = plan["canvas"]
    for w in plan.get("warnings") or []:
        log(f"warning: {w}")
    src = os.path.join(work_dir, "src.mp4")

    db.set_phase(jid, "starting comfyui", 6)
    comfy_client.ensure_server(log)
    with tts_guard.paused(log):
        missing = [n for n in graphs_sam3.REQUIRED_NODES if n not in comfy_client.object_info()]
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {missing} for tracking (needs ComfyUI >= 0.37)")
        masks = {"props": {}}
        nouns = [("face", "face"), ("hair", "hair"), ("hand", "hand")] + [(f"prop{i}", x["object"]) for i, x in
                                                                       enumerate(p["props"])]
        try:
            for i, (tag, noun) in enumerate(nouns):
                db.set_phase(jid, f"tracking the {noun}", 7 + i)
                got = _sam_tracks(jid, noun, src, work_dir, f"sam_{tag}", cancel_check, deadline, log)
                if tag.startswith("prop"):
                    masks["props"][noun] = got
                else:
                    masks[tag] = got
        finally:
            comfy_client.free()
    stream("build", {"work_dir": work_dir, "masks": masks, "hair": p.get("hair", "from_photo")}, 12, 18)
    built = json.load(open(os.path.join(work_dir, "build.json")))
    warnings = list(built.get("warnings") or [])
    for w in warnings:
        log(f"warning: {w}")
    prop_refs = built.get("props") or []
    described = {x["object"]: x for x in p["props"]}
    prompt = build_prompt(p, [described[r["noun"]] for r in prop_refs])
    log(f"prompt: {prompt}")

    pieces = built["pieces"]
    picks = []
    db.set_phase(jid, "starting comfyui", 18)
    comfy_client.ensure_server(log)
    with tts_guard.paused(log):
        need = config.VIDEO_GEN_MIN_AVAIL_RAM_GB
        waited = ram_gate.wait_for_ram(
            need, config.VIDEO_GEN_RAM_WAIT_MAX_MINUTES * 60, cancel_check,
            lambda gb: db.set_phase(jid, f"waiting for memory: {gb:.1f} GB free, needs {need:g} GB", 19), log)
        deadline += waited
        refs = [comfy_client.upload_input(face)] + [comfy_client.upload_input(r["path"]) for r in prop_refs]
        try:
            span = (86 - 20) / len(pieces)
            for k, pc in enumerate(pieces):
                vid = comfy_client.upload_input(pc["path"], subfolder="face_swap")
                lo = 20 + span * k
                tries = []
                for t in range(p["tries"]):
                    seed = p["seed"] + SEED_STEP * t
                    graph, meta = graphs.build("r2v", {"prompt": prompt, "duration_s": 10, "resolution": "480p",
                                                       "ratio": "9:16", "seed": seed},
                                               {"ref_images": refs, "ref_videos": [vid]},
                                               f"face_swap/{jid}_p{pc['start']}_s{seed}")
                    graph["cond"]["inputs"].update(width=W, height=H, length=plan["piece"])
                    dest = os.path.join(work_dir, f"render_{pc['start']}_s{seed}.mp4")
                    label = f"piece {k + 1}/{len(pieces)}" + (f" try {t + 1}" if t else "")
                    log(f"{label}: r2v {W}x{H} {plan['piece']}f seed={seed} steps={meta['steps']}")
                    _run_prompt(jid, graph, label, heartbeat, cancel_check, deadline, dest, log,
                                prange=(int(lo + span * t / p["tries"]), int(lo + span * (t + 1) / p["tries"])))
                    score_out = os.path.join(work_dir, f"score_{pc['start']}_s{seed}.json")
                    stream("score", {"work_dir": work_dir, "renders": [{"start": pc["start"], "path": dest}],
                                     "out": score_out}, int(heartbeat.progress), int(heartbeat.progress))
                    sc = json.load(open(score_out))[0]
                    tries.append({**sc, "seed": seed})
                    log(f"{label}: motion sync mean {sc['mean']} min {sc['min']}")
                    if sc["mean"] >= GOOD_MEAN and sc["min"] >= GOOD_MIN:
                        break
                best = max(tries, key=lambda r: (r["mean"], r["min"]))
                if best["mean"] < GOOD_MEAN or best["min"] < GOOD_MIN:
                    warnings.append(f"the piece from {pc['start'] / 24:.1f} s follows the video loosely "
                                    f"(sync {best['mean']}/{best['min']} after {len(tries)} tries)")
                picks.append({"start": pc["start"], "path": best["path"], "seed": best["seed"], "tries": len(tries),
                              "mean": best["mean"], "min": best["min"]})
        finally:
            comfy_client.free()

        if p.get("chin_clean", True) is not False or p.get("destubble"):
            db.set_phase(jid, "finding the new face", 87)
            try:
                for pk in picks:
                    got = _sam_tracks(jid, "face", pk["path"], work_dir, f"sam_out_{pk['start']}", cancel_check,
                                      deadline, log)
                    # the largest track is the subject; finish() tolerates a missing mask
                    pk["face_mask"] = max(got, key=os.path.getsize) if got else None
            finally:
                comfy_client.free()

    out_local = os.path.join(work_dir, "swap.mp4")
    proof_local = os.path.join(work_dir, "swap-proof.png")
    report_local = os.path.join(work_dir, "swap-report.json")
    stream("finish", {"work_dir": work_dir, "picks": picks, "chin_clean": p.get("chin_clean", True) is not False,
                      "destubble": bool(p.get("destubble")), "out": out_local, "proof": proof_local,
                      "report": report_local, "warnings": warnings}, 90, 95)
    if not os.path.exists(out_local):
        raise RuntimeError("face_swap finished but wrote no output")
    db.set_phase(jid, "uploading", 95)
    for f, remote, mime in ((proof_local, f"outputs/{jid}-proof.png", "image/png"),
                            (report_local, f"outputs/{jid}-swap.json", "application/json")):
        try:
            log(f"{os.path.basename(f)} -> {db.upload_file(remote, f, mime)}")
        except Exception as exc:  # noqa: BLE001 - the video is the deliverable
            log(f"upload failed for {remote} (the video itself is fine): {exc}")
    report = json.load(open(report_local))
    log(f"face_swap done: {len(picks)} piece(s), sync mean {report.get('sync_mean')} min {report.get('sync_min')}")
    heartbeat.progress = 96
    return out_local, "mp4", "video/mp4"
