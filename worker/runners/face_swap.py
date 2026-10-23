"""face_swap runner: put the person from a photo into a video's performance with MiniMax
H3 reference-to-video. The stages are the accepted spike's (worker/faceswap/fsh3.py, run by
worker/faceswap/engine.py in the planar_patch venv); this runner downloads the inputs, holds
ComfyUI (server up, TTS paused, RAM checked) and uploads the results.

params (jsonb):
  face_swap:
    org_id / job_id   uuids, scoping and logging only
    target            {bucket, path}  the video whose performance is kept
    face              {bucket, path}  a clear, front-on photo of the person to put in
    consent           {confirmed: true, note, kind, by_user, turn_id, at} the platform's record
                      that the person agreed. Refused without it; it is stored with both
                      files' sha256 in the job's provenance (fsh3.record_provenance)
    subject           man | woman | person (wording of the prompt; never guessed)
    person            what the person in the photo looks like, positive words only: face
                      shape, skin, hair (H3 takes the hair from the photo). Empty: the local
                      Qwen3-VL 4B describes the photo (fsh3.describe)
    scene             the video's clothes and setting, as what the person wears ("the same
                      light blue t-shirt with a yellow print, in the same dim room under a
                      grey ceiling"); positive words only. Empty: described from a frame
    props             up to 3 things the person in the video wears that must stay: each
                      {object: a short noun SAM tracks ("headphones"), description: how it
                      looks and sits} or just the noun (then it is described). Each is cut out
                      of the video and passed as its own reference picture; without that the
                      model drops it (the headphones vanished, 2026-10-23)
    hair              from_photo (default: the video's hair is painted out) | from_video
    start_s / end_s   the range of the target to use (default all of it, at most 60 s)
    tries             seeds per piece at most (default 2, max 4); a piece is re-rendered only
                      when its motion drifts from the video
    seed              first seed (default 43); re-rolls use seed+1, seed+2, ...
    chin_clean        fill small dark specks on the chin (default true)
    destubble         lift a stubble shadow on the chin (default false; never on a beard)

Output: outputs/<job_id>.mp4, outputs/<job_id>-proof.png (video | result at six moments),
outputs/<job_id>-sbs.mp4 (side by side) and outputs/<job_id>-swap.json (pieces, seeds, sync,
the prompt, provenance, warnings).

GPU: SAM 3.1 (face, hair, hands, props), maybe the Qwen3-VL 4B, one H3 r2v render per piece
and try, then SAM 3.1 on the kept renders. Gated to the cloud lane like video_gen: a 5090
renders a 10 s piece in ~80 s. On the PC (cloud fallback) pieces are 5 s, because 10 s at
448x768 has never fitted its 32 GB.
"""
import json
import os

import config
import db
import proc
from venvs import venv_python
from runners import gate_common
from videogen import comfy_client, graphs_sam3, media_type, ram_gate, tts_guard

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_DIR = os.path.join(_WORKER_DIR, "faceswap")
REQUIREMENTS = os.path.join(_WORKER_DIR, "patch", "requirements.txt")   # opencv + numpy, shared venv
SUBJECTS = ("man", "woman", "person")
MAX_PROPS = 3
MAX_TRIES = 4
MAX_SECONDS = 60


def _src_ok(s):
    return isinstance(s, dict) and s.get("bucket") and s.get("path")


def _text(v, lo, hi):
    return isinstance(v, str) and lo <= len(v.strip()) <= hi


def validate(p):
    """Raise RuntimeError with an actionable message; return the normalised params."""
    if not (p.get("org_id") and p.get("job_id") and _src_ok(p.get("target")) and _src_ok(p.get("face"))):
        raise RuntimeError("face_swap needs params.face_swap.{org_id, job_id, target, face, consent, subject}")
    consent = p.get("consent")
    if not (isinstance(consent, dict) and consent.get("confirmed") is True):
        raise RuntimeError("face_swap refuses without consent.confirmed: the person in the photo must have agreed")
    if p.get("subject", "person") not in SUBJECTS:
        raise RuntimeError(f"subject must be one of {list(SUBJECTS)}")
    for k in ("person", "scene"):
        if p.get(k) is not None and not _text(p[k], 0, 600):
            raise RuntimeError(f"{k} must be text, up to 600 chars")
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
    props = [{"object": x["object"].strip(), **({"description": x["description"].strip()} if x.get("description")
                                                 else {})} for x in props]
    return dict(p, subject=p.get("subject", "person"), props=props, tries=tries,
                seed=int(p.get("seed") if p.get("seed") is not None else 43))


def consent_text(c, jid):
    """The consent line fsh3 stores in provenance.json beside the two files' sha256."""
    return (f"{c.get('note') or 'confirmed'} | kind={c.get('kind')} by_user={c.get('by_user')} turn={c.get('turn_id')} "
            f"at={c.get('at')} farm_job={jid}")


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = validate(dict((job.get("params") or {}).get("face_swap") or {}))
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}

    db.set_phase(jid, "downloading", 1)
    target = media_type.ensure_extension(gate_common.download(p["target"]["bucket"], p["target"]["path"], work_dir,
                                                              "target", log), ("video",), name="target", probe=True)
    face = media_type.ensure_extension(gate_common.download(p["face"]["bucket"], p["face"]["path"], work_dir,
                                                            "face", log), ("image",), name="face photo", probe=True)
    heartbeat.progress = 2
    db.set_phase(jid, "building venv", 2)
    py = venv_python(REQUIREMENTS, log, run_kw)

    out_local = os.path.join(work_dir, "swap.mp4")
    sbs_local = os.path.join(work_dir, "swap-sbs.mp4")
    proof_local = os.path.join(work_dir, "swap-proof.png")
    report_local = os.path.join(work_dir, "swap-report.json")
    spec = {k: p[k] for k in ("subject", "person", "scene", "props", "hair", "start_s", "end_s", "tries", "seed",
                              "chin_clean", "destubble") if p.get(k) is not None}
    spec.update(work_dir=work_dir, face=face, target=target, cloud=bool(config.WORKER_ENGINES),
                comfy_url=config.COMFYUI_URL, consent=consent_text(p["consent"], jid), out=out_local,
                side_by_side=sbs_local, proof=proof_local, report=report_local)
    spec_path = os.path.join(work_dir, "fs-spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)

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

    db.set_phase(jid, "starting comfyui", 3)
    comfy_client.ensure_server(log)
    with tts_guard.paused(log):
        missing = [n for n in graphs_sam3.REQUIRED_NODES if n not in comfy_client.object_info()]
        if missing:
            raise RuntimeError(f"ComfyUI is missing nodes {missing} for tracking (needs ComfyUI >= 0.37)")
        need = config.VIDEO_GEN_MIN_AVAIL_RAM_GB
        waited = ram_gate.wait_for_ram(
            need, config.VIDEO_GEN_RAM_WAIT_MAX_MINUTES * 60, cancel_check,
            lambda gb: db.set_phase(jid, f"waiting for memory: {gb:.1f} GB free, needs {need:g} GB", 3), log)
        try:
            proc.run_streaming([py, "-u", os.path.join(FS_DIR, "engine.py"), "run", spec_path], cwd=FS_DIR,
                               on_line=on_line, cancel_check=cancel_check,
                               timeout_seconds=int(timeout_seconds + waited))
        except proc.Canceled:
            comfy_client.interrupt()
            raise
        finally:
            comfy_client.free()
    if not os.path.exists(out_local):
        raise RuntimeError("face_swap finished but wrote no output")

    db.set_phase(jid, "uploading", 96)
    for f, remote, mime in ((proof_local, f"outputs/{jid}-proof.png", "image/png"),
                            (sbs_local, f"outputs/{jid}-sbs.mp4", "video/mp4"),
                            (report_local, f"outputs/{jid}-swap.json", "application/json")):
        try:
            log(f"{os.path.basename(f)} -> {db.upload_file(remote, f, mime)}")
        except Exception as exc:  # noqa: BLE001 - the video is the deliverable
            log(f"upload failed for {remote} (the video itself is fine): {exc}")
    report = json.load(open(report_local))
    log(f"face_swap done: {len(report.get('pieces') or [])} piece(s), sync mean {report.get('sync_mean')} "
        f"min {report.get('sync_min')}")
    heartbeat.progress = 97
    return out_local, "mp4", "video/mp4"
