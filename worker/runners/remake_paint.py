"""remake_paint runner (PC, no GPU): the first half of a Viggle Remake person swap (engine `remake` is the second).
Runs worker/remake/driver.py paint in its own venv; keyframes are painted on the Codex subscription (codex exec,
REMAKE_CODEX_BIN / REMAKE_CODEX_AUTH), squeezed to a shape image_gen returns (worker/remake/shapes.py).

params.remake_paint:
  org_id / job_id    uuids (scoping, logging)
  stage              prepare | paint | repaint
  prepare:  target {bucket, path} (the video), start_s, end_s (<= 60 s), crop "w:h:x:y" (optional)
  paint:    session {bucket, path} (the prepare job's outputs/<id>.tgz), face {bucket, path} (identity picture: a
            photo, or a cast sheet), consent {confirmed: true, ...} (the platform's record; refused without it),
            name, identity, outfit, person, keyframes [{frame | at_s, where}] (required), others [{desc}],
            shots_with_person [i], kind person|anthro|creature, note
  repaint:  session (a paint job's .tgz), consent, repaint {frames [..], note}
  mode object (Viggle's edit session: a car or other object replaced, the footage otherwise kept; no consent):
  prepare:  + mode "object"
  paint:    session, name (the new object, e.g. "SU7 Ultra"), object (the thing in the video to replace), replacement
            (what it becomes, its looks), edit (one sentence), instruction (the keyframe edit: what changes, the new
            object's OWN shape, what fills the space), keyframes [{frame | at_s, where, instruction?}], refs [{bucket,
            path}] (0-4 pictures of the new object), others [{desc}] (kept as they are), shots_with_object [i], note
  repaint:  session, repaint {frames [..], instruction?}
Output: outputs/<job_id>.tgz (the session folder, the next stage's input) plus outputs/<job_id>-report.json and the
pictures the agent reviews under outputs/<job_id>/ (strips, character sheet, keyframe comparisons, window sheets).
"""
import glob
import json
import os
import re
import shutil

import config
import db
import proc
from runners import gate_common
from venvs import venv_python
from videogen import media_type

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RM_DIR = os.path.join(_WORKER_DIR, "remake")
REQUIREMENTS = os.path.join(RM_DIR, "requirements.txt")
STAGES = ("prepare", "paint", "repaint")
MAX_SECONDS = 60
NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_")


def codex_bin():
    """The codex executable for the painter. Viggle's pool runs subprocess.run(["codex", ...]) with no shell, so on
    Windows it must be a real codex.exe (an npm codex.cmd shim is not found). REMAKE_CODEX_BIN wins (a path, or a glob);
    else the newest VS Code ChatGPT extension's bundled binary, whose folder is versioned and changes on every
    extension update; else codex(.exe) on PATH."""
    exe = "codex.exe" if os.name == "nt" else "codex"
    pats = [config.REMAKE_CODEX_BIN] if config.REMAKE_CODEX_BIN else [
        os.path.join(os.path.expanduser("~"), ".vscode", "extensions", "openai.chatgpt-*", "bin", "*", exe)]
    hits = [h for pat in pats for h in glob.glob(pat) if os.path.isfile(h)]
    if hits:
        ver = lambda h: [int(x) for x in re.findall(r"\d+", h.split("openai.chatgpt-")[-1].split(os.sep)[0])]
        return max(hits, key=ver)
    found = shutil.which(exe)
    if found and found.lower().endswith((".exe", "codex")):
        return found
    raise RuntimeError(f"no codex executable for painting: set REMAKE_CODEX_BIN (looked for {pats} and {exe} on PATH)")


def _src_ok(s):
    return isinstance(s, dict) and s.get("bucket") and s.get("path")


def _text(v, lo, hi):
    return isinstance(v, str) and lo <= len(v.strip()) <= hi


def _validate_object(p, stage):
    if stage == "repaint":
        rp = p.get("repaint") or {}
        if not rp.get("frames"):
            raise RuntimeError("repaint needs repaint.frames")
        if rp.get("instruction") is not None and not _text(rp["instruction"], 20, 2000):
            raise RuntimeError("repaint.instruction: 20-2000 chars")
        return p
    if not (_text(p.get("name"), 1, 40) and set(p["name"]) <= NAME_OK):
        raise RuntimeError("name: 1-40 letters, digits, spaces, - or _")
    for k, lo, hi in (("object", 5, 400), ("replacement", 10, 800), ("edit", 10, 400), ("instruction", 40, 2000)):
        if not _text(p.get(k), lo, hi):
            raise RuntimeError(f"{k} must be text, {lo}-{hi} chars")
    kfs = p.get("keyframes") or []
    if not (1 <= len(kfs) <= 12 and all(isinstance(k, dict) and (k.get("frame") is not None or k.get("at_s") is not None)
                                         and _text(k.get("where"), 3, 300)
                                         and (k.get("instruction") is None or _text(k["instruction"], 40, 2000))
                                         for k in kfs)):
        raise RuntimeError("keyframes: 1-12 of {frame | at_s, where, instruction?}, one per shot that shows the object")
    refs = p.get("refs") or []
    if not (len(refs) <= 4 and all(_src_ok(r) for r in refs)):
        raise RuntimeError("refs: 0-4 pictures {bucket, path} of the new object")
    return p


def validate(p):
    stage = p.get("stage")
    if not (p.get("org_id") and p.get("job_id") and stage in STAGES):
        raise RuntimeError(f"remake_paint needs params.remake_paint.{{org_id, job_id, stage in {list(STAGES)}}}")
    if p.get("mode") not in (None, "person", "face", "whole", "object"):   # face | whole: a person swap's own modes
        raise RuntimeError("mode is face or whole (a person) or object")
    if stage == "prepare":
        if not _src_ok(p.get("target")):
            raise RuntimeError("prepare needs target {bucket, path}")
        s0, s1 = float(p.get("start_s") or 0), p.get("end_s")
        if s1 is not None and not 0 < float(s1) - s0 <= MAX_SECONDS:
            raise RuntimeError(f"end_s must be after start_s, at most {MAX_SECONDS} s later")
        return p
    if not _src_ok(p.get("session")):
        raise RuntimeError(f"{stage} needs session {{bucket, path}}: the previous stage's outputs/<id>.tgz")
    if p.get("mode") == "object":
        return _validate_object(p, stage)
    c = p.get("consent")
    if not (isinstance(c, dict) and c.get("confirmed") is True):
        raise RuntimeError("remake_paint refuses without consent.confirmed: the person in the picture must have agreed")
    if stage == "repaint":
        if not (p.get("repaint") or {}).get("frames"):
            raise RuntimeError("repaint needs repaint.frames")
        return p
    if not _src_ok(p.get("face")):
        raise RuntimeError("paint needs face {bucket, path}")
    if not (_text(p.get("name"), 1, 40) and set(p["name"]) <= NAME_OK):
        raise RuntimeError("name: 1-40 letters, digits, spaces, - or _")
    for k, lo, hi in (("identity", 10, 600), ("outfit", 5, 400), ("person", 5, 400)):
        if not _text(p.get(k), lo, hi):
            raise RuntimeError(f"{k} must be text, {lo}-{hi} chars")
    kfs = p.get("keyframes") or []
    if not (1 <= len(kfs) <= 12 and all(isinstance(k, dict) and (k.get("frame") is not None or k.get("at_s") is not None)
                                         and _text(k.get("where"), 3, 300) for k in kfs)):
        raise RuntimeError("keyframes: 1-12 of {frame | at_s, where}, one per shot that shows the person")
    if p.get("kind", "person") not in ("person", "anthro", "creature"):
        raise RuntimeError("kind must be person, anthro or creature")
    return p


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = validate(dict((job.get("params") or {}).get("remake_paint") or {}))
    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check, "timeout_seconds": timeout_seconds}
    db.set_phase(jid, "downloading", 1)
    spec = {"work_dir": work_dir, "home": os.path.join(work_dir, "home"), "stage": p["stage"],
            "sid": f"rm{p['job_id'].replace('-', '')[:12]}",
            "tar_out": os.path.join(work_dir, "session.tgz"), "report_out": os.path.join(work_dir, "report.json"),
            "codex_bin": codex_bin() if p["stage"] != "prepare" else "", "codex_auth": config.REMAKE_CODEX_AUTH}
    spec["kind"] = "edit" if p.get("mode") == "object" else "swap"
    if p["stage"] == "prepare":
        spec["target"] = media_type.ensure_extension(gate_common.download(p["target"]["bucket"], p["target"]["path"],
                                                                          work_dir, "target", log), ("video",),
                                                     name="target", probe=True)
        s0 = float(p.get("start_s") or 0)
        spec.update(start_s=s0, dur_s=(float(p["end_s"]) - s0) if p.get("end_s") is not None else None, crop=p.get("crop"))
    else:
        spec["tar_in"] = gate_common.download(p["session"]["bucket"], p["session"]["path"], work_dir, "session_in", log)
        if p["stage"] == "paint" and p.get("mode") == "object":
            spec["refs"] = [media_type.ensure_extension(gate_common.download(r["bucket"], r["path"], work_dir,
                                                                             f"ref{i + 1}", log), ("image",),
                                                        name="reference picture", probe=True)
                            for i, r in enumerate(p.get("refs") or [])]
            spec.update({k: p[k] for k in ("name", "object", "replacement", "edit", "instruction", "keyframes",
                                           "others", "shots_with_object", "note") if p.get(k) is not None})
        elif p["stage"] == "paint":
            spec["photo"] = media_type.ensure_extension(gate_common.download(p["face"]["bucket"], p["face"]["path"],
                                                                             work_dir, "face", log), ("image",),
                                                        name="face picture", probe=True)
            spec.update({k: p[k] for k in ("name", "identity", "outfit", "person", "keyframes", "others",
                                           "shots_with_person", "kind", "note") if p.get(k) is not None})
        else:
            spec["repaint"] = p["repaint"]
    # the sid lives in the session tar: reuse it when continuing a session
    if p["stage"] != "prepare":
        import tarfile
        with tarfile.open(spec["tar_in"]) as t:
            spec["sid"] = t.getnames()[0].split("/")[0]
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

    proc.run_streaming([py, "-u", os.path.join(RM_DIR, "driver.py"), "paint", spec_path], cwd=RM_DIR,
                       on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)
    report = json.load(open(spec["report_out"], encoding="utf-8"))
    db.set_phase(jid, "uploading", 96)
    sess = os.path.join(spec["home"], "sessions", spec["sid"])
    pics = list(report.get("strips") or []) + list(report.get("comparisons") or [])
    pics += [report["sheet"]] if report.get("sheet") else []
    pics += [w["sheet"] for w in report.get("windows") or [] if not w.get("passthrough")]
    pics += [f"case/paint/g2/f{f}.png" for f in report.get("keyframes") or []]
    uploaded = {}
    for rel in pics:
        local = os.path.join(sess, *rel.split("/"))
        if os.path.exists(local):
            mime = "image/png" if rel.endswith(".png") else "image/jpeg"
            uploaded[rel] = f"outputs/{jid}/{rel}"
            db.upload_file(uploaded[rel], local, mime)
    report["uploaded"] = uploaded
    report["session"] = {"bucket": "renders", "path": f"outputs/{jid}.tgz"}
    json.dump(report, open(spec["report_out"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    db.upload_file(f"outputs/{jid}-report.json", spec["report_out"], "application/json")
    heartbeat.progress = 97
    log(f"remake_paint {p['stage']} done: {len(uploaded)} picture(s), keyframes {report.get('keyframes')}")
    return spec["tar_out"], "tgz", "application/gzip"
