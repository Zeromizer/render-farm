"""planar_patch runner: tracked replacement of flat elements (plates, badges, labels) in a clip.

params (jsonb):
  planar_patch:
    org_id / job_id       uuids, for scoping and logging only
    source: {bucket, path}  the clip to fix (an earlier video_gen output, typically)
    patches: [            applied in order, each a corner pin driven by a planar track
      {
        name              label for logs and the proof sheet
        library           "<subject>/<element>" from worker/patch/library (artwork, alpha and
                          tuned defaults bundled on the worker), e.g. "atto3evo/plate"
        artwork           {bucket, path}   PNG/JPG of the clean element (overrides library)
        alpha             {bucket, path}   grayscale mask of the artwork's own size (letters-only
                          elements); absent = rounded rectangle (plates)
        key_frame         frame with the clearest view (default: middle)
        key_box           [x0, y0, x1, y1] of the WHOLE artwork rectangle on the key frame; absent
                          = template match (`template`: gray | edges | none)
        min_score         reject a template match weaker than this (0.35)
        refine            re-detect the dark rectangle per frame (plates: true; lettering: false)
        smooth / win      corner smoothing half-window (2) and tracking window scale (1.0)
        clear             inpaint the artwork rect (shrunk by this fraction) before pasting, so
                          generated letters do not ghost under a letters-only alpha (0.03 for badges)
        blur / feather    artwork softness (1.2) and alpha edge feather (2.0), px
        match             brightness-match the artwork to the footage inside the quad (true)
      }
    ]
    debug                 true keeps per-12th-frame overlay PNGs in the work dir (worker-side only)

Output: mp4 at outputs/<job_id>.mp4 like every engine, plus a sibling proof sheet at
outputs/<job_id>-proof.png (original over patched, zoomed on each element at five frames).

WHY THIS ENGINE EXISTS: H3 renders of a specific car get the plate and the tailgate badge wrong in
some frames ("2026" drifts, "ATTO 3" becomes "ATTO 5") even with a subject LoRA, reference images
and a RefMod. Regenerating rolls the dice again. A planar track + corner pin of the clean artwork
(what Mocha/After Effects do by hand) fixes it deterministically in seconds, CPU only. The compute
lives in worker/patch/planar.py in its own cached venv (opencv-python-headless); this side owns
downloads, params and rows, the same split as matte.
"""
import json
import os

import db
import proc
from venvs import venv_python
from runners import gate_common

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR = os.path.join(_WORKER_DIR, "patch")
LIBRARY_DIR = os.path.join(PATCH_DIR, "library")

# Keys the platform may set per patch and forward verbatim to planar.py.
TUNABLE = ("key_frame", "key_box", "template", "min_score", "refine", "smooth", "win",
           "clear", "blur", "feather", "match")


def resolve_library(ref):
    """'<subject>/<element>' -> dict with absolute artwork/alpha paths and tuned defaults."""
    if not isinstance(ref, str) or ref.count("/") != 1 or ".." in ref:
        raise RuntimeError(f"library must be '<subject>/<element>', got {ref!r}")
    subject, element = ref.split("/")
    index = os.path.join(LIBRARY_DIR, subject, "library.json")
    if not os.path.exists(index):
        raise RuntimeError(f"no patch library for subject {subject!r} on this worker")
    with open(index, encoding="utf-8") as f:
        entries = json.load(f)
    entry = entries.get(element)
    if entry is None:
        raise RuntimeError(f"library {subject!r} has no element {element!r}; has {sorted(entries)}")
    out = dict(entry)
    for key in ("artwork", "alpha"):
        if out.get(key):
            out[key] = os.path.join(LIBRARY_DIR, subject, out[key])
    return out


def build_spec(p, work_dir, fetch):
    """params.planar_patch -> planar.py spec. `fetch(obj, name)` downloads {bucket, path} to a
    local path; pure apart from that so tests can cover the merge rules."""
    patches = p.get("patches") or []
    if not patches:
        raise RuntimeError("planar_patch needs at least one entry in params.planar_patch.patches")
    spec_patches = []
    for i, raw in enumerate(patches):
        name = raw.get("name") or f"patch{i}"
        entry = resolve_library(raw["library"]) if raw.get("library") else {}
        merged = {k: v for k, v in entry.items() if k not in ("artwork", "alpha")}
        for k in TUNABLE:
            if raw.get(k) is not None:
                merged[k] = raw[k]
        artwork = entry.get("artwork")
        alpha = entry.get("alpha")
        if raw.get("artwork"):
            artwork = fetch(raw["artwork"], f"{name}-artwork")
            alpha = None  # a custom artwork never inherits the library's mask
        if raw.get("alpha"):
            alpha = fetch(raw["alpha"], f"{name}-alpha")
        if not artwork:
            raise RuntimeError(f"patch {name!r} needs library or artwork")
        merged.update({"name": name, "artwork": artwork, "alpha": alpha})
        spec_patches.append(merged)
    return {"patches": spec_patches}


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    p = (job.get("params") or {}).get("planar_patch") or {}
    src = p.get("source") or {}
    if not (p.get("org_id") and p.get("job_id") and src.get("bucket") and src.get("path")):
        raise RuntimeError("planar_patch needs params.planar_patch.{org_id, job_id, source.bucket, source.path, patches}")

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}

    db.set_phase(jid, "downloading", 1)
    video = gate_common.download(src["bucket"], src["path"], work_dir, "source", log)

    def fetch(obj, name):
        if not (isinstance(obj, dict) and obj.get("bucket") and obj.get("path")):
            raise RuntimeError(f"{name}: expected {{bucket, path}}, got {obj!r}")
        return gate_common.download(obj["bucket"], obj["path"], work_dir, name, log)

    spec = build_spec(p, work_dir, fetch)
    out_local = os.path.join(work_dir, "patched.mp4")
    proof_local = os.path.join(work_dir, "patched-proof.png")
    spec.update({"clip": video, "out": out_local, "proof": proof_local,
                 "debug_dir": os.path.join(work_dir, "debug") if p.get("debug") else None})
    spec_path = os.path.join(work_dir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    log("patches: " + ", ".join(f"{x['name']}({os.path.basename(x['artwork'])})" for x in spec["patches"]))
    heartbeat.progress = 3

    db.set_phase(jid, "building venv", 3)
    py = venv_python(os.path.join(PATCH_DIR, "requirements.txt"), log, run_kw)

    measured = []

    def on_line(line):
        if line.startswith("PHASE "):
            db.set_phase(jid, line[6:].strip()[:60])
            return
        if line.startswith("PROGRESS "):
            try:
                heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.9)
            except ValueError:
                pass
            return
        if line.startswith("MEASURED"):
            measured.append(line)
        log(f"  {line}")

    proc.run_streaming([py, "-u", os.path.join(PATCH_DIR, "planar.py"), spec_path], cwd=PATCH_DIR,
                       on_line=on_line, cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    if not os.path.exists(out_local):
        raise RuntimeError("planar.py finished but produced no output file")
    for line in measured:
        log(line.strip())
    log(f"planar_patch done -> {os.path.getsize(out_local)} bytes")

    # Proof sheet is a sibling of the output at a derivable path, same convention as matte.
    if os.path.exists(proof_local):
        try:
            remote = db.upload_file(f"outputs/{jid}-proof.png", proof_local, "image/png")
            log(f"proof sheet -> {remote} ({os.path.getsize(proof_local)} bytes)")
        except Exception as exc:  # noqa: BLE001 - the patched clip is the deliverable
            log(f"proof sheet upload failed (clip itself is fine): {exc}")
    return out_local, "mp4", "video/mp4"
