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
        clear             inpaint the footage under the element before pasting, so generated letters
                          do not ghost under a letters-only alpha (0.03 for badges). With an alpha the
                          cleared area is the alpha's shape grown by `clear_grow` px (2); without one,
                          the artwork rect shrunk by `clear` per side; `clear_shape` auto|alpha|rect
                          forces one (lettering that swells vs the artwork needs rect, a round badge
                          against a body crease needs alpha). `clear_radius` = inpaint radius (5)
        blur / feather    artwork softness (1.2) and alpha edge feather (2.0), px
        match             brightness-match the artwork to the footage inside the quad (true)
        frames / fade     [first, last] inclusive frame range to track and paste in (turntables:
                          a face is toward the camera for part of the clip; list the artwork twice
                          with two ranges when it comes round twice) and a linear blend of that many
                          frames at each end of the range (0)
      }
    ]
    debug                 true keeps per-12th-frame overlay PNGs in the work dir (worker-side only)

Output: mp4 at outputs/<job_id>.mp4 like every engine, plus a sibling proof sheet at
outputs/<job_id>-proof.png (original over patched, zoomed on each element at five frames).

Inputs may be extensionless content-addressed objects (assets/sha256/<hex>): the source clip is
read by content, artwork/alpha are sniffed and renamed (videogen/media_type.py) before OpenCV
sees them. Clips over 720 frames (30 s at 24 fps) are refused: the whole clip is held in RAM.

Library discovery: publish_library_index() uploads patch/library-index.json to the renders bucket
on worker start (subject -> label, views, elements -> {view, element, notes, artwork size}) so the
platform can map a catalogue model to a subject and offer the patches it actually has.

WHY THIS ENGINE EXISTS: H3 renders of a specific car get the plate and the tailgate badge wrong in
some frames ("2026" drifts, "ATTO 3" becomes "ATTO 5") even with a subject LoRA, reference images
and a RefMod. Regenerating rolls the dice again. A planar track + corner pin of the clean artwork
(what Mocha/After Effects do by hand) fixes it deterministically in seconds, CPU only. The compute
lives in worker/patch/planar.py in its own cached venv (opencv-python-headless); this side owns
downloads, params and rows, the same split as matte.
"""
import json
import os
import struct

import db
import proc
from venvs import venv_python
from runners import gate_common
from videogen import media_type

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR = os.path.join(_WORKER_DIR, "patch")
LIBRARY_DIR = os.path.join(PATCH_DIR, "library")

# Keys the platform may set per patch and forward verbatim to planar.py.
TUNABLE = ("key_frame", "key_box", "template", "min_score", "refine", "smooth", "win",
           "clear", "clear_grow", "clear_radius", "clear_shape", "blur", "feather", "match",
           "frames", "fade")


def resolve_library(ref):
    """'<subject>/<element>' -> dict with absolute artwork/alpha paths and tuned defaults."""
    if not isinstance(ref, str) or ref.count("/") != 1 or ".." in ref:
        raise RuntimeError(f"library must be '<subject>/<element>', got {ref!r}")
    subject, element = ref.split("/")
    if element.startswith("_"):
        raise RuntimeError(f"{ref!r} is metadata, not an element")
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


def _png_size(path):
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            w, h = struct.unpack(">II", head[16:24])
            return [w, h]
    except OSError:
        pass
    return None


def library_index():
    """Every subject library on this worker, for the platform's discovery index."""
    out = {}
    if not os.path.isdir(LIBRARY_DIR):
        return out
    for subject in sorted(os.listdir(LIBRARY_DIR)):
        index = os.path.join(LIBRARY_DIR, subject, "library.json")
        if not os.path.exists(index):
            continue
        with open(index, encoding="utf-8") as f:
            entries = json.load(f)
        meta = entries.get("_subject") or {}
        elements = {}
        for name, e in entries.items():
            if name.startswith("_"):
                continue
            elements[name] = {"view": e.get("view"), "element": e.get("element"), "notes": e.get("notes"),
                              "letters_only": bool(e.get("alpha")),
                              "artwork_size": _png_size(os.path.join(LIBRARY_DIR, subject, e["artwork"]))}
        out[subject] = {"label": meta.get("label", subject), "views": meta.get("views") or
                        sorted({e["view"] for e in elements.values() if e.get("view")}),
                        "notes": meta.get("notes"), "elements": elements}
    return out


INDEX_REMOTE = "patch/library-index.json"


def publish_library_index(log):
    """Upload the index to the renders bucket; a failure is logged, never fatal (worker start)."""
    idx = library_index()
    local = os.path.join(config_cache_dir(), "patch-library-index.json")
    with open(local, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
    remote = db.upload_file(INDEX_REMOTE, local, "application/json")
    log(f"patch library index -> {remote} ({len(idx)} subject(s): {', '.join(sorted(idx)) or 'none'})")
    return remote


def config_cache_dir():
    import config
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return config.CACHE_DIR


def localize_image(local, name):
    """Give a downloaded artwork/alpha the extension its bytes say (assets/sha256/<hex> objects are
    extensionless and cv2.imread needs the suffix). Non-images are refused here, not in OpenCV."""
    return media_type.ensure_extension(local, ("image",), name=name, probe=False)


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
    video = media_type.ensure_extension(video, ("video",), name="source clip", probe=True)

    def fetch(obj, name):
        if not (isinstance(obj, dict) and obj.get("bucket") and obj.get("path")):
            raise RuntimeError(f"{name}: expected {{bucket, path}}, got {obj!r}")
        # Artwork/alpha arrive as extensionless sha256 objects; give them the suffix cv2 needs.
        return localize_image(gate_common.download(obj["bucket"], obj["path"], work_dir, name, log), name)

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
