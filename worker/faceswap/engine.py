"""face_swap engine driver: runs the accepted fsh3 stages (fsh3.py, the spike's code, unchanged in method) for
one farm job, on the worker's own ComfyUI instead of a one-shot pod.

  engine.py run <spec.json>

spec: work_dir, face, target (local files), consent (text), cloud (bool: 10 s pieces on a cloud card, 5 s on the
PC), comfy_dir, comfy_url, report, proof, and the job's settings:
  subject man|woman|person, person, scene, props [{object, description}], hair from_photo|from_video,
  start_s, end_s, tries, seed, chin_clean, destubble

Stdout: PHASE <text> / PROGRESS <0-100> lines for the runner, anything else is log.
Run in the planar_patch venv (opencv-python-headless + numpy).
"""
import json
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "extract"))
from common import find_ffmpeg_tool  # noqa: E402

SUBJECTS = {"man": ("the man", "his", "He"), "woman": ("the woman", "her", "She"),
            "person": ("the person", "their", "They")}
SEED_STEP = 1


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


def setup_env(spec):
    """fsh3/comfy read ComfyUI's location at import and call ffmpeg from PATH."""
    os.environ["COMFYUI_URL"] = spec["comfy_url"]
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.basename(os.path.normpath(spec["work_dir"])))[:40]
    os.environ["FSH3_SUBFOLDER"] = f"fs_{tag}"          # this job's inputs, apart from any other job's
    os.environ["FSH3_FETCH_DIR"] = os.path.join(spec["work_dir"], "comfy_out")
    ff = find_ffmpeg_tool("ffmpeg")
    os.environ["PATH"] = os.path.dirname(ff) + os.pathsep + os.environ.get("PATH", "")


def config_for(spec):
    """The fsh3 config.json for a job's settings. Texts given by the caller are used as they are (the agent sees
    the photo and the video); only what is left empty is filled by fsh3.describe (the local Qwen3-VL 4B)."""
    subject, his, He = SUBJECTS[spec.get("subject") or "person"]
    wears = "wear" if He == "They" else "wears"
    person = (spec.get("person") or "").strip().rstrip(".")
    scene = (spec.get("scene") or "").strip().rstrip(".")
    tries = int(spec.get("tries") or 2)
    seed = int(spec.get("seed") if spec.get("seed") is not None else 43)
    return {
        "subject": subject,
        "possessive": his,
        "face_desc": f"{his} face, {person}" if person else "",
        "scene_desc": (f"the same hand gestures, the same framing and camera motion. {He} {wears} {scene}."
                       if scene else ""),
        "props": [{"noun": p["object"],
                   **({"sentence": f"{He} {wears} the {p['description'].rstrip('.')} from {{pic}}, worn exactly as in "
                                   f"<Video 1>."} if p.get("description") else {})}
                  for p in spec.get("props") or []],
        "chin_clean": spec.get("chin_clean", True) is not False,
        "destubble": bool(spec.get("destubble")),
        "seeds": [seed + SEED_STEP * k for k in range(tries)],
    }


def trimmed(spec, dest):
    s0, s1 = spec.get("start_s"), spec.get("end_s")
    if s0 is None and s1 is None:
        return spec["target"]
    cmd = [find_ffmpeg_tool("ffmpeg"), "-v", "error", "-y", "-ss", f"{float(s0 or 0):.3f}", "-i", spec["target"]]
    if s1 is not None:
        cmd += ["-t", f"{float(s1) - float(s0 or 0):.3f}"]
    subprocess.run(cmd + ["-c:v", "libx264", "-crf", "10", "-c:a", "aac", "-b:a", "192k", dest], check=True)
    return dest


def local_runner(fsh3, comfy, on_render):
    """fsh3.render(run=...): render the pending graphs on this machine's ComfyUI. The graphs name their inputs
    face.jpg / <prop>.png / ref_000.mp4; each is uploaded into the job's input subfolder and the graph pointed at
    it."""
    def run(job):
        comfy.free()            # SAM 3.1 / the VLM out before H3 loads
        for gpath, s in fsh3.pending(job):
            g = json.load(open(gpath))
            for k, node in g.items():
                if node.get("class_type") == "LoadImage" and k.startswith("load_ref_image_"):
                    node["inputs"]["image"] = comfy.put_input(job.p(node["inputs"]["image"]))
                elif node.get("class_type") == "LoadVideo":
                    node["inputs"]["file"] = comfy.put_input(job.p("pieces", node["inputs"]["file"]))
            seed = int(re.search(r"_s(\d+)", os.path.basename(gpath)).group(1))
            g["save"]["inputs"]["filename_prefix"] = f"{os.environ['FSH3_SUBFOLDER']}/p{s:03d}_s{seed}"
            emit("PHASE", f"rendering piece at {s / fsh3.FPS:.1f} s, seed {seed}")
            files = [f for f in comfy.run(g) if f.endswith(".mp4")]
            if not files:
                raise RuntimeError(f"ComfyUI returned no video for piece {s} seed {seed}")
            os.makedirs(job.p("renders"), exist_ok=True)
            shutil.copyfile(files[0], fsh3.render_path(job, s, seed))
            on_render()
    return run


def proof_sheet(fsh3, job, dest, cols=6):
    """The side-by-side (video | result) at six moments, in one row."""
    import cv2
    import numpy as np
    sbs = job.p("final_side-by-side.mp4")
    n = fsh3.count(sbs)
    at = [int(i * (n - 1) / (cols - 1)) for i in range(cols)]
    seq, tiles = fsh3.Seq(sbs), []
    for i in at:
        f = seq.get(i)
        tiles.append(cv2.resize(f, (f.shape[1] * 360 // f.shape[0], 360), interpolation=cv2.INTER_AREA))
    band = np.hstack(tiles)
    head = np.full((22, band.shape[1], 3), 30, np.uint8)
    cv2.putText(head, "video | swapped, at " + ", ".join(f"{i / fsh3.FPS:.1f}s" for i in at), (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(dest, np.vstack([head, band]))


def run(spec):
    setup_env(spec)
    import fsh3
    import comfy
    if not spec.get("cloud"):
        # the 16 GB PC peaked at ~31/31 GB host RAM on 5 s at 448x768; 10 s there was never tried
        fsh3.PIECE, fsh3.STRIDE = 124, 100
    job = fsh3.Job("fs", root=spec["work_dir"])
    job.save_cfg(config_for(spec))

    emit("PHASE", "prep")
    fsh3.prep(job, spec["face"], trimmed(spec, job.p("target_in.mp4")), spec["consent"])
    emit("PROGRESS", 5)
    emit("PHASE", "tracking face, hair, hands and props")
    fsh3.masks(job)
    emit("PROGRESS", 15)
    c = job.cfg
    if c["props"]:
        emit("PHASE", "cutting out the props")
        fsh3.cutout(job)
    c = job.cfg
    if not c["face_desc"] or not c["scene_desc"] or any(not p.get("sentence") for p in c["props"]):
        emit("PHASE", "describing the photo and the video")
        fsh3.describe(job)
    emit("PROGRESS", 20)
    if spec.get("hair", "from_photo") == "from_video":
        fsh3.ff("-i", job.src, "-c", "copy", job.p("ref.mp4"))
    else:
        emit("PHASE", "painting the hair out")
        fsh3.erase_hair(job)
    emit("PHASE", "cutting pieces")
    fsh3.pieces(job)
    n_pieces = len(fsh3.pending(job))
    done = [0]

    def on_render():
        done[0] += 1
        emit("PROGRESS", 22 + min(63, int(63 * done[0] / max(1, n_pieces * 1.3))))
    picks = fsh3.render(job, run=local_runner(fsh3, comfy, on_render))
    comfy.free()                # H3 out before SAM 3.1 finds the new faces
    emit("PROGRESS", 86)
    emit("PHASE", "chin clean-up, stitch, sound")
    fsh3.finish(job)
    emit("PROGRESS", 96)

    final = json.load(open(job.p("final.json")))
    cs = final.get("sync_windows") or []
    renders = os.listdir(job.p("renders"))
    pieces = []
    for s in sorted(picks, key=int):
        f = os.path.basename(picks[s]["file"])
        pieces.append({"start": int(s), "seed": int(re.search(r"_s(\d+)_", f).group(1)),
                       "tries": sum(1 for r in renders if r.startswith(f"p{int(s):03d}_s") and r.endswith(".mp4")),
                       "mean": round(picks[s]["mean"], 3), "min": round(picks[s]["min"], 3)})
    warnings = [f"the piece from {p['start'] / fsh3.FPS:.1f} s follows the video loosely (sync {p['mean']}/{p['min']} "
                f"after {p['tries']} tries)" for p in pieces
                if p["mean"] < c["sync_min_mean"] or p["min"] < c["sync_min_window"]]
    bad = [round((1 + i * 24) / fsh3.FPS, 1) for i, v in enumerate(cs) if v < 0.3]
    if bad:
        warnings.append(f"the motion drifts from the video around {bad} s; check those moments")
    report = {"frames": fsh3.count(job.p("final.mp4")), "canvas": list(job.size), "pieces": pieces,
              "sync_mean": round(sum(cs) / len(cs), 3) if cs else None, "sync_min": round(min(cs), 3) if cs else None,
              "sync_windows": cs, "prompt": final.get("prompt"), "provenance": {k: final.get(k) for k in (
                  "face_sha256", "target_sha256", "consent", "recorded_utc")},
              "final_sha256": final.get("final_sha256"), "warnings": warnings}
    with open(spec["report"], "w") as f:
        json.dump(report, f, indent=1)
    proof_sheet(fsh3, job, spec["proof"])
    shutil.copyfile(job.p("final.mp4"), spec["out"])
    shutil.copyfile(job.p("final_side-by-side.mp4"), spec["side_by_side"])
    emit("PROGRESS", 100)
    print(f"face_swap: {len(pieces)} piece(s), sync mean {report['sync_mean']} min {report['sync_min']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        sys.exit("usage: engine.py run <spec.json>")
    run(json.load(open(sys.argv[2])))
