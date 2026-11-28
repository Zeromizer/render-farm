"""Music-from-reference spike: YuE2 arms (farm python job; venv = ../yue2_official/requirements.txt,
the official yue2-infer runtime, plus the PC's ComfyUI int8 YuE2 + instrumental LoRA).

Per reference (prep state in ~/music-from-ref/<ref>/: ref.json, no_vocals.wav, ace_understand.json):
  A   conform   YuE2's official planner writes a NEW score from the reference's style words
                (ACE's caption of the music stem) + measured BPM + key; conform_score.py puts it on
                the reference's clock (exact tempo, beat phase via a head trim, first chorus on the
                lift); ComfyUI int8 YuE2 + LoRA plays it (the 13 Nov hybrid, no voices).
  A2  chords    SheetSage2 "full" transcript of the reference's music stem with EVERY note turned
                into a rest: only its tempo, key, bar layout, sections and chord symbols remain.
                YuE2 + LoRA plays that skeleton in the same style words.
  X   copy      SheetSage2 melody transcript -> YuE2 melody mode (the Oct cover spike). A positive
                control for the melody-overlap measure only; never offered for use.
Takes: ~/music-from-ref/takes/<ref>/<arm>_<seed>.wav, trimmed to the reference length + fade.

params.args = [<job json, repo-relative>, <out dir>]
Job JSON: {"refs": [{"name": "taycan"}], "seeds": [11, 222], "copy_seed": 11}
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "yue2_official"))
import run_official as ro  # noqa: E402
from analyze import comfy_prompt, ensure_comfy, sheetsage  # noqa: E402

HOME = os.path.join(os.path.expanduser("~"), "music-from-ref")
log = ro.log


def style_words(name, info):
    und = {}
    p = os.path.join(HOME, name, "ace_understand.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            und = json.load(f)
    words = (und.get("caption_instrumental") or "").strip() or "modern pop, punchy drums, confident"
    words = re.sub(r"\s+", " ", words)[:300]
    key = info["key"]["key"].replace(" major", " major").replace(" minor", " minor")
    return f"Instrumental, {words}, {int(round(info['bpm']))} BPM, in {key}, no vocals, no singing, no choir, no spoken words"


def skeleton(abc):
    """Every note -> a rest of the same length; ties dropped; chord symbols, bar lines, sections,
    voices and header kept. Returns (text, sections [(label, bars)], bar_seconds)."""
    out, sections, voice = [], [], None
    q = m = None
    for line in abc.splitlines():
        s = line.strip()
        if s.startswith("Q:"):
            q = int(s.split("=")[1])
        if s.startswith("M:"):
            m = s[2:].strip()
        if re.match(r"^[A-Za-z]:", s) or s.startswith("V:") or not s:
            if s.startswith("V:") and "clef" not in s:
                voice = s[2:].strip().split()[0]
            out.append(line)
            continue
        if s.startswith("%"):
            sections.append([s[1:].strip() or "verse", 0])
            out.append(line)
            continue
        pieces = re.split(r'("[^"]*")', s)
        for k in range(0, len(pieces), 2):
            seg = re.sub(r"\[[^\]]*\]([0-9]*/?[0-9]*)", r"z\1", pieces[k])
            seg = re.sub(r"[\^_=]*[A-Ga-g][',]*([0-9]*/?[0-9]*)", r"z\1", seg)
            pieces[k] = seg.replace("-", "")
        new = "".join(pieces)
        out.append(new)
        if voice == "Vocal" and sections:
            for tok in [t for t in new.split("|") if t.strip()]:
                mm = re.fullmatch(r"Z([0-9]*)", tok.strip())
                sections[-1][1] += int(mm.group(1) or 1) if mm else 1
    n, d = (int(x) for x in (m or "4/4").split("/"))
    bar_s = (4 * n / d) * 60 / (q or 120)
    return "\n".join(out) + "\n", [(l, b) for l, b in sections if b], bar_s


def render(name, tag, abc_text, style, tags, seconds, seed, mode="full", head_trim=0.0, max_extra=6.0):
    """ComfyUI int8 YuE2 + instrumental LoRA (as audio_gen), a given ABC. Returns facts."""
    import numpy as np
    import soundfile as sf
    max_duration = round(round((seconds + head_trim + max_extra) / 0.04) * 0.04, 2)
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ro.CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": ro.LORA,
                                                     "strength_model": 0.0, "strength_clip": 1.0}},
        "3": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": abc_text}},
        "5": {"class_type": "YuE2GenerateMusic",
              "inputs": {"clip": ["2", 1], "style": style, "lyrics": tags, "seed": int(seed), "mode": mode, "abc": ["3", 0],
                         "max_duration": max_duration, "temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["5", 1], "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "seed": int(seed), "steps": 32, "cfg": 1.0, "sampler_name": "dpm_2",
                         "scheduler": "sgm_uniform", "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": f"music_from_ref/{name}_{tag}"}},
    }
    t0 = time.monotonic()
    outputs = comfy_prompt(g, timeout_s=1200)
    wall = round(time.monotonic() - t0, 1)
    raw = os.path.join(HOME, "_raw", f"{name}__{tag}.flac")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    for node_out in outputs.values():
        for f in node_out.get("audio", []) or []:
            q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
            with urllib.request.urlopen(ro.COMFY + "/view?" + q, timeout=120) as resp, open(raw, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            break
    y, sr = sf.read(raw, always_2d=True)
    raw_s = len(y) / sr
    a = int(round(head_trim * sr))
    y = y[a:a + int(round(seconds * sr))].copy()
    fade = int(min(2.0, max(0.8, 0.05 * seconds)) * sr)
    if len(y) > fade:
        y[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    dest = os.path.join(HOME, "takes", name, f"{tag}.wav")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    sf.write(dest, y, sr, subtype="PCM_16")
    return {"wall_s": wall, "raw_s": round(raw_s, 2), "length_s": round(len(y) / sr, 2), "max_duration": max_duration,
            "short": len(y) / sr < seconds - 0.5}


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    job_path, out = sys.argv[1], sys.argv[2]
    job_path = job_path if os.path.isabs(job_path) else os.path.join(os.getcwd(), job_path)
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    threading.Thread(target=ro.monitor, daemon=True).start()
    py = sys.executable
    seeds = job.get("seeds", [11, 222])
    results = {"takes": [], "plans": []}

    def save():
        results["elapsed_s"] = ro.elapsed()
        results["vram_peak_mib"] = dict(ro._state["peak"])
        with open(os.path.join(out, "yue_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

    refs = []
    for ref in job["refs"]:
        with open(os.path.join(HOME, ref["name"], "ref.json"), encoding="utf-8") as f:
            info = json.load(f)
        refs.append((ref["name"], info, style_words(ref["name"], info)))

    # ---- phase 1: official planner (GPU, ComfyUI emptied first), conform each plan
    ro.set_phase("comfy-free")
    log("comfy: " + ro.comfy_free())
    choice, found = ro.probe_attention(py)
    log(f"attention probe: {found} -> {choice}")
    if choice != "flash":
        os.environ["YUE2_ATTENTION"] = choice
        os.environ["PYTHONPATH"] = os.path.join(ro.HERE, "shim") + os.pathsep + os.environ.get("PYTHONPATH", "")
    plans = []
    for name, info, style in refs:
        for seed in seeds:
            tag = f"A_{seed}"
            ro.set_phase(f"plan:{name}:{tag}")
            log(f"=== plan {name} {tag}: {style}")
            work = os.path.join(HOME, "_plans", f"{name}__{tag}")
            shutil.rmtree(work, ignore_errors=True)
            pl = work + ".lyrics.txt"
            os.makedirs(os.path.dirname(pl), exist_ok=True)
            with open(pl, "w", encoding="utf-8", newline="\n") as f:
                f.write("[Intro]\n\n[Chorus]\n\n[Outro]\n")
            cmd = [py, "-B", os.path.join(ro.SCRIPTS, "instrumental.py"), "run", "--style", style, "--planning-lyrics-file", pl,
                   "--seed", str(seed), "--id", f"{name}_{tag}", "--models-root", ro.MODELS, "--offline", "--prepare-only",
                   "--output", work]
            code, secs = ro.stream(cmd, os.path.join(out, f"plan_{name}_{tag}.log"))
            row = {"ref": name, "take": tag, "plan_exit": code, "plan_s": secs, "style": style}
            src = os.path.join(work, "conversion", "score.abc")
            if code == 0 and os.path.isfile(src):
                cut = os.path.join(HOME, "_plans", f"{name}__{tag}.conformed.abc")
                r = subprocess.run([py, "-B", os.path.join(HERE, "conform_score.py"), src, cut, "--seconds", str(info["duration_s"]),
                                    "--bpm", str(info["bpm"]), "--first-beat", str(info.get("first_beat") or 0.0)] +
                                   (["--lift", str(info["lift_s"])] if info.get("lift_s") is not None else []),
                                   capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"), timeout=120)
                log("conform: " + (r.stdout.strip() or r.stderr.strip()[-600:]))
                if r.returncode == 0:
                    row["conform"] = json.loads(r.stdout.strip().splitlines()[-1])
                    row["abc"] = cut
                    ps = os.path.join(work, "prepared", "style.txt")
                    if os.path.isfile(ps):
                        with open(ps, encoding="utf-8") as f:
                            row["prepared_style"] = f.read().strip()
                    shutil.copyfile(src, os.path.join(out, f"{name}__{tag}.plan.abc"))
                    shutil.copyfile(cut, os.path.join(out, f"{name}__{tag}.conformed.abc"))
                else:
                    row["conform_error"] = r.stderr.strip()[-600:]
            results["plans"].append({k: v for k, v in row.items() if k != "abc"})
            plans.append(row)
            save()

    # ---- phase 2: ComfyUI renders (A conformed, A2 chord skeleton, X melody copy)
    ensure_comfy()
    jobs = []
    for row in plans:
        if row.get("abc"):
            info = next(i for n, i, _ in refs if n == row["ref"])
            c = row["conform"]
            with open(row["abc"], encoding="utf-8") as f:
                text = f.read()
            jobs.append(dict(name=row["ref"], tag=row["take"], abc_text=text, style=row.get("prepared_style") or row["style"],
                             tags=ro.timed_tags(c["sections"], c["bar_seconds"]), seconds=info["duration_s"],
                             seed=int(row["take"].split("_")[1]), mode="full", head_trim=c["head_trim_s"]))
    for name, info, style in refs:
        stem = os.path.join(HOME, name, "no_vocals.wav")
        ro.set_phase(f"sheetsage:{name}")
        try:
            full = sheetsage(stem, "full")
            with open(os.path.join(out, f"{name}__sheetsage_full.abc"), "w", encoding="utf-8") as f:
                f.write(full)
            skel, sections, bar_s = skeleton(full)
            with open(os.path.join(out, f"{name}__A2_skeleton.abc"), "w", encoding="utf-8") as f:
                f.write(skel)
            tags = ro.timed_tags(sections, bar_s)
            for seed in seeds:
                jobs.append(dict(name=name, tag=f"A2_{seed}", abc_text=skel, style=style, tags=tags, seconds=info["duration_s"],
                                 seed=seed, mode="full"))
            mel = sheetsage(stem, "melody")
            with open(os.path.join(out, f"{name}__sheetsage_melody.abc"), "w", encoding="utf-8") as f:
                f.write(mel)
            jobs.append(dict(name=name, tag=f"X_{job.get('copy_seed', 11)}", abc_text=mel, style=style, tags=tags,
                             seconds=info["duration_s"], seed=job.get("copy_seed", 11), mode="melody"))
        except Exception as e:
            log(f"{name}: sheetsage failed: {e}")
            results["takes"].append({"ref": name, "take": "A2/X", "error": f"sheetsage: {e}"[:400]})
    for k, jb in enumerate(jobs):
        ro.set_phase(f"render:{jb['name']}:{jb['tag']}")
        log(f"PROGRESS {10 + int(88 * k / max(1, len(jobs)))}")
        log(f"=== render {jb['name']} {jb['tag']} mode={jb['mode']}")
        row = {"ref": jb["name"], "take": jb["tag"], "mode": jb["mode"], "style": jb["style"], "tags": jb["tags"]}
        try:
            row.update(render(**jb))
            shutil.copyfile(os.path.join(HOME, "takes", jb["name"], f"{jb['tag']}.wav"), os.path.join(out, f"{jb['name']}__{jb['tag']}.wav"))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:500]
        row["vram_peak_mib"] = ro._state["peak"].get(f"render:{jb['name']}:{jb['tag']}")
        results["takes"].append(row)
        log(f"{jb['name']} {jb['tag']}: " + json.dumps({k: row.get(k) for k in ("wall_s", "raw_s", "length_s", "short", "error")}))
        save()
    try:
        ro.comfy_post("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass
    save()
    log("PROGRESS 100")


if __name__ == "__main__":
    main()
