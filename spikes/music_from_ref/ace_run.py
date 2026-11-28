"""ACE-Step 1.5 arms, run inside ACE-Step's own uv environment (cwd = its checkout).

Per reference (state from analyze.py prep, ~/music-from-ref/<ref>/):
  understand  ACE's own LM describes the reference's MUSIC stem (no voice): caption, bpm, key.
              Saved as ace_understand.json; the YuE2 job also takes its style words from it.
  B1 (style)  text2music with reference_audio = the music stem. Per the authors, the reference
              latents carry timbre / mix / playing style, averaged over time, with melody and
              rhythm removed. Plus the measured bpm + key, the reference's length, instrumental.
              LM planning ("thinking") on, as the authors default.
  B2 (cover)  task "cover" from the music stem at audio_cover_strength 0.3 (the authors: 0.2 for
              style transfer, 1.0 = strict structure). Keeps more groove, risks the tune.
Outputs ~/music-from-ref/takes/<ref>/<arm>_<seed>.wav (trimmed to the reference length, short
fade) and copies to <out>/.

Job JSON: {"refs": [{"name": "taycan"}, ...], "seeds": [11, 222], "cover_strength": 0.3,
           "lm": "acestep-5Hz-lm-1.7B", "dit": "acestep-v15-turbo"}
"""
import json
import os
import re
import shutil
import sys
import time
import traceback

import numpy as np
import soundfile as sf

ACE_HOME = os.getcwd()
sys.path.insert(0, ACE_HOME)
HOME = os.path.join(os.path.expanduser("~"), "music-from-ref")
T0 = time.monotonic()


def log(msg):
    print(msg, flush=True)


def clean_caption(text):
    """Drop sentences / clauses about voices; this bed is instrumental."""
    parts = re.split(r"(?<=[.;])\s+|,\s+", text or "")
    keep = [p for p in parts if not re.search(r"vocal|voice|sing|singer|rap|lyric|choir|spoken|narrat|speech", p, re.I)]
    out = ", ".join(p.strip().rstrip(".,;") for p in keep if p.strip())
    return out[:480]


def ace_key(k):
    """'C# minor' from analyze.py's key dict."""
    return f"{k['key'].split()[0]} {k['mode']}"


def finish(src, dest, seconds):
    y, sr = sf.read(src, always_2d=True)
    n = min(len(y), int(round(seconds * sr)))
    y = y[:n].copy()
    fade = int(min(2.0, max(0.8, 0.05 * seconds)) * sr)
    y[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    sf.write(dest, y, sr, subtype="PCM_16")
    return len(y) / sr, (sf.info(src).duration)


def main():
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music, understand_music
    from acestep.llm_inference import LLMHandler

    job_path, out = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    seeds = job.get("seeds", [11, 222])
    strength = float(job.get("cover_strength", 0.3))
    os.makedirs(out, exist_ok=True)
    results = {"commit": None, "takes": [], "understand": {}}
    try:
        results["commit"] = open(os.path.join(ACE_HOME, ".git", "HEAD")).read().strip()
    except Exception:
        pass

    def save():
        results["elapsed_s"] = int(time.monotonic() - T0)
        with open(os.path.join(out, "ace_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=ACE_HOME, config_path=job.get("dit", "acestep-v15-turbo"), device="cuda",
                                     offload_to_cpu=True)
    log(f"DiT init ok={ok}: {msg}")
    if not ok:
        raise SystemExit(1)
    llm = LLMHandler()
    msg, ok = llm.initialize(checkpoint_dir=os.path.join(ACE_HOME, "checkpoints"), lm_model_path=job.get("lm", "acestep-5Hz-lm-1.7B"),
                             backend=job.get("lm_backend", "pt"), device="cuda", offload_to_cpu=True)
    log(f"LM init ok={ok}: {msg}")
    if not ok:
        raise SystemExit(1)
    tmp = os.path.join(HOME, "_ace_tmp")
    refs = job["refs"]
    for ri, ref in enumerate(refs):
        name = ref["name"]
        d = os.path.join(HOME, name)
        with open(os.path.join(d, "ref.json"), encoding="utf-8") as f:
            info = json.load(f)
        stem = os.path.join(d, "no_vocals.wav")
        dur = float(info["duration_s"])
        bpm = int(round(info["bpm"]))
        key = ace_key(info["key"])
        t = time.monotonic()
        codes = dit.convert_src_audio_to_codes(stem)
        if not codes or codes.startswith("❌"):
            log(f"{name}: codes failed: {codes[:300]}")
            u = None
        else:
            u = understand_music(llm, codes)
        und = {"caption": getattr(u, "caption", ""), "bpm": getattr(u, "bpm", None), "keyscale": getattr(u, "keyscale", ""),
               "timesignature": getattr(u, "timesignature", ""), "success": getattr(u, "success", False),
               "error": getattr(u, "error", None), "seconds": round(time.monotonic() - t, 1)}
        und["caption_instrumental"] = clean_caption(und["caption"])
        results["understand"][name] = und
        with open(os.path.join(d, "ace_understand.json"), "w", encoding="utf-8") as f:
            json.dump(und, f, indent=1, ensure_ascii=False)
        log(f"{name}: understand {json.dumps(und, ensure_ascii=False)[:600]}")
        caption = und["caption_instrumental"] or "instrumental, punchy modern drums, confident"
        caption = caption + ", instrumental, no vocals"
        save()
        arms = [("B1", dict(task_type="text2music", reference_audio=stem, thinking=True)),
                ("B2", dict(task_type="cover", src_audio=stem, audio_cover_strength=strength, thinking=False))]
        for arm, extra in arms:
            for seed in seeds:
                tag = f"{arm}_{seed}"
                log(f"PROGRESS {int(100 * (ri + 0.5) / len(refs))}")
                log(f"=== {name} {tag}")
                row = {"ref": name, "take": tag, "arm": arm, "seed": seed, "caption": caption, "bpm": bpm, "keyscale": key,
                       "duration": round(max(10.0, dur), 2)}
                try:
                    params = GenerationParams(caption=caption, lyrics="[Instrumental]", instrumental=True, bpm=bpm, keyscale=key,
                                              timesignature="4", duration=max(10.0, dur), inference_steps=8, shift=3.0,
                                              guidance_scale=1.0, seed=seed, **extra)
                    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
                    sd = os.path.join(tmp, name, tag)
                    shutil.rmtree(sd, ignore_errors=True)
                    t = time.monotonic()
                    res = generate_music(dit, llm, params=params, config=config, save_dir=sd)
                    row["wall_s"] = round(time.monotonic() - t, 1)
                    if not res.success or not res.audios:
                        row["error"] = str(getattr(res, "error", None) or getattr(res, "status_message", ""))[:500]
                    else:
                        src = res.audios[0].get("path")
                        dest = os.path.join(HOME, "takes", name, f"{tag}.wav")
                        row["length_s"], row["raw_s"] = finish(src, dest, dur)
                        shutil.copyfile(dest, os.path.join(out, f"{name}__{tag}.wav"))
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:500]
                    log(traceback.format_exc()[-1500:])
                results["takes"].append(row)
                log(f"{name} {tag}: {json.dumps({k: row.get(k) for k in ('wall_s', 'length_s', 'raw_s', 'error')})}")
                save()
    save()


if __name__ == "__main__":
    main()
