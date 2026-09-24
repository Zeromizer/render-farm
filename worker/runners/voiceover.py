"""voiceover runner: a narrated voice-over track with per-word timings.

params (jsonb):
  voiceover:
    org_id / job_id     uuids, for scoping and logging only (the platform's
                        wait_for_voiceover scopes on params.voiceover.job_id)
    engine              omnivoice (default) | chatterbox | edge
    lines               [{id, text, say?, pause_after_ms?}]  1-60 lines
                        text = the WRITTEN words (what captions and cues use)
                        say  = the spelling handed to the voice (e.MAS -> ee-Mars);
                               the platform fills it from its pronunciation map
    voice / rate        edge only: voice name and percent
    instruct            omnivoice only: voice design when there is no reference
    language            "en" / "zh" / ...: whisper hint + OmniVoice language
    ref                 optional {bucket, path}: a clip to clone (omnivoice/chatterbox)
    ref_text            optional transcript of ref (omnivoice)
    speed / class_temperature / exaggeration / cfg_weight / gap_ms / lead_ms

Output: vo.wav (48 kHz mono 16-bit, -16 LUFS) uploaded as outputs/<jid>.wav,
plus outputs/<jid>-words.json, and the same timings written back onto the row
as params.voiceover_result = {duration_s, lines, words} so the platform can
hand them to the agent without a second download.

WHY THIS ENGINE EXISTS: the builder could not narrate at all. The voices
already lived on this box (the TTS studio's OmniVoice and Chatterbox, Apache-2.0
and MIT) but only behind a UI; edge-tts was a per-project script. An explainer
is timed to its voice, so the voice has to come with word timings - that is
the part this engine adds over the studio.

GPU: OmniVoice/Chatterbox load for the job (10-20 s) and exit. ComfyUI is asked
to drop its cached models first, and the studio's resident workers (if anyone
turned them back on) are paused, because both together can exceed the card.
"""
import json
import os

import config
import db
import proc
from runners import gate_common
from venvs import venv_python

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VO_DIR = os.path.join(_WORKER_DIR, "voiceover")

ENGINES = ("omnivoice", "chatterbox", "edge")
MAX_LINES = 60
MAX_LINE_CHARS = 600
MAX_TOTAL_CHARS = 6000


def validate(p):
    """Return (engine, lines) or raise with a message the agent can act on."""
    if not (p.get("org_id") and p.get("job_id")):
        raise RuntimeError("voiceover needs params.voiceover.{org_id, job_id}")
    engine = p.get("engine") or "omnivoice"
    if engine not in ENGINES:
        raise RuntimeError(f"unknown voiceover engine {engine!r}; supported are {list(ENGINES)}")
    lines = p.get("lines")
    if not isinstance(lines, list) or not lines:
        raise RuntimeError("voiceover needs params.voiceover.lines: [{id, text, say?}]")
    if len(lines) > MAX_LINES:
        raise RuntimeError(f"voiceover takes at most {MAX_LINES} lines, got {len(lines)}")
    total = 0
    clean = []
    for i, ln in enumerate(lines):
        text = str((ln or {}).get("text") or "").strip()
        if not text:
            raise RuntimeError(f"voiceover line {i + 1} has no text")
        if len(text) > MAX_LINE_CHARS:
            raise RuntimeError(f"voiceover line {i + 1} is {len(text)} chars; split it (max {MAX_LINE_CHARS})")
        total += len(text)
        item = {"id": str(ln.get("id") or f"l{i + 1}"), "text": text,
                "say": str(ln.get("say") or text).strip()}
        if ln.get("pause_after_ms") is not None:
            item["pause_after_ms"] = int(ln["pause_after_ms"])
        clean.append(item)
    if total > MAX_TOTAL_CHARS:
        raise RuntimeError(f"voiceover script is {total} chars (max {MAX_TOTAL_CHARS})")
    if engine == "edge" and p.get("ref"):
        raise RuntimeError("edge voices cannot clone a reference; use engine omnivoice or chatterbox")
    return engine, clean


def studio_paths(engine):
    root = config.TTS_STUDIO_DIR
    venv = ".venv" if engine == "omnivoice" else ".venv-cbx"
    return {"python": os.path.join(root, venv, "Scripts", "python.exe"),
            "hf_home": os.path.join(root, ".hf_cache"),
            "bin": os.path.join(root, "bin"),
            "script": os.path.join(VO_DIR, "synth_studio.py")}


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    params = job.get("params") or {}
    p = params.get("voiceover") or {}
    engine, lines = validate(p)

    run_kw = {"on_line": lambda l: log(f"  {l}"), "cancel_check": cancel_check,
              "timeout_seconds": timeout_seconds}

    spec = {"engine": engine, "lines": lines,
            "gap_ms": int(p.get("gap_ms", 250)), "lead_ms": int(p.get("lead_ms", 150))}
    for key in ("voice", "rate", "instruct", "language", "ref_text", "speed",
                "class_temperature", "exaggeration", "cfg_weight"):
        if p.get(key) is not None:
            spec[key] = p[key]

    if p.get("ref"):
        ref = p["ref"]
        if not (ref.get("bucket") and ref.get("path")):
            raise RuntimeError("voiceover ref must be {bucket, path}")
        db.set_phase(jid, "downloading reference", 2)
        spec["ref_path"] = gate_common.download(ref["bucket"], ref["path"], work_dir, "ref", log)

    if engine != "edge":
        spec["studio"] = studio_paths(engine)
        if not os.path.exists(spec["studio"]["python"]):
            raise RuntimeError(f"{engine}: the TTS studio venv is missing at {spec['studio']['python']}")

    db.set_phase(jid, "building venv", 4)
    py = venv_python(os.path.join(VO_DIR, "requirements.txt"), log, run_kw)

    spec_path = os.path.join(work_dir, "vo-spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    out_dir = os.path.join(work_dir, "vo")

    def on_line(line):
        if line.startswith("PHASE "):
            db.set_phase(jid, ("voice: " + line[6:].strip())[:60])
            return
        if line.startswith("PROGRESS "):
            try:
                heartbeat.progress = 5 + int(min(100, int(line.split()[1])) * 0.9)
            except ValueError:
                pass
            return
        log(f"  {line}")

    cmd = [py, "-u", os.path.join(VO_DIR, "voiceover.py"), spec_path, out_dir]
    if engine == "edge":
        proc.run_streaming(cmd, cwd=VO_DIR, on_line=on_line,
                           cancel_check=cancel_check, timeout_seconds=timeout_seconds)
    else:
        from videogen import comfy_client, tts_guard
        try:
            comfy_client.free()  # best effort; does not start ComfyUI
        except Exception as exc:  # noqa: BLE001
            log(f"voiceover: comfyui free skipped ({str(exc)[:80]})")
        with tts_guard.paused(log):
            proc.run_streaming(cmd, cwd=VO_DIR, on_line=on_line,
                               cancel_check=cancel_check, timeout_seconds=timeout_seconds)

    out_wav = os.path.join(out_dir, "vo.wav")
    words_path = os.path.join(out_dir, "words.json")
    if not (os.path.exists(out_wav) and os.path.exists(words_path)):
        raise RuntimeError("voiceover finished but vo.wav / words.json are missing")
    with open(words_path, encoding="utf-8") as f:
        result = json.load(f)

    try:
        db.upload_file(f"outputs/{jid}-words.json", words_path, "application/json")
    except Exception as exc:  # noqa: BLE001 - the row copy below is the primary
        log(f"voiceover: words sidecar upload failed (continuing): {str(exc)[:160]}")
    try:
        db.update_job(jid, {"params": {**params, "voiceover_result": result}})
    except Exception as exc:  # noqa: BLE001
        log(f"voiceover: could not record timings on the row: {str(exc)[:160]}")

    log(f"voiceover done: {engine}, {result.get('duration_s')} s, {len(result.get('words') or [])} words")
    heartbeat.progress = 97
    return out_wav, "wav", "audio/wav"
