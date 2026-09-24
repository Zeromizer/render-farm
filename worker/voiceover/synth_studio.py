"""Synthesise voice-over lines with the TTS studio's models, in the studio's venv.

  <studio python> synth_studio.py <studio-spec.json> <out_dir>

Writes raw-00.wav, raw-01.wav ... at the model's native rate. Runs under
C:\\Coding\\Voice Output\\.venv (OmniVoice) or .venv-cbx (Chatterbox), because
those venvs already hold the right torch build and the weights are cached in
the studio's .hf_cache. Mirrors tts_worker.py / chatterbox_worker.py; the
studio itself stays off - this loads the model for one job and exits.

VOICE CONSISTENCY: lines are generated one at a time (so line boundaries are
exact), and an OmniVoice designed or auto voice can drift between separate
calls. So unless a reference clip was given, OmniVoice's line 1 becomes the
reference for every later line: the whole VO is one voice cloned from its own
first line.
"""
import json
import os
import sys


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


def main():
    spec_path, out_dir = sys.argv[1], sys.argv[2]
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    # Both must be set BEFORE the model packages import (same as the workers).
    if spec.get("hf_home"):
        os.environ["HF_HOME"] = spec["hf_home"]
    if spec.get("bin"):
        os.environ["PATH"] = spec["bin"] + os.pathsep + os.environ.get("PATH", "")

    lines = spec["lines"]
    engine = spec["engine"]
    ref = spec.get("ref_path") or None
    ref_text = spec.get("ref_text") or None
    if engine == "omnivoice":
        omnivoice(spec, lines, out_dir, ref, ref_text)
    elif engine == "chatterbox":
        chatterbox(spec, lines, out_dir, ref)
    else:
        raise SystemExit(f"unknown studio engine {engine!r}")


def omnivoice(spec, lines, out_dir, ref, ref_text):
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    emit("PHASE", "loading omnivoice")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    sr = 24000  # OmniVoice's output rate (hard-coded in tts_worker.py too)
    base = {}
    for col in ("class_temperature", "speed"):
        if spec.get(col) is not None:
            base[col] = float(spec[col])
    # Shawn prefers 0.9 over the model's flat 0.0 default (studio UI default).
    base.setdefault("class_temperature", 0.9)
    if spec.get("language"):
        base["language"] = spec["language"]

    for i, text in enumerate(lines):
        emit("PHASE", f"speaking line {i + 1}/{len(lines)}")
        kw = dict(base)
        if ref:
            kw["ref_audio"] = ref
            if ref_text:
                kw["ref_text"] = ref_text
        elif i == 0 and spec.get("instruct"):
            kw["instruct"] = spec["instruct"]
        audio = model.generate(text=text, **kw)
        out = os.path.join(out_dir, f"raw-{i:02d}.wav")
        sf.write(out, audio[0], sr)
        if i == 0 and not ref:
            ref, ref_text = out, text  # the rest of the VO clones line 1
        emit("PROGRESS", str(int((i + 1) / len(lines) * 100)))


def chatterbox(spec, lines, out_dir, ref):
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS

    emit("PHASE", "loading chatterbox")
    model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    base = {}
    for col in ("exaggeration", "cfg_weight"):
        if spec.get(col) is not None:
            base[col] = float(spec[col])
    for i, text in enumerate(lines):
        emit("PHASE", f"speaking line {i + 1}/{len(lines)}")
        kw = dict(base)
        if ref:
            kw["audio_prompt_path"] = ref
        wav = model.generate(text, **kw).detach().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        out = os.path.join(out_dir, f"raw-{i:02d}.wav")
        # No self-cloning here: Chatterbox's built-in voice comes from fixed
        # conditionals, so it is already the same voice on every call.
        ta.save(out, wav, model.sr)
        emit("PROGRESS", str(int((i + 1) / len(lines) * 100)))


if __name__ == "__main__":
    main()
