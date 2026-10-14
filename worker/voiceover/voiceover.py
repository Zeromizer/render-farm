"""Voice-over with word timings: one spec in, vo.wav + words.json out.

  python voiceover.py <spec.json> <out_dir>

Runs in the voiceover venv (edge-tts + faster-whisper). The studio engines
(OmniVoice, Chatterbox) are NOT importable here - they live in the TTS studio's
own torch venvs with their weights already cached - so for those this script
drives synth_studio.py under the studio's python and only does the timing and
assembly itself.

spec:
  engine        edge | omnivoice | chatterbox
  voice         edge: a voice name (en-SG-WayneNeural ...). ignored otherwise
  rate          edge: percent, e.g. 8 for +8%
  lines         [{id, text, say}]  text = what the screen shows, say = what to speak
  gap_ms        silence between lines (default 250); a line may set pause_after_ms
  lead_ms       silence before the first line (default 150)
  language      e.g. "en", "zh" - whisper hint, and OmniVoice's language
  studio        {python, hf_home, bin, script}  for omnivoice/chatterbox
  ref_path      optional reference clip for cloning (studio engines)
  ref_text      optional transcript of the reference (OmniVoice)
  instruct      optional OmniVoice voice design ("female, warm, singaporean accent")
  speed, class_temperature, exaggeration, cfg_weight   engine knobs

Every line is synthesised SEPARATELY so its boundaries are exact, then laid end
to end; word times inside a line come from the synthesiser (edge) or from
faster-whisper on that line alone (studio engines), mapped onto the written
tokens by align.py.
"""
import asyncio
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "extract"))
import align  # noqa: E402
from common import find_ffmpeg_tool  # noqa: E402

TARGET_LUFS = -16.0
SR = 48000
HEAD_TRIM = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.04"
TAIL_TRIM = ("areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.08,"
             "areverse")


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


def ff(*args):
    r = subprocess.run([find_ffmpeg_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {r.stderr[-600:]}")


def duration(path):
    out = subprocess.run([find_ffmpeg_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


# ---------------------------------------------------------------- edge

async def _edge_line(text, voice, rate, mp3):
    import edge_tts

    last = None
    for attempt in range(4):
        comm = edge_tts.Communicate(text, voice, boundary="WordBoundary", rate=f"{int(rate):+d}%")
        words, audio = [], bytearray()
        try:
            async for ch in comm.stream():
                if ch["type"] == "WordBoundary":
                    s = ch["offset"] / 1e7
                    words.append((s, s + ch["duration"] / 1e7, ch["text"]))
                elif ch["type"] == "audio":
                    audio.extend(ch["data"])
        except Exception as exc:  # NoAudioReceived, websocket resets - transient
            last = exc
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        if words and audio:
            with open(mp3, "wb") as f:
                f.write(bytes(audio))
            return words
        last = RuntimeError("no audio / no WordBoundary events")
        await asyncio.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"edge-tts failed 4x on {text[:48]!r}: {last}")


def synth_edge(spec, lines, out_dir):
    voice = spec.get("voice") or "en-SG-WayneNeural"
    rate = spec.get("rate") or 0
    timed = {}
    for i, ln in enumerate(lines):
        emit("PHASE", f"speaking line {i + 1}/{len(lines)}")
        mp3 = os.path.join(out_dir, f"line-{i:02d}.mp3")
        words = asyncio.run(_edge_line(ln["say"], voice, rate, mp3))
        # Trim the TAIL only: edge leaves ~1 s of silence after the last word,
        # which would double every gap. The head stays, because WordBoundary
        # offsets are measured from the untrimmed start.
        ff("-i", mp3, "-af", TAIL_TRIM, "-ar", str(SR), "-ac", "1",
           os.path.join(out_dir, f"line-{i:02d}.wav"))
        timed[i] = words
        emit("PROGRESS", str(int((i + 1) / len(lines) * 70)))
    return timed


# ---------------------------------------------------------------- studio

def synth_studio(spec, lines, out_dir):
    st = spec.get("studio") or {}
    py = st.get("python")
    if not (py and os.path.exists(py)):
        raise SystemExit(f"{spec['engine']}: studio python not found at {py!r}")
    sspec = {k: spec.get(k) for k in ("engine", "ref_path", "ref_text", "instruct", "language",
                                      "speed", "class_temperature", "exaggeration", "cfg_weight")}
    sspec["lines"] = [ln["say"] for ln in lines]
    sspec["hf_home"] = st.get("hf_home")
    sspec["bin"] = st.get("bin")
    sp = os.path.join(out_dir, "studio-spec.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(sspec, f, ensure_ascii=False)
    emit("PHASE", f"loading {spec['engine']}")
    p = subprocess.Popen([py, "-u", st.get("script") or os.path.join(HERE, "synth_studio.py"), sp, out_dir],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            try:
                emit("PROGRESS", str(int(int(line.split()[1]) * 0.6)))
            except ValueError:
                pass
        elif line.startswith("PHASE "):
            emit("PHASE", line[6:])
        else:
            print(f"  [studio] {line}", flush=True)
            tail = (tail + [line])[-20:]
    if p.wait() != 0:
        raise SystemExit(f"{spec['engine']} synthesis failed: " + " | ".join(tail[-6:]))

    # Studio output: native rate, with whatever lead/tail silence the model
    # left. Trim both ends so the gap between lines is the gap we choose, then
    # resample. Timings are measured AFTER this, so the trim cannot shift them.
    trim = HEAD_TRIM + "," + TAIL_TRIM
    for i in range(len(lines)):
        raw = os.path.join(out_dir, f"raw-{i:02d}.wav")
        if not os.path.exists(raw):
            raise SystemExit(f"studio produced no audio for line {i + 1}")
        ff("-i", raw, "-af", trim, "-ar", str(SR), "-ac", "1", os.path.join(out_dir, f"line-{i:02d}.wav"))
    return None


def whisper_times(lines, out_dir, language):
    """Word times per line from faster-whisper, prompted with the line's own words."""
    from faster_whisper import WhisperModel

    emit("PHASE", "timing words")
    model = WhisperModel(os.environ.get("VO_WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    timed = {}
    for i, ln in enumerate(lines):
        segs, _ = model.transcribe(os.path.join(out_dir, f"line-{i:02d}.wav"), language=language or None,
                                   word_timestamps=True, initial_prompt=ln["say"], vad_filter=False,
                                   beam_size=5, condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append((float(w.start), float(w.end), w.word.strip()))
        timed[i] = words
        emit("PROGRESS", str(60 + int((i + 1) / len(lines) * 25)))
    return timed


# ---------------------------------------------------------------- assembly

def assemble(spec, lines, timed, out_dir):
    gap = int(spec.get("gap_ms", 250)) / 1000.0
    t = int(spec.get("lead_ms", 150)) / 1000.0
    words, meta = [], []
    for i, ln in enumerate(lines):
        wav = os.path.join(out_dir, f"line-{i:02d}.wav")
        d = duration(wav)
        ws = align.align(ln["text"], timed.get(i) or [], line_start=t, line_end=d)
        for w in ws:
            w["line"] = ln["id"]
        words += ws
        meta.append({"id": ln["id"], "text": ln["text"], "say": ln["say"], "start": round(t, 3),
                     "end": round(t + d, 3), "match_rate": round(align.match_rate(ws), 3)})
        after = ln.get("pause_after_ms")
        t += d + (int(after) / 1000.0 if after is not None else gap)

    total = meta[-1]["end"] + 0.35  # a short breath of room tone after the last word
    emit("PHASE", "mixing")
    inputs, chains = [], []
    for i, m in enumerate(meta):
        inputs += ["-i", os.path.join(out_dir, f"line-{i:02d}.wav")]
        ms = int(round(m["start"] * 1000))
        chains.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(meta)))
    fc = (";".join(chains) + f";{mix}amix=inputs={len(meta)}:normalize=0,apad,"
          f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11,aresample={SR}[out]")
    out_wav = os.path.join(out_dir, "vo.wav")
    ff(*inputs, "-filter_complex", fc, "-map", "[out]", "-t", f"{total:.3f}",
       "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", out_wav)
    return out_wav, {"duration_s": round(total, 3), "lines": meta, "words": words}


def main():
    spec_path, out_dir = sys.argv[1], sys.argv[2]
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for n, ln in enumerate(spec["lines"]):
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        lines.append({"id": ln.get("id") or f"l{n + 1}", "text": text,
                      "say": (ln.get("say") or text).strip(), "pause_after_ms": ln.get("pause_after_ms")})
    if not lines:
        raise SystemExit("no lines to speak")

    engine = spec.get("engine") or "edge"
    if engine == "edge":
        timed = synth_edge(spec, lines, out_dir)
    elif engine in ("omnivoice", "chatterbox"):
        synth_studio(spec, lines, out_dir)
        timed = whisper_times(lines, out_dir, spec.get("language"))
    else:
        raise SystemExit(f"unknown engine {engine!r}")

    out_wav, result = assemble(spec, lines, timed, out_dir)
    result.update(engine=engine, voice=spec.get("voice") if engine == "edge" else None)
    with open(os.path.join(out_dir, "words.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    for d in os.listdir(out_dir):
        if d.startswith(("raw-", "line-")) and d.endswith(".mp3"):
            os.remove(os.path.join(out_dir, d))
    emit("PROGRESS", "100")
    low = [m for m in result["lines"] if m["match_rate"] < 0.6]
    for m in low:
        print(f"[voiceover] WARNING line {m['id']} match_rate {m['match_rate']}: the voice may not have "
              f"said what was written", flush=True)
    print(f"[voiceover] WROTE {out_wav} {result['duration_s']}s, {len(result['words'])} words", flush=True)


if __name__ == "__main__":
    main()
