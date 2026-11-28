"""Music-from-reference spike, analysis side (farm python job, venv = requirements.txt here).

params.args = [mode, <job json, repo-relative>, <out dir>]
  prep     per reference: audio out of the video, htdemucs split (music vs voice), and the
           measures a new bed must match: tempo + beat grid, key, the main lift, energy curve.
           Also a muted 640 px preview + an mp3 of the mix for the rating page.
  measure  every take under ~/music-from-ref/takes/<ref>/ (plus the job's control beds) against
           its reference: tempo error, beat phase, key, lift timing, harmony distance (chroma
           DTW), melody overlap (SheetSage2 melody transcripts via the PC's ComfyUI), voices.

Job JSON: {"refs": [{"name": "taycan", "video": "spikes/music_from_ref/in/taycan.mp4",
                     "bpm_hint": 103.4, "control": "spikes/music_from_ref/in/control_taycan.wav"}]}
State that later jobs read lives on the PC under ~/music-from-ref/<ref>/ (ref.json, stems).
Prints a line every few seconds (farm cancel/timeout only act on output).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

import numpy as np
import soundfile as sf

HOME = os.path.join(os.path.expanduser("~"), "music-from-ref")
COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
ENCODER = "sheetsage2_bf16.safetensors"
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
T0 = time.monotonic()
_phase = {"name": "start"}


def log(msg):
    print(msg, flush=True)


def heartbeat():
    while True:
        time.sleep(10)
        log(f"alive t={int(time.monotonic() - T0)}s phase={_phase['name']}")


def ffmpeg():
    exe = os.path.join(os.environ.get("FFMPEG_DIR", ""), "ffmpeg.exe")
    if os.path.isfile(exe):
        return exe
    found = shutil.which("ffmpeg")
    if found:
        return found
    pat = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages", "*", "**", "ffmpeg.exe")
    hits = glob.glob(pat, recursive=True)
    if hits:
        return hits[0]
    raise RuntimeError("ffmpeg not found")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {r.stderr[-600:]}")
    return r


# ---------------------------------------------------------------- separation

_demucs = {}


def separate(path, dest_dir):
    """htdemucs two-stem split -> dest_dir/{vocals,no_vocals}.wav (44.1 kHz stereo)."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if "m" not in _demucs:
        m = get_model("htdemucs")
        m.eval()
        _demucs["m"] = m.cuda() if torch.cuda.is_available() else m
    m = _demucs["m"]
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != m.samplerate:
        import librosa
        y = librosa.resample(y.T, orig_sr=sr, target_sr=m.samplerate).T
        sr = m.samplerate
    if y.shape[1] == 1:
        y = np.repeat(y, 2, 1)
    wav = torch.from_numpy(y.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(m, ((wav - mean) / std)[None].to(next(m.parameters()).device), split=True, overlap=0.25,
                          progress=False)[0].cpu() * std + mean
    names = m.sources
    vocals = src[names.index("vocals")].numpy().T
    rest = sum(src[k] for k, n in enumerate(names) if n != "vocals").numpy().T
    os.makedirs(dest_dir, exist_ok=True)
    sf.write(os.path.join(dest_dir, "vocals.wav"), vocals, sr, subtype="PCM_16")
    sf.write(os.path.join(dest_dir, "no_vocals.wav"), rest, sr, subtype="PCM_16")
    return vocals, rest, sr


def voice_stats(vocals, rest, sr):
    v, a = vocals.mean(1), rest.mean(1)
    hop = sr // 10
    n = min(len(v), len(a)) // hop
    rv = np.sqrt(np.mean(v[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    ra = np.sqrt(np.mean(a[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    rel = 20 * np.log10(rv / ra)
    return {"voiced_s": round(float((rel > -12).sum()) / 10, 1),
            "vocal_to_band_db": round(float(20 * np.log10(np.sqrt(np.mean(v ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-12))), 1)}


# ---------------------------------------------------------------- measures

def load_mono(path, seconds=None, sr=22050):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True, duration=seconds)
    return y, sr


# Krumhansl-Kessler key profiles
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma(y, sr):
    import librosa
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)


def key_of(ch):
    prof = ch.mean(1)
    best = None
    for tonic in range(12):
        for mode, p in (("major", MAJ), ("minor", MIN)):
            r = float(np.corrcoef(prof, np.roll(p, tonic))[0, 1])
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    return {"key": f"{NOTES[tonic]} {mode}", "tonic": tonic, "mode": mode, "r": round(r, 3)}


def key_relation(a, b):
    """same / relative / fifth / parallel / other, from {'tonic','mode'} dicts."""
    if a["tonic"] == b["tonic"] and a["mode"] == b["mode"]:
        return "same"
    rel = lambda k: (k["tonic"] + (3 if k["mode"] == "minor" else -3)) % 12  # noqa: E731
    if a["mode"] != b["mode"] and rel(a) == b["tonic"]:
        return "relative"
    if a["tonic"] == b["tonic"]:
        return "parallel"
    if a["mode"] == b["mode"] and (a["tonic"] - b["tonic"]) % 12 in (5, 7):
        return "fifth"
    return "other"


def tempo_beats(y, sr, hint=None):
    import librosa
    on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo, frames = librosa.beat.beat_track(onset_envelope=on, sr=sr, hop_length=512, start_bpm=hint or 120.0)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    gaps = np.diff(beats)
    bpm = float(60 / np.median(gaps)) if len(gaps) > 3 else float(np.atleast_1d(tempo)[0])
    mids = ((frames[:-1] + frames[1:]) // 2) if len(frames) > 1 else frames
    contrast = float(np.mean(on[frames]) / max(1e-6, np.mean(on[mids]))) if len(frames) > 3 else None
    return bpm, beats, contrast


def energy_db(y, sr, hop_s=0.1):
    hop = int(sr * hop_s)
    n = len(y) // hop
    r = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    return 20 * np.log10(r)


def lift_of(db, hop_s=0.1, win_s=1.5, dur=None):
    """Steepest sustained rise: mean dB over the next win_s minus the previous win_s."""
    w = int(win_s / hop_s)
    n = len(db)
    lo, hi = int(1.0 / hop_s), n - int(2.5 / hop_s)
    best = (None, -99.0)
    for i in range(max(w, lo), min(hi, n - w)):
        rise = float(np.mean(db[i:i + w]) - np.mean(db[i - w:i]))
        if rise > best[1]:
            best = (i, rise)
    if best[0] is None:
        return None, None
    return round(best[0] * hop_s, 2), round(best[1], 1)


def octave_fold(r):
    return r * 2.0 ** (-round(np.log2(r))) if r > 0 else r


def beat_hit(ref_beats, take_beats, tol=0.07, offset=0.0):
    if len(ref_beats) == 0 or len(take_beats) == 0:
        return None
    tb = np.asarray(take_beats) + offset
    d = np.min(np.abs(np.asarray(ref_beats)[:, None] - tb[None, :]), 1)
    return round(float(np.mean(d <= tol)), 3)


def dtw_cost(ca, cb):
    import librosa
    D, wp = librosa.sequence.dtw(X=ca, Y=cb, metric="cosine")
    return round(float(D[-1, -1] / len(wp)), 4)


# ---------------------------------------------------------------- SheetSage melody (ComfyUI)

def comfy_up():
    try:
        urllib.request.urlopen(COMFY + "/system_stats", timeout=5).read()
        return True
    except Exception:
        return False


def ensure_comfy(wait_s=240):
    if comfy_up():
        return
    bat = os.path.join(COMFY_DIR, "run-headless.bat")
    log(f"comfyui not running; launching {bat}")
    subprocess.Popen(["cmd", "/c", bat], cwd=COMFY_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if comfy_up():
            return
        time.sleep(3)
    raise RuntimeError("ComfyUI did not come up")


def comfy_upload(path, subfolder="music_from_ref"):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    parts = []
    for k, v in (("subfolder", subfolder), ("type", "input"), ("overwrite", "true")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                 f'Content-Type: application/octet-stream\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(COMFY + "/upload/image", data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    j = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def comfy_prompt(g, timeout_s=900):
    req = urllib.request.Request(COMFY + "/prompt", data=json.dumps({"prompt": g, "client_id": "music-from-ref"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"/prompt {e.code}: {e.read().decode('utf-8', 'replace')[:1500]}") from None
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        e = json.loads(urllib.request.urlopen(COMFY + f"/history/{pid}", timeout=60).read()).get(pid)
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                err = next((m[1] for m in st.get("messages") or [] if m[0] == "execution_error"), {})
                raise RuntimeError(f"{err.get('node_type')}: {err.get('exception_message')}")
            return e.get("outputs") or {}
        time.sleep(1.5)
    raise RuntimeError("ComfyUI prompt timed out")


def sheetsage(path, mode="melody"):
    """SheetSage2 ABC transcript of an audio file (ComfyUI native node)."""
    name = comfy_upload(path)
    g = {"1": {"class_type": "LoadAudio", "inputs": {"audio": name}},
         "2": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": ENCODER}},
         "3": {"class_type": "SheetSage2AudioToABC", "inputs": {"audio_encoder": ["2", 0], "audio": ["1", 0], "mode": mode}},
         "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}}
    out = comfy_prompt(g)
    texts = []
    for node_out in out.values():
        for v in node_out.get("text", []) or []:
            texts.append(v if isinstance(v, str) else json.dumps(v))
    return "\n".join(texts)


def melody_notes(abc):
    """Pitch sequence (MIDI numbers) of every note in an ABC text, all voices, in order
    (SheetSage melody mode: the Vocal voice carries the lead line; Ins the second line)."""
    import re
    key_acc = {}
    seq = {"Vocal": [], "Ins": []}
    voice = "Vocal"
    base = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
    for line in abc.splitlines():
        line = line.strip()
        if line.startswith("V:"):
            voice = "Ins" if "Ins" in line else "Vocal"
            continue
        if not line or line.startswith("%") or re.match(r"^[A-Za-z]:", line):
            continue
        line = re.sub(r'"[^"]*"', "", line)
        for m in re.finditer(r"([\^_=]*)([A-Ga-g])([',]*)", line):
            acc, letter, octv = m.groups()
            p = base[letter.upper()] + (12 if letter.islower() else 0)
            p += 12 * octv.count("'") - 12 * octv.count(",")
            p += acc.count("^") - acc.count("_")
            seq.setdefault(voice, []).append(p)
    return seq


def melody_overlap(a, b, n=5):
    """Transposition-free comparison of two pitch sequences: Jaccard of interval n-grams and the
    longest shared run of intervals."""
    ia = [y - x for x, y in zip(a, a[1:])]
    ib = [y - x for x, y in zip(b, b[1:])]
    if len(ia) < n or len(ib) < n:
        return {"ngram_jaccard": None, "longest_run": 0, "notes": [len(a), len(b)]}
    ga = {tuple(ia[k:k + n]) for k in range(len(ia) - n + 1)}
    gb = {tuple(ib[k:k + n]) for k in range(len(ib) - n + 1)}
    # longest common substring of intervals
    best, prev = 0, [0] * (len(ib) + 1)
    for x in ia:
        cur = [0] * (len(ib) + 1)
        for j, y in enumerate(ib):
            if x == y:
                cur[j + 1] = prev[j] + 1
                best = max(best, cur[j + 1])
        prev = cur
    # trivial runs (repeated notes = interval 0) say nothing; count runs with motion only
    return {"ngram_jaccard": round(len(ga & gb) / len(ga | gb), 3), "longest_run": best, "notes": [len(a), len(b)]}


# ---------------------------------------------------------------- modes

def read_job(path):
    p = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def prep(job, out):
    ff = ffmpeg()
    log(f"ffmpeg: {ff}")
    for k, ref in enumerate(job["refs"]):
        name = ref["name"]
        _phase["name"] = f"prep:{name}"
        log(f"PROGRESS {5 + int(80 * k / len(job['refs']))}")
        d = os.path.join(HOME, name)
        os.makedirs(d, exist_ok=True)
        video = ref["video"] if os.path.isabs(ref["video"]) else os.path.join(os.getcwd(), ref["video"])
        mix = os.path.join(d, "mix.wav")
        run([ff, "-y", "-v", "error", "-i", video, "-vn", "-ac", "2", "-ar", "44100", mix])
        o = os.path.join(out, name)
        os.makedirs(o, exist_ok=True)
        run([ff, "-y", "-v", "error", "-i", video, "-an", "-vf", "scale=-2:640", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(o, "preview.mp4")])
        run([ff, "-y", "-v", "error", "-i", mix, "-b:a", "128k", os.path.join(o, "reference.mp3")])
        log(f"{name}: separating")
        vocals, rest, sr = separate(mix, d)
        run([ff, "-y", "-v", "error", "-i", os.path.join(d, "no_vocals.wav"), "-b:a", "128k", os.path.join(o, "reference_music.mp3")])
        dur = len(rest) / sr
        y, ysr = load_mono(os.path.join(d, "no_vocals.wav"))
        bpm, beats, contrast = tempo_beats(y, ysr, ref.get("bpm_hint"))
        ch = chroma(y, ysr)
        db = energy_db(y, ysr)
        lift, rise = lift_of(db)
        info = {"name": name, "duration_s": round(dur, 3), "bpm": round(bpm, 2), "bpm_hint": ref.get("bpm_hint"),
                "beats": [round(float(b), 3) for b in beats], "first_beat": round(float(beats[0]), 3) if len(beats) else None,
                "beat_contrast": round(contrast, 2) if contrast else None, "key": key_of(ch), "lift_s": lift, "lift_rise_db": rise,
                "energy_db_0p3": [round(float(np.mean(db[i:i + 3])), 1) for i in range(0, len(db), 3)][:400],
                "voice": voice_stats(vocals, rest, sr), "median_db": round(float(np.median(db)), 1)}
        with open(os.path.join(d, "ref.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        shutil.copyfile(os.path.join(d, "ref.json"), os.path.join(o, "ref.json"))
        log(f"{name}: {json.dumps({k: info[k] for k in ('duration_s', 'bpm', 'bpm_hint', 'first_beat', 'key', 'lift_s', 'lift_rise_db', 'voice')})}")


def measure(job, out):
    ff = ffmpeg()
    ensure_comfy()
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    results = {"refs": {}, "takes": []}
    melodies = {}
    takes = []
    for ref in job["refs"]:
        name = ref["name"]
        tdir = os.path.join(HOME, "takes", name)
        keep = job.get("measure_arms")
        for f in sorted(glob.glob(os.path.join(tdir, "*.wav"))):
            tid = os.path.splitext(os.path.basename(f))[0]
            if keep and tid.split("_")[0] not in keep:
                continue
            takes.append((name, tid, f))
        if ref.get("control"):
            c = ref["control"] if os.path.isabs(ref["control"]) else os.path.join(os.getcwd(), ref["control"])
            takes.append((name, "C_job", c))
    refinfo = {}
    for ref in job["refs"]:
        name = ref["name"]
        d = os.path.join(HOME, name)
        with open(os.path.join(d, "ref.json"), encoding="utf-8") as f:
            info = json.load(f)
        y, sr = load_mono(os.path.join(d, "no_vocals.wav"))
        info["_chroma"] = chroma(y, sr)
        _phase["name"] = f"sheetsage:{name}"
        try:
            melodies[name] = melody_notes(sheetsage(os.path.join(d, "no_vocals.wav")))
        except Exception as e:
            log(f"sheetsage ref {name} failed: {e}")
            melodies[name] = None
        refinfo[name] = info
        results["refs"][name] = {k: v for k, v in info.items() if not k.startswith("_") and k not in ("beats", "energy_db_0p3")}
    sepdir = os.path.join(HOME, "_sep")
    for k, (name, tid, path) in enumerate(takes):
        _phase["name"] = f"measure:{name}/{tid}"
        log(f"PROGRESS {5 + int(90 * k / max(1, len(takes)))}")
        ri = refinfo[name]
        dur = ri["duration_s"]
        row = {"ref": name, "take": tid, "arm": tid.split("_")[0]}
        try:
            seg = os.path.join(sepdir, f"{name}__{tid}.wav")
            os.makedirs(sepdir, exist_ok=True)
            run([ff, "-y", "-v", "error", "-i", path, "-t", f"{dur:.3f}", "-ac", "2", "-ar", "44100", seg])
            vocals, rest, sr = separate(seg, os.path.join(sepdir, f"{name}__{tid}"))
            row["voice"] = voice_stats(vocals, rest, sr)
            y, ysr = load_mono(os.path.join(sepdir, f"{name}__{tid}", "no_vocals.wav"))
            row["length_s"] = round(sf.info(path).duration, 2)
            bpm, beats, contrast = tempo_beats(y, ysr, ri["bpm"])
            ratio = octave_fold(bpm / ri["bpm"])
            row.update({"bpm": round(bpm, 1), "tempo_err_pct": round(abs(ratio - 1) * 100, 1),
                        "beat_contrast": round(contrast, 2) if contrast else None})
            row["beat_hit"] = beat_hit(ri["beats"], [b for b in beats if b <= dur])
            offs = np.arange(-0.5, 0.501, 0.01)
            hits = [beat_hit(ri["beats"], [b for b in beats if b <= dur], offset=o) or 0 for o in offs]
            row["beat_hit_best"] = round(max(hits), 3)
            row["beat_best_offset_s"] = round(float(offs[int(np.argmax(hits))]), 2)
            if job.get("align"):
                # delaying the take by o == trimming (period - o) mod period off its head
                period = 60.0 / bpm
                trim = (-row["beat_best_offset_s"]) % period
                if trim > 0.02:
                    seg2 = os.path.join(sepdir, f"{name}__{tid}.aligned.wav")
                    run([ff, "-y", "-v", "error", "-ss", f"{trim:.3f}", "-i", path, "-t", f"{dur:.3f}", "-ac", "2", "-ar", "44100", seg2])
                    path = seg2
                    vocals, rest, sr = separate(seg2, os.path.join(sepdir, f"{name}__{tid}.al"))
                    y, ysr = load_mono(os.path.join(sepdir, f"{name}__{tid}.al", "no_vocals.wav"))
                    bpm, beats, contrast = tempo_beats(y, ysr, ri["bpm"])
                    row["beat_hit"] = beat_hit(ri["beats"], [b for b in beats if b <= dur])
                row["aligned_trim_s"] = round(trim, 3) if trim > 0.02 else 0.0
            ch = chroma(y, ysr)
            kk = key_of(ch)
            row["key"] = kk["key"]
            row["key_rel"] = key_relation(kk, ri["key"])
            row["chroma_profile_r"] = round(float(np.corrcoef(ch.mean(1), ri["_chroma"].mean(1))[0, 1]), 3)
            row["chroma_dtw"] = dtw_cost(ri["_chroma"], ch)
            db = energy_db(y, ysr)
            lift, rise = lift_of(db)
            row["lift_s"], row["lift_rise_db"] = lift, rise
            row["lift_err_s"] = round(abs(lift - ri["lift_s"]), 2) if lift is not None and ri["lift_s"] is not None else None
            row["first_s_vs_median_db"] = round(float(np.mean(db[:10]) - np.median(db)), 1)
            secs = [np.mean(db[i * 10:(i + 1) * 10]) for i in range(4, int(len(db) / 10) - 1)]
            row["energy_sd_db"] = round(float(np.std(secs)), 1) if secs else None
            if melodies.get(name):
                try:
                    mt = melody_notes(sheetsage(os.path.join(sepdir, f"{name}__{tid}", "no_vocals.wav")))
                    mo = {}
                    for va in ("Vocal", "Ins"):
                        for vb in ("Vocal", "Ins"):
                            mo[f"{va}-{vb}"] = melody_overlap(melodies[name][va], mt[vb])
                    row["melody"] = mo
                    row["melody_max_jaccard"] = max((m["ngram_jaccard"] or 0) for m in mo.values())
                    row["melody_max_run"] = max(m["longest_run"] for m in mo.values())
                except Exception as e:
                    row["melody_error"] = str(e)[:300]
            mp3 = os.path.join(out, "audio", f"{name}__{tid}.mp3")
            fade = min(2.0, max(0.8, 0.05 * dur))
            run([ff, "-y", "-v", "error", "-i", path, "-t", f"{dur:.3f}", "-af", f"afade=t=out:st={max(0, dur - fade):.3f}:d={fade:.2f}",
                 "-b:a", "160k", mp3])
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:400]
        results["takes"].append(row)
        log(f"{name}/{tid}: " + json.dumps({k: row.get(k) for k in ("tempo_err_pct", "beat_hit", "beat_hit_best", "key_rel", "chroma_dtw",
                                                                       "lift_err_s", "melody_max_jaccard", "melody_max_run", "voice", "error")}))
        with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    mode, job_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    job = read_job(job_path)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    threading.Thread(target=heartbeat, daemon=True).start()
    {"prep": prep, "measure": measure}[mode](job, out)
    log("PROGRESS 100")


if __name__ == "__main__":
    main()
