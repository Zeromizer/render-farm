"""audio_gen runner: an instrumental music bed from a text description, generated
by the headless ComfyUI at config.COMFYUI_DIR (the same server video_gen uses).

params (jsonb):
  audio_gen:
    org_id / user_id / track_id   uuids, for scoping and logging only. The platform
                         (render-platform lib/music-generate.ts) scopes on org_id and
                         files the result against track_id; nothing here writes to rp_*.
    model                yue2_inst (default; see audiogen/graphs.py for what that is)
    prompt               required: the STYLE description (genre, mood, tempo, instruments)
    tags                 required: what goes in the model's lyrics input. For yue2_inst that
                         is timed section tags, one per line: "[intro 0:00-0:03]" ...
                         (intro | verse | pre-chorus | chorus | bridge | outro only)
    duration_s           4-180, the length to deliver (default 30)
    seed                 int (default 0)
    cfg_scale            optional, passed through when the graph has the input

Output: 48 kHz stereo 16-bit WAV, trimmed to duration_s with a short fade-out, uploaded as
outputs/<jid>.wav like every other engine. The measured length is written back onto the row
as params.audio_gen_result so the platform does not have to claim a length it did not check.

WHY THIS ENGINE EXISTS: the platform's music library could only hold audio that already
existed somewhere (an upload, a post, a reference video). This is the one source that starts
from a sentence, and like video_gen and matte it lives here because nothing else has a GPU.

VRAM: ~9-12 GB peak for the int8 checkpoint, so the TTS workers are paused for the duration
exactly as for H3, and ComfyUI is told to drop the model afterwards. The ComfyUI queue is
serial: a bed waits behind an H3 clip that is already sampling, and vice versa.
"""
import json
import os
import subprocess
import time
from datetime import datetime

import httpx

import db
import proc
from audiogen import graphs
from videogen import comfy_client, estimate, tts_guard
from videogen.segments import _tool   # the ffmpeg resolver that survives the Startup-shortcut PATH

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_AUDIO_EXT = (".flac", ".wav", ".mp3", ".opus", ".ogg", ".m4a")
_FADE = 0.4
# Shorter than this share of the request and it is not the bed that was asked for.
_MIN_SHARE = 0.7


def _ff(cmd):
    cmd = [_tool(cmd[0])] + list(cmd[1:])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed ({r.returncode}): {r.stderr[-800:]}")
    return r


def _duration(path):
    r = _ff(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    return float(json.loads(r.stdout)["format"]["duration"])


def _fetch_audio(outputs, dest_stem):
    """Download the saved audio named in a /history outputs dict; returns the local path.

    comfy_client.fetch_output only knows video containers. Same rule as there: only
    type == "output" entries count, because loader nodes report their inputs as previews."""
    for node_out in outputs.values():
        for key in ("audio", "audios"):
            for f in node_out.get(key, []) or []:
                name = f.get("filename", "")
                if f.get("type", "output") != "output" or not name.lower().endswith(_AUDIO_EXT):
                    continue
                dest = dest_stem + os.path.splitext(name)[1].lower()
                params = {"filename": name, "subfolder": f.get("subfolder", ""), "type": "output"}
                with httpx.stream("GET", comfy_client._url("/view"), params=params,
                                  timeout=comfy_client._TIMEOUT) as r:
                    if r.status_code != 200:
                        raise comfy_client.ComfyError(f"/view failed HTTP {r.status_code} for {name}")
                    with open(dest, "wb") as out:
                        for chunk in r.iter_bytes(1 << 20):
                            out.write(chunk)
                return dest
    raise comfy_client.ComfyError(f"no audio in comfyui outputs: {json.dumps(outputs)[:800]}")


def finish(src, dest, duration_s, log):
    """Trim to duration_s, fade the tail, write 48 kHz stereo s16 WAV. Returns the length written.

    The model ends near its last section tag, not on it, and a bed that is 31.7 s under a
    30 s video is a bed somebody has to cut by hand. A take that ends EARLY keeps its own
    ending (it was composed) and only gets the fade if it was cut off by the cap."""
    have = _duration(src)
    if have < duration_s * _MIN_SHARE:
        raise RuntimeError(f"the model stopped at {have:.1f} s of the {duration_s:g} s asked for; "
                           f"try again (a retry uses a new seed)")
    length = min(have, float(duration_s))
    fade_at = max(0.0, length - _FADE)
    _ff(["ffmpeg", "-v", "error", "-y", "-i", src, "-t", f"{length:.3f}",
         "-af", f"afade=t=out:st={fade_at:.3f}:d={_FADE}",
         "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", dest])
    log(f"audio_gen: {have:.2f} s generated -> {length:.2f} s delivered")
    return length, have


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    params = job.get("params") or {}
    p = params.get("audio_gen") or {}
    deadline = time.monotonic() + timeout_seconds

    model = p.get("model") or "yue2_inst"
    style = (p.get("prompt") or "").strip()
    tags = (p.get("tags") or "").strip()
    if not style:
        raise RuntimeError("audio_gen needs params.audio_gen.prompt (the style description)")
    if not tags:
        raise RuntimeError("audio_gen needs params.audio_gen.tags (the section tags that stand in for lyrics)")
    try:
        duration_s = float(p.get("duration_s") or 30)
    except (TypeError, ValueError):
        raise RuntimeError(f"audio_gen duration_s must be a number, got {p.get('duration_s')!r}")
    if not 4 <= duration_s <= 180:
        raise RuntimeError(f"audio_gen duration_s must be 4-180, got {duration_s:g}")
    seed = int(p.get("seed") or 0)

    # Fail on a missing graph before waking ComfyUI and pausing TTS for nothing.
    graph, meta = graphs.build(model, style, tags, duration_s, seed, f"audio_gen/{jid}",
                               cfg_scale=p.get("cfg_scale"))
    log(f"audio_gen {meta}: {duration_s:g} s, seed {seed}, style {style[:80]!r}")

    db.set_phase(jid, "starting comfyui", 3)
    comfy_client.ensure_server(log)
    with tts_guard.paused(log):
        try:
            def on_status(phase, frac, eta):
                heartbeat.progress = int(5 + 85 * max(0.0, min(1.0, frac)))
                db.set_phase(jid, f"music: {phase} - ~{estimate.fmt_eta(eta)} left"[:120], heartbeat.progress)

            since = datetime.now().isoformat()
            db.set_phase(jid, "music: queued", 5)
            prompt_id = comfy_client.submit(graph)
            log(f"comfyui prompt {prompt_id} (audio_gen)")
            try:
                outputs = comfy_client.wait(prompt_id, on_status, cancel_check,
                                            max(30, int(deadline - time.monotonic())),
                                            hint=graphs.hint(model, duration_s), since_iso=since)
            except comfy_client._CanceledSignal:
                raise proc.Canceled()
            db.set_phase(jid, "music: fetching", 92)
            raw = _fetch_audio(outputs, os.path.join(work_dir, "audio_gen-raw"))
        finally:
            # Hand the card back whether we succeeded, failed or were canceled.
            comfy_client.free()

    db.set_phase(jid, "music: trimming", 95)
    out_local = os.path.join(work_dir, "audio_gen.wav")
    length, have = finish(raw, out_local, duration_s, log)
    # Best effort: the audio is the deliverable, the note about it is not.
    try:
        db.update_job(jid, {"params": {**params, "audio_gen_result": {
            "duration_s": round(length, 3), "generated_s": round(have, 3), "model": model}}})
    except Exception as exc:
        log(f"audio_gen: could not record the measured length (continuing): {str(exc)[:160]}")

    heartbeat.progress = 97
    return out_local, "wav", "audio/wav"
