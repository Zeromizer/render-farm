"""Lip sync for the platform's lip_sync_clip tool: re-animate the mouth of a talking clip to a
given voice track (the ElevenLabs voice-over), keeping everything else in the frame.

Model: LatentSync 1.5 (ByteDance, code Apache-2.0, weights OpenRAIL++), the authors' own
scripts.inference at 7b380d6 (the last commit before 1.6; 1.6 needs 18 GB, the 4080 SUPER has 16)
in a Linux GPU container built from tools/lipsync/Dockerfile, i.e. their setup_env.sh. Run with
their 1.5 inference.sh settings: configs/unet/stage2.yaml, 20 steps, guidance 2.0.

Spike 2026-11-19 (aimotion-spin lipsync-spike, farm job a5fda126): their demo1 162 s, an 8 s
MaczTech presenter shot 91 s, a 3 s close-up 73 s (about 40 s of each is model load), VRAM peak
12.2 GB. Image build 26 min, once; weights 6 min, once (docker volume latentsync-ckpt).

usage: python lipsync.py --video V --audio A --duration D [--clip-in S] [--voice-in T]
                         [--steps 20] [--guidance 2.0] --out farm_out/lipsync.mp4
  V, A        the clip and the voice-over, materialised by the farm from the job's assets manifest
  S           where in the clip the used span starts (s, default 0)
  T           where in the voice-over that span's words start (s; negative = silence first)
  D           the span's length (s)
Output: the span, mouth re-synced, at the clip's own fps and size, with the voice slice as its
audio track (for review; the edit plays the voice-over from its own track). Prints
"LIPSYNC_ERROR <reason>" and exits 2 on a clip LatentSync cannot sync (no face).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

IMAGE = "aimotion/latentsync:7b380d6"
VOLUME = "latentsync-ckpt"
LABEL = "aimotion-lipsync=1"
HERE = os.path.dirname(os.path.abspath(__file__))
# The farm worker is a windowless pythonw: a console child (docker.exe) can die with 0xC000013A
# (the first spike build did). No window, no stdin.
NOWIN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0), "stdin": subprocess.DEVNULL}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def say(*a):
    print(*a, flush=True)


def fail(reason, code=2):
    say(f"LIPSYNC_ERROR {reason}")
    sys.exit(code)


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    d = os.environ.get("FFMPEG_DIR")
    if d and os.path.exists(os.path.join(d, name + ".exe")):
        return os.path.join(d, name + ".exe")
    hits = glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", name + ".exe"),
                     recursive=True)
    if not hits:
        fail(f"{name} not found on the render PC", 3)
    return hits[0]


def run(cmd, what):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **NOWIN)
    if p.returncode:
        fail(f"{what} failed: {(p.stderr or p.stdout)[-600:]}", 3)
    return p.stdout


def stream(cmd, timeout, name=None):
    """Echo the child's output; heartbeat every 10 s (the farm only checks cancel/timeout on a line)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1, **NOWIN)
    last = [time.time()]
    tail = []

    def beat():
        while p.poll() is None:
            if time.time() - last[0] > 10:
                say(f"heartbeat {int(time.time() - t0)} s")
                last[0] = time.time()
            time.sleep(2)
    threading.Thread(target=beat, daemon=True).start()
    buf = ""
    for ch in iter(lambda: p.stdout.read(1), ""):
        if ch in "\r\n":
            if buf.strip():
                say(buf.strip()[-300:])
                tail.append(buf.strip())
                del tail[:-40]
            buf = ""
            last[0] = time.time()
        else:
            buf += ch
        if time.time() - t0 > timeout:
            p.kill()
            if name:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, **NOWIN)
            fail(f"timed out after {timeout} s", 3)
    p.wait()
    return p.returncode, tail


def probe(ffprobe, path):
    out = json.loads(run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path], "ffprobe"))
    v = next((s for s in out["streams"] if s.get("codec_type") == "video"), None)
    if not v:
        fail("the clip has no video stream")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": fps, "dur": float(out["format"].get("duration") or 0)}


def ensure_image():
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, **NOWIN).returncode == 0:
        return
    say(f"building {IMAGE} (once, ~25 min)")
    rc, tail = stream(["docker", "build", "--progress", "plain", "-t", IMAGE, "-f", os.path.join(HERE, "Dockerfile"), HERE], 3600)
    if rc:
        fail(f"docker build failed rc={rc}: {' | '.join(tail[-3:])}", 3)


def ensure_weights():
    script = ("cd /opt/LatentSync && "
              "if [ ! -f checkpoints/latentsync_unet.pt ] || [ ! -f checkpoints/whisper/tiny.pt ]; then "
              "huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints && "
              "huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints; fi")
    rc, tail = stream(["docker", "run", "--rm", "-v", f"{VOLUME}:/opt/LatentSync/checkpoints", IMAGE, "bash", "-lc", script], 3600)
    if rc:
        fail(f"weights download failed rc={rc}: {' | '.join(tail[-3:])}", 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--clip-in", type=float, default=0.0)
    ap.add_argument("--voice-in", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not 0.5 <= a.duration <= 60:
        fail("duration must be 0.5-60 s; split a longer talking span into shots")

    ff, fp = find_tool("ffmpeg"), find_tool("ffprobe")
    work = os.path.abspath(os.path.join(os.environ.get("RENDER_WORK_DIR") or os.getcwd(), "lipsync_work"))
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    src = probe(fp, a.video)
    if a.clip_in < 0 or a.clip_in + a.duration > src["dur"] + 0.05:
        fail(f"the span {a.clip_in:.2f}-{a.clip_in + a.duration:.2f} s is outside the clip (it runs {src['dur']:.2f} s)")
    say(f"clip {src['w']}x{src['h']} @ {src['fps']:.3f} fps, {src['dur']:.2f} s; span {a.clip_in:.2f}+{a.duration:.2f} s; "
        f"voice from {a.voice_in:.2f} s")
    print("PROGRESS 3", flush=True)

    # The span, re-encoded on exact frames (input seek + re-encode is frame accurate).
    run([ff, "-v", "error", "-y", "-ss", f"{a.clip_in:.3f}", "-i", a.video, "-t", f"{a.duration:.3f}", "-an",
         "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", os.path.join(work, "video.mp4")], "trim clip")
    # The voice for exactly that span: silence before the voice starts (voice_in < 0) and after it ends.
    lead = max(0.0, -a.voice_in)
    af = [f"adelay={int(round(lead * 1000))}:all=1"] if lead > 0 else []
    af.append(f"apad=whole_dur={a.duration:.3f}")
    run([ff, "-v", "error", "-y", "-ss", f"{max(0.0, a.voice_in):.3f}", "-i", a.audio, "-t", f"{max(0.05, a.duration - lead):.3f}",
         "-af", ",".join(af), "-t", f"{a.duration:.3f}", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         os.path.join(work, "audio.wav")], "cut voice")
    print("PROGRESS 8", flush=True)

    ensure_image()
    print("PROGRESS 15", flush=True)
    ensure_weights()
    print("PROGRESS 20", flush=True)

    # A cancelled job leaves its container running: clear ours first (python jobs hold the GPU one at a time).
    stale = run(["docker", "ps", "-q", "--filter", f"label={LABEL}"], "docker ps").split()
    if stale:
        subprocess.run(["docker", "rm", "-f", *stale], capture_output=True, **NOWIN)
    name = f"lipsync-{os.getpid()}"
    t0 = time.time()
    rc, tail = stream(["docker", "run", "--rm", "--name", name, "--label", LABEL, "--gpus", "all", "--shm-size", "8g",
                       "-v", f"{VOLUME}:/opt/LatentSync/checkpoints", "--mount", f"type=bind,src={work},dst=/w",
                       IMAGE, "python", "-m", "scripts.inference",
                       "--unet_config_path", "configs/unet/stage2.yaml",
                       "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
                       "--inference_steps", str(a.steps), "--guidance_scale", str(a.guidance),
                       "--video_path", "/w/video.mp4", "--audio_path", "/w/audio.wav", "--video_out_path", "/w/synced.mp4"],
                      timeout=int(600 + 60 * a.duration), name=name)
    synced = os.path.join(work, "synced.mp4")
    if rc or not os.path.exists(synced):
        text = " ".join(tail).lower()
        if "face not detected" in text or "no face" in text:
            fail("no face found in part of the span: lip sync needs the speaker's face visible in every frame "
                 "(cut the span at the frames where it leaves, or keep the original clip there)")
        fail(f"LatentSync failed rc={rc}: {' | '.join(tail[-4:])[-600:]}", 3)
    say(f"LatentSync {time.time() - t0:.0f} s")
    print("PROGRESS 90", flush=True)

    # LatentSync works at 25 fps: back to the clip's own fps, size and exact length, with the voice slice.
    fps = round(src["fps"], 3) if src["fps"] > 1 else 30
    run([ff, "-v", "error", "-y", "-i", synced, "-i", os.path.join(work, "audio.wav"), "-map", "0:v", "-map", "1:a",
         "-vf", f"fps={fps},scale={src['w']}:{src['h']}:flags=lanczos", "-t", f"{a.duration:.3f}",
         "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "192k", out], "finish")
    res = probe(fp, out)
    say(json.dumps({"ok": True, "out": out, "fps": res["fps"], "duration": res["dur"], "size": [res["w"], res["h"]],
                    "seconds": round(time.time() - t0, 1)}))
    print("PROGRESS 100", flush=True)


if __name__ == "__main__":
    main()
