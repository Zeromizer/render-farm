"""Smooth 60 fps export, prep half (render-platform "60 fps" download button).

AI footage arrives at 24 fps and the platform renders at 30, so a fifth of the frames repeat. For a 60 fps
download this job clones the version's workspace at its render commit into C:\\rp-smooth\\<render id>
(the live workspace is only read by `git clone`), runs RIFE v4.26 on every footage clip the composition
references, keeps file names and audio, and commits on branch `smooth60`. The platform then queues a
normal hyperframes render of that branch with fps=60.

usage: python smooth60.py SRC_WS GIT_REF DST_WS OUT_DIR [--fps=60]

Model: the author's own RIFE v4.26 (hzwer/Practical-RIFE pinned + hzwer/RIFE HF weights), run in memory.
2026-11-06 A/B (render-farm spike/fps-ab): at native resolution it keeps the Denza D9 grille straight, where
rife-ncnn v4.6 bent the slats; ~3 s per 7 s clip on the 4080.
Cuts inside a clip get the nearest real frame, never a blend. Alpha clips (cutout mattes) are left alone.
"""
import glob, io, json, os, re, shutil, subprocess, sys, time, urllib.request, zipfile
import numpy as np
import cv2

SRC, REF, DST, OUT = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4])
TARGET = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--fps=")), "60"))
ROOT = r"C:\rp-smooth"
assert os.path.normcase(os.path.dirname(os.path.abspath(DST))) == os.path.normcase(ROOT), f"DST must be a child of {ROOT}"
CACHE = os.path.join(ROOT, "_rife")
PR_SHA = "bbfd2ea90910789a860ea3e2b32a240cd577b75e"
W426 = "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip"
KEEP_DAYS = 3
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
t_start = time.time()


def say(*a):
    print(*a, flush=True)


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    d = os.environ.get("FFMPEG_DIR")
    if d and os.path.exists(os.path.join(d, name + ".exe")):
        return os.path.join(d, name + ".exe")
    return glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", name + ".exe"), recursive=True)[0]


FF, FP = find_tool("ffmpeg"), find_tool("ffprobe")


def stream(cmd, tag):
    """Echo the child's output line by line: the farm only checks cancel/timeout when a line arrives."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        if line.strip():
            tail = (tail + [line.rstrip()])[-15:]
            say(f"[{tag}] {line.rstrip()[:200]}")
    if p.wait() != 0:
        raise RuntimeError(f"{tag} failed rc={p.returncode}: " + "\n".join(tail))


# 0. housekeeping: old exports (the render of each has long since finished)
for d in glob.glob(os.path.join(ROOT, "*")):
    if os.path.basename(d).startswith("_") or os.path.normcase(d) == os.path.normcase(DST):
        continue
    try:
        if time.time() - os.path.getmtime(d) > KEEP_DAYS * 86400:
            say("removing old export", d)
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", d])
    except OSError:
        pass

# 1. clone at the render commit
if os.path.exists(DST):
    stream(["cmd", "/c", "rmdir", "/s", "/q", DST], "rm")
stream(["git", "clone", "-q", "--no-checkout", SRC, DST], "clone")
stream(["git", "-C", DST, "checkout", "-q", "-B", "smooth60", REF], "checkout")
say("PROGRESS 4")

# 2. RIFE v4.26
pr = os.path.join(CACHE, "Practical-RIFE")
if not os.path.isdir(os.path.join(pr, ".git")):
    stream(["git", "clone", "-q", "https://github.com/hzwer/Practical-RIFE", pr], "rife-clone")
stream(["git", "-C", pr, "checkout", "-q", PR_SHA], "rife-checkout")
tl = os.path.join(pr, "train_log")
if not os.path.exists(os.path.join(tl, "flownet.pkl")):
    say("downloading", W426)
    data = urllib.request.urlopen(W426, timeout=300).read()
    os.makedirs(tl, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            base = os.path.basename(m)
            if base.endswith((".py", ".pkl")) and "__MACOSX" not in m:
                open(os.path.join(tl, base), "wb").write(z.read(m))
sys.path.insert(0, pr)
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from train_log.RIFE_HDv3 import Model
model = Model(); model.load_model(tl, -1); model.eval(); model.device()
say("rife", getattr(model, "version", "?"), "on", dev)
say("PROGRESS 8")


# 3. which clips does the composition use?
def probe(path):
    j = json.loads(subprocess.run([FP, "-v", "error", "-count_packets", "-show_streams", "-show_format", "-of", "json", path],
                                  capture_output=True, text=True).stdout or "{}")
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return None
    num, den = (v.get("r_frame_rate") or "0/1").split("/")
    tags = v.get("tags") or {}
    pf = v.get("pix_fmt") or ""
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": int(num) / max(1, int(den)), "pix_fmt": pf,
            "frames": int(v.get("nb_read_packets") or 0),
            "audio": any(s.get("codec_type") == "audio" for s in j.get("streams", [])),
            "alpha": any(x in pf for x in ("yuva", "rgba", "argb", "bgra", "abgr", "gbrap", "ya"))
            or str(tags.get("alpha_mode", tags.get("ALPHA_MODE", ""))) == "1"}


text = ""
for h in glob.glob(os.path.join(DST, "**", "*.*"), recursive=True):
    if h.lower().endswith((".html", ".js", ".json", ".css")) and os.sep + ".git" + os.sep not in h and "node_modules" not in h:
        text += open(h, encoding="utf-8", errors="replace").read()
refs = set(os.path.basename(m) for m in re.findall(r'''["'(/]([^"'()/\\]+\.(?:mp4|mov|webm|m4v))''', text, re.I))
report = {"ref": REF, "target": TARGET, "rife": "v4.26", "clips": {}, "skipped": {}}
todo = []
for p in glob.glob(os.path.join(DST, "**", "*.*"), recursive=True):
    if not p.lower().endswith((".mp4", ".mov", ".webm", ".m4v")) or os.sep + ".git" + os.sep in p or "node_modules" in p:
        continue
    rel = os.path.relpath(p, DST)
    info = probe(p)
    if not info:
        continue
    why = ("not referenced" if os.path.basename(p) not in refs else "alpha" if info["alpha"]
           else f"already {info['fps']:.0f} fps" if info["fps"] >= TARGET - 0.5 else "no frames" if info["frames"] < 2 else None)
    if why:
        report["skipped"][rel] = why
    else:
        todo.append((p, rel, info))
say("PROGRESS 10", json.dumps({"todo": [t[1] for t in todo], "skipped": report["skipped"]}))


# 4. interpolate each clip, streaming frames through ffmpeg pipes
def to_t(f, h, w, ph, pw):
    x = torch.from_numpy(f).to(dev, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float() / 255.
    return F.pad(x, (0, pw - w, 0, ph - h))


done_frames = 0
total_out = sum(int(round(i["frames"] * TARGET / i["fps"])) for _, _, i in todo) or 1
for p, rel, info in todo:
    c0 = time.time()
    w, h, n_src = info["w"], info["h"], info["frames"]
    n_out = int(round(n_src * TARGET / info["fps"]))
    ph, pw = ((h - 1) // 128 + 1) * 128, ((w - 1) // 128 + 1) * 128
    dec = subprocess.Popen([FF, "-v", "error", "-i", p, "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    tmp_out = p + ".smooth.mp4"
    enc_cmd = [FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(TARGET), "-i", "-"]
    if info["audio"]:
        enc_cmd += ["-i", p, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    enc_cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "15", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_out]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    size = w * h * 3

    def read():
        b = dec.stdout.read(size)
        return np.frombuffer(b, np.uint8).reshape(h, w, 3) if len(b) == size else None

    small = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    a, b = read(), read()
    k = 0                     # a = source frame k, b = k+1 (None past the end)
    diffs, cut, ta, tb = [], False, None, None
    held = written = 0

    def pair_changed():
        global cut, ta, tb
        ta = tb = None
        if b is None:
            cut = False
            return
        d = float(np.abs(small(a) - small(b)).mean())
        med = float(np.median(diffs[-48:])) if diffs else d
        cut = d > max(20.0, 6.0 * med) and len(diffs) >= 2
        diffs.append(d)

    pair_changed()
    for j in range(n_out):
        pos = j * info["fps"] / TARGET
        while b is not None and pos >= k + 1 - 1e-6:
            a, b, k = b, read(), k + 1
            pair_changed()
        t = pos - k
        if b is None or t < 1e-3:
            frame = a
        elif cut:
            frame = a if t < 0.5 else b
            held += 1
        else:
            if ta is None:
                ta, tb = to_t(a, h, w, ph, pw), to_t(b, h, w, ph, pw)
            o = model.inference(ta, tb, float(t), 1.0)
            frame = (o[0].clamp(0, 1) * 255. + 0.5).byte().permute(1, 2, 0).cpu().numpy()[:h, :w]
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        written += 1
        done_frames += 1
        if j % 60 == 0:
            say(f"PROGRESS {10 + int(80 * done_frames / total_out)}")
    enc.stdin.close()
    dec.stdout.close(); dec.wait()
    if enc.wait() != 0:
        raise RuntimeError(f"encode failed for {rel}")
    os.replace(tmp_out, p)
    report["clips"][rel] = {"src_fps": info["fps"], "src_frames": n_src, "out_frames": written, "cut_frames_held": held,
                            "secs": round(time.time() - c0, 1)}
    say(json.dumps({rel: report["clips"][rel]}))

# 5. commit on smooth60 so the farm's clone of DST renders the new clips
g = lambda *a: subprocess.run(["git", "-C", DST, *a], capture_output=True, text=True)
g("add", "-A")
c = g("-c", "user.name=smooth60", "-c", "user.email=smooth60@local", "commit", "-q", "--allow-empty", "-m", f"smooth60: RIFE v4.26 footage to {TARGET} fps")
if c.returncode != 0:
    raise RuntimeError("commit failed: " + c.stderr[-800:])
report["commit"] = g("rev-parse", "HEAD").stdout.strip()
report["secs"] = round(time.time() - t_start, 1)
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
say("PROGRESS 100")
say(json.dumps({k: v for k, v in report.items() if k != "skipped"})[:4000])
