"""Smooth 60 fps export, prep half (render-platform "60 fps" download button).

AI footage arrives at 24 fps and the platform renders at 30, so a fifth of the frames repeat. For a 60 fps
download this job clones the version's workspace at its render commit into C:\\rp-smooth\\<render id>
(the live workspace is only read by `git clone`), runs RIFE v4.26 on every footage clip the composition
references, keeps file names and audio, and commits on branch `smooth60`. The platform then queues a
normal hyperframes render of that branch with fps=60.

usage: python smooth60.py SRC_WS GIT_REF DST_WS OUT_DIR [--fps=60]

Model: rife-ncnn-vulkan rife-v4.6 (worker config RIFE_DIR), run on frames lanczos-upscaled 2x and
area-downscaled after. Why, from the 2026-11-06 A/B (render-farm spike/fps-ab):
  - at native 720p RIFE bends fine repeating detail (the Denza D9 grille slats); at 2x it does not;
  - RIFE v4.26 (latest) fixes the grille too, but its in-between frames are uneven on slow motion (a 5-frame
    stutter, 0.18-0.35 vs 0.066 step spread on the hand clip), with or without its own scale=2 — so not v4.26.
Cuts inside a clip get the nearest real frame, never a blend. Alpha clips (cutout mattes) are left alone.
"""
import glob, json, os, re, shutil, subprocess, sys, time
import numpy as np

SRC, REF, DST, OUT = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4])
TARGET = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--fps=")), "60"))
ROOT = r"C:\rp-smooth"
assert os.path.normcase(os.path.dirname(os.path.abspath(DST))) == os.path.normcase(ROOT), f"DST must be a child of {ROOT}"
KEEP_DAYS = 3
os.makedirs(OUT, exist_ok=True)
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

# 2. RIFE
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
RIFE = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
MODEL = os.path.join(RIFE_DIR, "rife-v4.6")
if not (os.path.exists(RIFE) and os.path.isdir(MODEL)):
    raise RuntimeError(f"rife-ncnn-vulkan / rife-v4.6 not found in {RIFE_DIR}")
UP = 2
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
report = {"ref": REF, "target": TARGET, "rife": "ncnn rife-v4.6 at 2x", "clips": {}, "skipped": {}}
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


# 4. interpolate each clip: ffmpeg lanczos 2x -> PNG -> rife-ncnn -n -> ffmpeg area 1x -> x264
def source_diffs(path, w, h):
    """Mean abs grey change per step at quarter size, for the cut guard."""
    sw, sh = max(16, w // 4), max(16, h // 4)
    p = subprocess.Popen([FF, "-v", "error", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough", "-vf", f"scale={sw}:{sh}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    out, prev = [], None
    while True:
        b = p.stdout.read(sw * sh)
        if len(b) < sw * sh:
            break
        g = np.frombuffer(b, np.uint8).astype(np.float32)
        if prev is not None:
            out.append(float(np.abs(g - prev).mean()))
        prev = g
    p.wait()
    return out


work = os.path.join(OUT, "work")
total_out = sum(int(round(i["frames"] * TARGET / i["fps"])) for _, _, i in todo) or 1
done_frames = 0
for p, rel, info in todo:
    c0 = time.time()
    w, h = info["w"], info["h"]
    src_dir, dst_dir = os.path.join(work, "src"), os.path.join(work, "dst")
    for d in (src_dir, dst_dir):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    stream([FF, "-v", "error", "-i", p, "-map", "0:v:0", "-fps_mode", "passthrough", "-vf", f"scale={w * UP}:{h * UP}:flags=lanczos",
            "-compression_level", "1", os.path.join(src_dir, "%08d.png")], "extract")
    names = sorted(os.listdir(src_dir))
    n_src = len(names)
    n_out = int(round(n_src * TARGET / info["fps"]))
    d = source_diffs(p, w, h)
    med = float(np.median(d)) if d else 0.0
    cuts = {i for i, x in enumerate(d) if x > max(20.0, 6.0 * med)}       # step i -> i+1 is a cut
    stream([RIFE, "-i", src_dir, "-o", dst_dir, "-n", str(n_out), "-m", MODEL, "-g", "0", "-j", "2:4:4", "-f", "%08d.png"], "rife")
    outs = sorted(os.listdir(dst_dir))
    # rife -n puts output i at source position i * n_src / n_out (main.cpp); across a cut use the nearest real frame
    held = 0
    for i, name in enumerate(outs):
        pos = i * n_src / len(outs)
        lo = int(np.floor(pos))
        if pos - lo > 1e-6 and lo in cuts:
            shutil.copy2(os.path.join(src_dir, names[min(n_src - 1, int(round(pos)))]), os.path.join(dst_dir, name))
            held += 1
    tmp_out = p + ".smooth.mp4"
    cmd = [FF, "-v", "error", "-y", "-framerate", str(TARGET), "-i", os.path.join(dst_dir, "%08d.png")]
    if info["audio"]:
        cmd += ["-i", p, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-vf", f"scale={w}:{h}:flags=area", "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-bf", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_out]
    stream(cmd, "encode")
    os.replace(tmp_out, p)
    done_frames += len(outs)
    report["clips"][rel] = {"src_fps": info["fps"], "src_frames": n_src, "out_frames": len(outs), "cut_frames_held": held,
                            "secs": round(time.time() - c0, 1)}
    say(f"PROGRESS {10 + int(80 * done_frames / total_out)}", json.dumps({rel: report["clips"][rel]}))
shutil.rmtree(work, ignore_errors=True)

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
