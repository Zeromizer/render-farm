"""Spike (2026-11-06): fps A/B on a render-platform job. Clone the job workspace at a fixed commit into
C:\\rp-workspace-tests (the live workspace is only read by `git clone`), RIFE every referenced opaque
footage clip below TARGET fps up to TARGET, keep file names, commit, and report per-clip stats.

usage: python fps_prep.py SRC_WS REF DST_WS OUT_DIR TARGET_FPS [--probe-only]

Cuts inside a clip: output frames whose two source neighbours straddle a cut get the nearest source
frame instead of a RIFE blend. Alpha clips (webm/mov with alpha) are left alone and listed.
"""
import glob, json, os, re, shutil, subprocess, sys, time
import cv2, numpy as np

SRC, REF, DST, OUT, TARGET = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4]), int(sys.argv[5])
PROBE_ONLY = "--probe-only" in sys.argv
assert os.path.normcase(DST).startswith(os.path.normcase(r"C:\rp-workspace-tests")), "DST must be under C:\\rp-workspace-tests"
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
RIFE = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
MODEL = os.path.join(RIFE_DIR, "rife-v4.6")
os.makedirs(OUT, exist_ok=True)


def say(*a):
    print(*a, flush=True)


def find_ffmpeg(name):
    p = shutil.which(name)
    if p:
        return p
    d = os.environ.get("FFMPEG_DIR")
    if d and os.path.exists(os.path.join(d, name + ".exe")):
        return os.path.join(d, name + ".exe")
    hits = glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", name + ".exe"), recursive=True)
    return hits[0]


FF, FP = find_ffmpeg("ffmpeg"), find_ffmpeg("ffprobe")


def stream(cmd, tag):
    """Run a child and echo its output line by line (the farm only checks cancel/timeout on output)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-20:]
            say(f"[{tag}] {line[:200]}")
    if p.wait() != 0:
        raise RuntimeError(f"{tag} failed rc={p.returncode}: " + "\n".join(tail))


def probe(path):
    j = json.loads(subprocess.run([FP, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                                  capture_output=True, text=True).stdout or "{}")
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v:
        return None
    num, den = (v.get("r_frame_rate") or "0/1").split("/")
    an, ad = (v.get("avg_frame_rate") or "0/1").split("/")
    tags = v.get("tags") or {}
    pf = v.get("pix_fmt") or ""
    alpha = any(x in pf for x in ("yuva", "rgba", "argb", "bgra", "abgr", "gbrap", "ya")) \
        or str(tags.get("alpha_mode", tags.get("ALPHA_MODE", ""))) == "1"
    return {"w": v.get("width"), "h": v.get("height"), "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"),
            "r_fps": round(int(num) / max(1, int(den)), 3), "avg_fps": round(int(an) / max(1, int(ad)), 3),
            "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
            "dur": float(j.get("format", {}).get("duration", 0) or 0), "audio": bool(a), "alpha": bool(alpha),
            "bitrate": int(j.get("format", {}).get("bit_rate", 0) or 0)}


def diffs(frame_dir):
    names = sorted(os.listdir(frame_dir))
    prev, out = None, []
    for n in names:
        f = cv2.imread(os.path.join(frame_dir, n), cv2.IMREAD_REDUCED_GRAYSCALE_4).astype(np.float32)
        if prev is not None:
            out.append(float(np.abs(f - prev).mean()))
        prev = f
    return names, out


def judder(d):
    """Share of steps that moved far less than their neighbours (repeated frames)."""
    if len(d) < 5:
        return 0.0
    med = np.array([np.median(d[max(0, i - 3):i + 4]) for i in range(len(d))])
    return round(float(np.mean((np.array(d) < 0.25 * med) & (med > 0.3))) * 100, 1)


def cuts(d):
    if len(d) < 5:
        return []
    med = np.median(d)
    return [i for i, x in enumerate(d) if x > max(12.0, 6 * med)]   # step i -> i+1 is a cut


# 1. clone the job workspace at REF (read-only on the live one)
t0 = time.time()
if os.path.exists(DST):
    stream(["cmd", "/c", "rmdir", "/s", "/q", DST], "rm")
stream(["git", "clone", "--no-checkout", SRC, DST], "clone")
stream(["git", "-C", DST, "checkout", "-q", "-B", "fps-ab", REF], "checkout")
say("PROGRESS 5", json.dumps({"clone_s": round(time.time() - t0, 1)}))

# 2. which media do the compositions reference?
html = ""
for h in glob.glob(os.path.join(DST, "**", "*.*"), recursive=True):
    if h.lower().endswith((".html", ".js", ".json", ".css")) and "node_modules" not in h and ".git" + os.sep not in h:
        html += open(h, encoding="utf-8", errors="replace").read()
comp_fps = re.findall(r'data-fps=["\']?(\d+)', html)
refs = set(os.path.basename(m) for m in re.findall(r'''["'(]([^"'()]+\.(?:mp4|mov|webm|m4v))''', html, re.I))
report = {"ref": REF, "target": TARGET, "comp_fps_attrs": sorted(set(comp_fps)), "rife_model": MODEL, "clips": {}, "skipped": {}}
vids = [p for p in glob.glob(os.path.join(DST, "**", "*.*"), recursive=True)
        if p.lower().endswith((".mp4", ".mov", ".webm", ".m4v")) and ".git" not in p and "node_modules" not in p]
todo = []
for p in vids:
    rel = os.path.relpath(p, DST)
    info = probe(p)
    if info is None:
        continue
    used = os.path.basename(p) in refs
    if not used:
        report["skipped"][rel] = {"why": "not referenced", **info}
    elif info["alpha"]:
        report["skipped"][rel] = {"why": "alpha (left as is)", **info}
    elif info["r_fps"] >= TARGET - 0.5:
        report["skipped"][rel] = {"why": f"already >= {TARGET}", **info}
    else:
        todo.append((p, rel, info))
say("PROGRESS 8", json.dumps({"videos": len(vids), "todo": [t[1] for t in todo], "comp_fps": report["comp_fps_attrs"]}))

# 3. RIFE each clip
tmp = os.path.join(OUT, "tmp")
for k, (p, rel, info) in enumerate(todo):
    c0 = time.time()
    src_dir, dst_dir = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
    for d in (src_dir, dst_dir):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    stream([FF, "-v", "error", "-i", p, "-map", "0:v:0", "-fps_mode", "passthrough", os.path.join(src_dir, "%08d.png")], "extract")
    names, d = diffs(src_dir)
    n_src = len(names)
    cut_steps = cuts(d)
    entry = {**info, "src_frames": n_src, "src_judder_pct": judder(d), "cuts_at_step": cut_steps}
    if PROBE_ONLY:
        report["clips"][rel] = entry
        say(f"PROGRESS {8 + 80 * (k + 1) // len(todo)}", json.dumps({rel: entry}))
        continue
    fps_src = info["r_fps"] or info["avg_fps"]
    n_out = int(round(n_src * TARGET / fps_src))
    stream([RIFE, "-i", src_dir, "-o", dst_dir, "-n", str(n_out), "-m", MODEL, "-g", "0", "-j", "2:4:4", "-f", "%08d.png"], "rife")
    outs = sorted(os.listdir(dst_dir))
    # cut guard: output i sits at source position i*n_src/n_out (rife main.cpp); straddling a cut -> nearest frame
    fixed = 0
    cutset = set(cut_steps)
    for i, name in enumerate(outs):
        pos = i * n_src / len(outs)
        lo = int(np.floor(pos))
        if pos - lo > 1e-6 and lo in cutset:
            shutil.copy2(os.path.join(src_dir, names[min(n_src - 1, int(round(pos)))]), os.path.join(dst_dir, name))
            fixed += 1
    new = p + ".rife.mp4"
    cmd = [FF, "-v", "error", "-y", "-framerate", str(TARGET), "-i", os.path.join(dst_dir, "%08d.png")]
    if info["audio"]:
        cmd += ["-i", p, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "14", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", new]
    stream(cmd, "encode")
    _, d2 = diffs(dst_dir)
    os.replace(new, p)
    entry.update({"out_frames": len(outs), "out_fps": TARGET, "cut_frames_held": fixed, "out_judder_pct": judder(d2),
                  "out_dur": probe(p)["dur"], "secs": round(time.time() - c0, 1)})
    report["clips"][rel] = entry
    say(f"PROGRESS {8 + 80 * (k + 1) // len(todo)}", json.dumps({rel: entry}))
shutil.rmtree(tmp, ignore_errors=True)

# 4. commit inside the TEST copy so the farm's clone of it sees the new clips
if not PROBE_ONLY and todo:
    g = lambda *a: subprocess.run(["git", "-C", DST, *a], capture_output=True, text=True)
    g("add", "-A")
    c = g("-c", "user.name=fps-spike", "-c", "user.email=spike@local", "commit", "-q", "-m", f"fps-ab: RIFE footage to {TARGET} fps")
    report["commit"] = g("rev-parse", "HEAD").stdout.strip()
    report["commit_err"] = c.stderr[-500:]
report["total_s"] = round(time.time() - t0, 1)
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
say("PROGRESS 99")
say(json.dumps({k: v for k, v in report.items() if k != "skipped"})[:6000])
