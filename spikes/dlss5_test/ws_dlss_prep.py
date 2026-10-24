"""Spike (2026-10-24): copy a render-platform job workspace and run DLSS 5 (Visual Enhancer v18) on its
clean footage only, before HyperFrames composites the text. The live workspace is never written.

usage: python ws_dlss_prep.py SRC_WS DST_WS PRESET.json OUT_DIR
"""
import glob, json, os, shutil, subprocess, sys, time
import cv2, numpy as np

SRC, DST, PRESET, OUT = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3]), os.path.abspath(sys.argv[4])
VE = r"C:\tools\VisualEnhancer-v18\VE_CLI.exe"
assert os.path.normcase(DST).startswith(os.path.normcase(r"C:\rp-workspace-tests")), "DST must be under C:\rp-workspace-tests"
CLIPS = ["badge-push.mp4", "generated-03-seedance-20261021104218-t8ymx.mp4", "drive-in-poster.mp4",
         "generated-14-edit-replace-8e7f12f3-8165-4667-a607-5735a031061c-take2.mp4", "park-arrive-fixed-b.mp4",
         "generated-16-boot-unload-v2.mp4", "boot-unload-v2-c.mp4", "generated-12-seedance-20261021111128-nwyrf.mp4",
         "generated-04-seedance-20261021104224-dso6n.mp4"]
MATTE, HOLD, HOLD_SRC = "car-rest-matte.webm", "rest-last.png", "generated-04-seedance-20261021104224-dso6n.mp4"
os.makedirs(OUT, exist_ok=True)
log = {"clips": {}}
ff = (glob.glob(r"C:\tools\VisualEnhancer-v18\**\ffmpeg.exe", recursive=True) or [shutil.which("ffmpeg")])[0]
log["ffmpeg"] = ff


def ve(src, dst):
    t0 = time.time()
    p = subprocess.run([VE, "render", src, "--preset", PRESET, "--codec", "h264", "--quality", "max", "--json", "-o", dst],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=os.path.dirname(VE), timeout=1800)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"VE failed on {src}: rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return round(time.time() - t0, 1)


def frames(path):
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(".production-checks", "snapshots", ".transcode-cache", "node_modules"))
A = os.path.join(DST, "assets"); tmp = os.path.join(OUT, "tmp"); os.makedirs(tmp, exist_ok=True)
print("PROGRESS 5", flush=True)
for i, c in enumerate(CLIPS):
    src = os.path.join(A, c); dst = os.path.join(tmp, c)
    log["clips"][c] = ve(src, dst)
    shutil.copy2(src, os.path.join(tmp, "orig-" + c))
    os.replace(dst, src)
    print(f"PROGRESS {5 + 70 * (i + 1) // len(CLIPS)}", json.dumps({c: log["clips"][c]}), flush=True)

# matte: enhance the RGB, keep the original alpha
m = os.path.join(A, MATTE)
rgb, rgb_e = os.path.join(tmp, "matte-rgb.mp4"), os.path.join(tmp, "matte-rgb-e.mp4")
subprocess.check_call([ff, "-v", "error", "-y", "-c:v", "libvpx-vp9", "-i", m, "-map", "0:v", "-c:v", "libx264", "-crf", "8",
                       "-pix_fmt", "yuv420p", rgb])
log["matte"] = ve(rgb, rgb_e)
shutil.copy2(m, os.path.join(tmp, "orig-" + MATTE))
new_m = os.path.join(tmp, "matte-new.webm")
subprocess.check_call([ff, "-v", "error", "-y", "-c:v", "libvpx-vp9", "-i", m, "-i", rgb_e, "-filter_complex",
                       "[0:v]alphaextract[a];[1:v][a]alphamerge,format=yuva420p[v]", "-map", "[v]", "-c:v", "libvpx-vp9",
                       "-crf", "18", "-b:v", "0", "-auto-alt-ref", "0", "-pix_fmt", "yuva420p", new_m])
os.replace(new_m, m)
print("PROGRESS 85", flush=True)

# hold frame: the enhanced clip's frame that matches the original still
hold = os.path.join(A, HOLD); shutil.copy2(hold, os.path.join(tmp, "orig-" + HOLD))
still = cv2.imread(hold, cv2.IMREAD_UNCHANGED)
of = frames(os.path.join(tmp, "orig-" + HOLD_SRC)); ef = frames(os.path.join(A, HOLD_SRC))
small = lambda f: cv2.resize(f[..., :3], (90, 160), interpolation=cv2.INTER_AREA).astype(np.float32)
s = small(still); errs = [float(np.abs(small(f) - s).mean()) for f in of]
k = int(np.argmin(errs)); log["hold"] = {"match_frame": k, "err": round(errs[k], 2), "frames": len(of), "still_shape": list(still.shape)}
e = ef[min(k, len(ef) - 1)]
if e.shape[:2] != still.shape[:2]:
    e = cv2.resize(e, (still.shape[1], still.shape[0]), interpolation=cv2.INTER_LANCZOS4)
if still.ndim == 3 and still.shape[2] == 4:
    e = np.dstack([e, still[..., 3]])
cv2.imwrite(hold, e)
print("PROGRESS 95", flush=True)

for f in glob.glob(os.path.join(tmp, "*")):
    if not os.path.basename(f).startswith("orig-"):
        os.remove(f)
shutil.rmtree(tmp)   # originals stay in SRC; keep the job output small
json.dump(log, open(os.path.join(OUT, "prep.json"), "w"), indent=1)
print(json.dumps(log))
