"""Spike (2026-11-06): is each model's in-between frame at timestep t really t of the way? 24->60 needs
t = .2 .4 .6 .8; a model that bends time makes a 5-frame stutter (seen in the v4.26 Denza render).
For sampled real pairs (a, b): phase-correlation shift of out(t) vs a, divided by shift of b vs a.

usage: python lin_check.py TEST_WS ORIG_REF OUT_DIR CLIP_REL [CLIP_REL ...]
"""
import glob, json, os, shutil, subprocess, sys
import cv2, numpy as np

WS, REF, OUT = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
CLIPS = sys.argv[4:]
RIFE_DIR = r"C:\Coding\tools\rife-ncnn-vulkan"
NCNN = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
PR = r"C:\rp-smooth\_rife\Practical-RIFE"
TS = [0.2, 0.4, 0.5, 0.6, 0.8]
os.makedirs(OUT, exist_ok=True)
say = lambda *a: print(*a, flush=True)
FF = shutil.which("ffmpeg") or glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", "ffmpeg.exe"), recursive=True)[0]

sys.path.insert(0, PR)
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
dev = torch.device("cuda")
from train_log.RIFE_HDv3 import Model
model = Model(); model.load_model(os.path.join(PR, "train_log"), -1); model.eval(); model.device()


def v426(a, b, t):
    h, w = a.shape[:2]; ph, pw = ((h - 1) // 128 + 1) * 128, ((w - 1) // 128 + 1) * 128
    tt = lambda f: F.pad(torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255., (0, pw - w, 0, ph - h))
    o = model.inference(tt(a), tt(b), t, 1.0)
    return (o[0].clamp(0, 1) * 255. + 0.5).byte().permute(1, 2, 0).cpu().numpy()[:h, :w]


tmp = os.path.join(OUT, "tmp")


def ncnn(a, b, t):
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    pa, pb, po = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png"), os.path.join(tmp, "o.png")
    cv2.imwrite(pa, a); cv2.imwrite(pb, b)
    subprocess.run([NCNN, "-0", pa, "-1", pb, "-o", po, "-s", str(t), "-m", os.path.join(RIFE_DIR, "rife-v4.6"), "-g", "0"],
                   capture_output=True)
    return cv2.imread(po)


def shift(x, y):
    gx = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32); gy = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(gx.shape[::-1], cv2.CV_32F)
    (dx, dy), r = cv2.phaseCorrelate(gx, gy, win)
    return np.array([dx, dy]), r


rep = {}
for rel in CLIPS:
    src = os.path.join(OUT, "orig.mp4")
    open(src, "wb").write(subprocess.run(["git", "-C", WS, "show", f"{REF}:{rel}"], capture_output=True).stdout)
    cap = cv2.VideoCapture(src); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    res = {"v426": {t: [] for t in TS}, "ncnn46": {t: [] for t in TS}}
    for i in range(2, len(frames) - 2, max(1, len(frames) // 24)):
        a, b = frames[i], frames[i + 1]
        full, r = shift(a, b)
        n = np.linalg.norm(full)
        if n < 1.5 or r < 0.2:
            continue
        u = full / n
        for name, fn in (("v426", v426), ("ncnn46", ncnn)):
            for t in TS:
                s, _ = shift(a, fn(a, b, t))
                res[name][t].append(float(np.dot(s, u) / n))
        say(f"PROGRESS 50 {rel[-10:]} pair {i} {n:.1f}px")
    rep[rel] = {m: {str(t): (round(float(np.median(v)), 3) if v else None) for t, v in d.items()} for m, d in res.items()}
    rep[rel]["pairs"] = len(res["v426"][0.5])
    say(json.dumps({rel: rep[rel]}))
shutil.rmtree(tmp, ignore_errors=True)
json.dump(rep, open(os.path.join(OUT, "lin.json"), "w"), indent=1)
say("done")
