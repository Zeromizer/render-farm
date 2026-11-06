"""Spike (2026-11-06): evenness of 24->60 on SLOW footage. v4.26 at native res made 0.42/0.68/0.30/0.68/0.33 px
steps on the Denza hand clip (ncnn v4.6 at 2x: 0.53 flat). For each config, build the 60 fps sequence in
memory exactly like tools/smooth60 and report per-phase median phase-correlation step + the slats ROI error.

usage: python even_check.py TEST_WS ORIG_REF OUT_DIR CLIP_REL [CLIP_REL ...]   (CONFIGS env: v426_s1,v426_s2,v426_up2,ncnn46_up2)
"""
import glob, json, os, shutil, subprocess, sys, time
import cv2, numpy as np

WS, REF, OUT = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
CLIPS = sys.argv[4:]
CONFIGS = os.environ.get("CONFIGS", "v426_s1,v426_s2,v426_up2,ncnn46_up2").split(",")
RIFE_DIR = r"C:\Coding\tools\rife-ncnn-vulkan"
NCNN = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
PR = r"C:\rp-smooth\_rife\Practical-RIFE"
os.makedirs(OUT, exist_ok=True)
say = lambda *a: print(*a, flush=True)

sys.path.insert(0, PR)
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
dev = torch.device("cuda")
from train_log.RIFE_HDv3 import Model
model = Model(); model.load_model(os.path.join(PR, "train_log"), -1); model.eval(); model.device()


def seq_v426(frames, scale, up):
    h0, w0 = frames[0].shape[:2]
    fr = [cv2.resize(f, (w0 * up, h0 * up), interpolation=cv2.INTER_LANCZOS4) for f in frames] if up > 1 else frames
    h, w = fr[0].shape[:2]
    tmp = max(128, int(128 / scale)); ph, pw = ((h - 1) // tmp + 1) * tmp, ((w - 1) // tmp + 1) * tmp
    tt = lambda f: F.pad(torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255., (0, pw - w, 0, ph - h))
    n_out = int(round(len(frames) * 60 / 24)); out = []
    for j in range(n_out):
        pos = j * 24 / 60; k = int(np.floor(pos + 1e-6)); t = pos - k
        if k >= len(frames) - 1 or t < 1e-3:
            o = fr[min(k, len(fr) - 1)]
        else:
            o = (model.inference(tt(fr[k]), tt(fr[k + 1]), float(t), scale)[0].clamp(0, 1) * 255. + 0.5).byte().permute(1, 2, 0).cpu().numpy()[:h, :w]
        out.append(cv2.resize(o, (w0, h0), interpolation=cv2.INTER_AREA) if up > 1 else o)
        if j % 100 == 0:
            say(f"[v426 s{scale} up{up}] {j}/{n_out}")
    return out


def seq_ncnn(frames, up):
    d = os.path.join(OUT, "tmp"); shutil.rmtree(d, ignore_errors=True)
    i_, o_ = os.path.join(d, "i"), os.path.join(d, "o"); os.makedirs(i_); os.makedirs(o_)
    h0, w0 = frames[0].shape[:2]
    for k, f in enumerate(frames):
        cv2.imwrite(os.path.join(i_, f"{k + 1:08d}.png"), cv2.resize(f, (w0 * up, h0 * up), interpolation=cv2.INTER_LANCZOS4) if up > 1 else f)
    n_out = int(round(len(frames) * 60 / 24))
    p = subprocess.Popen([NCNN, "-i", i_, "-o", o_, "-n", str(n_out), "-m", os.path.join(RIFE_DIR, "rife-v4.6"), "-g", "0", "-j", "2:4:4", "-f", "%08d.png"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in p.stdout:
        say("[ncnn]", line.rstrip()[:120])
    p.wait()
    out = [cv2.resize(cv2.imread(os.path.join(o_, n)), (w0, h0), interpolation=cv2.INTER_AREA) for n in sorted(os.listdir(o_))]
    shutil.rmtree(d, ignore_errors=True)
    return out


def evenness(seq):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in seq]
    win = cv2.createHanningWindow(g[0].shape[::-1], cv2.CV_32F)
    s = np.array([np.hypot(*cv2.phaseCorrelate(g[i - 1], g[i], win)[0]) for i in range(1, len(g))])
    L = (len(s) - 5) // 5 * 5
    m = np.array([np.median(s[k:L:5]) for k in range(5)])
    return [round(float(x), 3) for x in m], round(float(m.std() / max(m.mean(), 1e-6)), 3)


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
    cap.release()
    rep[rel] = {}
    for cfg in CONFIGS:
        t0 = time.time()
        if cfg.startswith("v426"):
            seq = seq_v426(frames, 2.0 if cfg == "v426_s2" else 1.0, 2 if cfg == "v426_up2" else 1)
        else:
            seq = seq_ncnn(frames, 2 if cfg.endswith("up2") else 1)
        med, cv = evenness(seq)
        rep[rel][cfg] = {"phase_medians": med, "unevenness": cv, "secs": round(time.time() - t0, 1)}
        say(f"PROGRESS 50", json.dumps({rel[-10:]: {cfg: rep[rel][cfg]}}))
json.dump(rep, open(os.path.join(OUT, "even.json"), "w"), indent=1)
say("done")
