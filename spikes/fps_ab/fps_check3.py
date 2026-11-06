"""Spike (2026-11-06): RIFE v4.26 (latest, official PyTorch: hzwer/Practical-RIFE @ pinned sha + the author's
hzwer/RIFE HF weights RIFEv4.26_0921.zip) vs our rife-ncnn-vulkan v4.6. Same self-check as fps_check2.py
(drop alternate real frames, re-create, compare per 32 px block).

usage: python fps_check3.py TEST_WS ORIG_REF OUT_DIR CLIP_REL=x0,y0,x1,y1 [CLIP_REL=roi ...]
configs: ncnn46_up2 | v426_s1 | v426_s2 (model's own scale=2 flow) | v426_up2 (lanczos 2x in, area out)
"""
import glob, io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile
import cv2, numpy as np

WS, REF, OUT = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
JOBS = [(a.split("=")[0], [int(v) for v in a.split("=")[1].split(",")]) for a in sys.argv[4:]]
CONFIGS = os.environ.get("CONFIGS", "ncnn46_up2,v426_s1,v426_s2,v426_up2").split(",")
assert os.path.normcase(WS).startswith(os.path.normcase(r"C:\rp-workspace-tests")), "test workspaces only"
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
NCNN = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
CACHE = r"C:\rp-workspace-tests\_rife"
PR_SHA = "bbfd2ea90910789a860ea3e2b32a240cd577b75e"
W426 = "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip"
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
say = lambda *a: print(*a, flush=True)
FF = shutil.which("ffmpeg") or glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", "ffmpeg.exe"), recursive=True)[0]


def stream(cmd, tag):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        if line.strip():
            say(f"[{tag}] {line.rstrip()[:160]}")
    if p.wait() != 0:
        raise RuntimeError(f"{tag} rc={p.returncode}")


# --- official RIFE 4.26 (code pinned, weights from the author's HF) ---
pr = os.path.join(CACHE, "Practical-RIFE")
if not os.path.isdir(os.path.join(pr, "model")):
    stream(["git", "clone", "-q", "https://github.com/hzwer/Practical-RIFE", pr], "clone")
stream(["git", "-C", pr, "checkout", "-q", PR_SHA], "checkout")
tl = os.path.join(pr, "train_log")
if not os.path.exists(os.path.join(tl, "flownet.pkl")):
    say("downloading", W426)
    data = urllib.request.urlopen(W426, timeout=300).read()
    say("weights zip bytes", len(data))
    os.makedirs(tl, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            base = os.path.basename(m)
            if base.endswith((".py", ".pkl")) and "__MACOSX" not in m:
                open(os.path.join(tl, base), "wb").write(z.read(m))
    say("train_log:", os.listdir(tl))
sys.path.insert(0, pr)
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
say("torch", torch.__version__, "device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
from train_log.RIFE_HDv3 import Model
model = Model(); model.load_model(tl, -1); model.eval(); model.device()
say("model version", getattr(model, "version", "?"))


def v426(A, B, t, scale=1.0, up=1):
    h0, w0 = A.shape[:2]
    if up > 1:
        A = cv2.resize(A, (w0 * up, h0 * up), interpolation=cv2.INTER_LANCZOS4)
        B = cv2.resize(B, (w0 * up, h0 * up), interpolation=cv2.INTER_LANCZOS4)
    h, w = A.shape[:2]
    tmp = max(128, int(128 / scale)); ph, pw = ((h - 1) // tmp + 1) * tmp, ((w - 1) // tmp + 1) * tmp
    tt = lambda f: F.pad(torch.from_numpy(f.transpose(2, 0, 1).copy()).to(dev, non_blocking=True).unsqueeze(0).float() / 255., (0, pw - w, 0, ph - h))
    o = model.inference(tt(A), tt(B), t, scale)
    o = (o[0].clamp(0, 1) * 255. + 0.5).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
    return cv2.resize(o, (w0, h0), interpolation=cv2.INTER_AREA) if up > 1 else o


def block_err(a, b, bs=32):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    h, w = (d.shape[0] // bs) * bs, (d.shape[1] // bs) * bs
    return d[:h, :w].reshape(h // bs, bs, w // bs, bs).mean(axis=(1, 3))


rep = {"pr_sha": PR_SHA, "weights": W426, "clips": {}}
tmpd = os.path.join(OUT, "tmp")
for ji, (rel, (X0, Y0, X1, Y1)) in enumerate(JOBS):
    shutil.rmtree(tmpd, ignore_errors=True); src = os.path.join(tmpd, "src"); os.makedirs(src)
    orig = os.path.join(tmpd, "orig.mp4")
    open(orig, "wb").write(subprocess.run(["git", "-C", WS, "show", f"{REF}:{rel}"], capture_output=True).stdout)
    stream([FF, "-v", "error", "-i", orig, "-map", "0:v:0", "-fps_mode", "passthrough", os.path.join(src, "%08d.png")], "extract")
    frames = [cv2.imread(os.path.join(src, n)) for n in sorted(os.listdir(src))]
    H, W = frames[0].shape[:2]; even = frames[::2]
    targets = [i for i in range(1, len(frames), 2) if i // 2 + 1 < len(even)]
    syn_all, entry = {}, {}
    for cfg in CONFIGS:
        t0 = time.time(); syn = {}
        if cfg == "ncnn46_up2":
            ev, out = os.path.join(tmpd, "ev"), os.path.join(tmpd, "out")
            shutil.rmtree(ev, ignore_errors=True); shutil.rmtree(out, ignore_errors=True); os.makedirs(ev); os.makedirs(out)
            for k, f in enumerate(even):
                cv2.imwrite(os.path.join(ev, f"{k + 1:08d}.png"), cv2.resize(f, (W * 2, H * 2), interpolation=cv2.INTER_LANCZOS4))
            stream([NCNN, "-i", ev, "-o", out, "-n", str(2 * len(even)), "-m", os.path.join(RIFE_DIR, "rife-v4.6"), "-g", "0",
                    "-j", "2:4:4", "-f", "%08d.png"], cfg)
            outs = sorted(os.listdir(out))
            for i in targets:
                syn[i] = cv2.resize(cv2.imread(os.path.join(out, outs[i])), (W, H), interpolation=cv2.INTER_AREA)
        else:
            scale = 2.0 if cfg == "v426_s2" else 1.0
            up = 2 if cfg == "v426_up2" else 1
            for n, i in enumerate(targets):
                syn[i] = v426(even[i // 2], even[i // 2 + 1], 0.5, scale, up)
                if n % 25 == 0:
                    say(f"[{cfg}] {n}/{len(targets)}")
        E = np.stack([block_err(frames[i], syn[i]) for i in targets])
        R = E[:, Y0 // 32:Y1 // 32, X0 // 32:X1 // 32]
        entry[cfg] = {"secs": round(time.time() - t0, 1), "mean": round(float(E.mean()), 3), "roi_mean": round(float(R.mean()), 3),
                      "roi_p99": round(float(np.percentile(R, 99)), 2), "roi_bad_blocks_10pct": int(((R > 12).mean(axis=0) > 0.10).sum()),
                      "whole_bad_blocks_10pct": int(((E > 12).mean(axis=0) > 0.10).sum()),
                      "roi_worst_frame": targets[int(R.mean(axis=(1, 2)).argmax())]}
        syn_all[cfg] = syn
        say(json.dumps({rel[-12:]: {cfg: entry[cfg]}}))
    picks = sorted(set(entry[c]["roi_worst_frame"] for c in entry))[:3]
    crop = lambda f: f[Y0:Y1, X0:X1]
    cv2.imwrite(os.path.join(OUT, f"cmp-{os.path.basename(rel)[-9:-4]}.jpg"),
                np.vstack([np.hstack([crop(frames[i])] + [crop(syn_all[c][i]) for c in syn_all]) for i in picks]))
    rep["clips"][rel] = {"roi": [X0, Y0, X1, Y1], "cols": ["real"] + list(syn_all), "rows": picks, "configs": entry}
    say(f"PROGRESS {100 * (ji + 1) // (len(JOBS) + 1)}")
shutil.rmtree(tmpd, ignore_errors=True)
json.dump(rep, open(os.path.join(OUT, "check3.json"), "w"), indent=1)
say("done")
