"""Spike (2026-11-06): which interpolator stops the Denza D9 grille warping? Same self-check as fps_check.py
(drop alternate real frames, re-create, compare per 32 px block) on ONE clip, for:
  ncnn46      rife-ncnn-vulkan rife-v4.6 (what fps_prep used)
  ncnn46_up2  same, frames lanczos-upscaled 2x before RIFE, area-downscaled after
  onnx49      RIFE 4.9 ensemble ONNX (~/.cache/rife/rife49.onnx, onnxruntime-directml)
  onnx49_up2  same with the 2x upscale

usage: python fps_check2.py TEST_WS ORIG_REF CLIP_REL OUT_DIR ROI(x0,y0,x1,y1 in clip px) [configs...]
"""
import glob, json, os, shutil, subprocess, sys, time
import cv2, numpy as np

WS, REF, REL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4])
X0, Y0, X1, Y1 = [int(v) for v in sys.argv[5].split(",")]
CONFIGS = sys.argv[6:] or ["ncnn46", "ncnn46_up2", "onnx49", "onnx49_up2"]
assert os.path.normcase(WS).startswith(os.path.normcase(r"C:\rp-workspace-tests")), "test workspaces only"
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
RIFE = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
ONNX = os.path.join(os.path.expanduser("~"), ".cache", "rife", "rife49.onnx")
os.makedirs(OUT, exist_ok=True)
say = lambda *a: print(*a, flush=True)
FF = shutil.which("ffmpeg") or glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", "ffmpeg.exe"), recursive=True)[0]


def stream(cmd, tag):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        if line.strip():
            say(f"[{tag}] {line.rstrip()[:160]}")
    if p.wait() != 0:
        raise RuntimeError(f"{tag} rc={p.returncode}")


def block_err(a, b, bs=32):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    h, w = (d.shape[0] // bs) * bs, (d.shape[1] // bs) * bs
    return d[:h, :w].reshape(h // bs, bs, w // bs, bs).mean(axis=(1, 3))


tmp = os.path.join(OUT, "tmp"); shutil.rmtree(tmp, ignore_errors=True)
src = os.path.join(tmp, "src"); os.makedirs(src)
orig = os.path.join(tmp, "orig.mp4")
open(orig, "wb").write(subprocess.run(["git", "-C", WS, "show", f"{REF}:{REL}"], capture_output=True).stdout)
stream([FF, "-v", "error", "-i", orig, "-map", "0:v:0", "-fps_mode", "passthrough", os.path.join(src, "%08d.png")], "extract")
names = sorted(os.listdir(src))
frames = [cv2.imread(os.path.join(src, n)) for n in names]
H, W = frames[0].shape[:2]
even = frames[::2]
targets = [i for i in range(1, len(frames), 2) if i // 2 + 1 < len(even)]   # real odd frame i sits between even[i//2] and even[i//2+1]
say("PROGRESS 5", json.dumps({"frames": len(frames), "size": [W, H], "targets": len(targets)}))


def ncnn(up):
    ev = os.path.join(tmp, f"ev{up}"); out = os.path.join(tmp, f"out{up}")
    shutil.rmtree(ev, ignore_errors=True); shutil.rmtree(out, ignore_errors=True); os.makedirs(ev); os.makedirs(out)
    for k, f in enumerate(even):
        g = cv2.resize(f, (W * up, H * up), interpolation=cv2.INTER_LANCZOS4) if up > 1 else f
        cv2.imwrite(os.path.join(ev, f"{k + 1:08d}.png"), g)
    stream([RIFE, "-i", ev, "-o", out, "-n", str(2 * len(even)), "-m", os.path.join(RIFE_DIR, "rife-v4.6"), "-g", "0",
            "-j", "2:4:4", "-f", "%08d.png"], f"ncnn{up}")
    outs = sorted(os.listdir(out))
    res = {}
    for i in targets:
        o = cv2.imread(os.path.join(out, outs[i]))
        res[i] = cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA) if up > 1 else o
    return res


_sess = None


def onnx(up):
    global _sess
    import onnxruntime as ort
    if _sess is None:
        prov = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        _sess = ort.InferenceSession(ONNX, providers=prov)
        say("onnx provider", _sess.get_providers()[0])
    res = {}
    for n, i in enumerate(targets):
        A, B = even[i // 2], even[i // 2 + 1]
        if up > 1:
            A = cv2.resize(A, (W * up, H * up), interpolation=cv2.INTER_LANCZOS4)
            B = cv2.resize(B, (W * up, H * up), interpolation=cv2.INTER_LANCZOS4)
        h, w = A.shape[:2]; ph, pw = (-h) % 64, (-w) % 64
        prep = lambda f: np.ascontiguousarray(np.pad(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0,
                                                     ((0, 0), (0, ph), (0, pw)), mode="edge")[None])
        o = _sess.run(None, {"img0": prep(A), "img1": prep(B), "timestep": np.array([0.5], np.float32)})[0]
        o = cv2.cvtColor(np.clip(np.squeeze(o)[:, :h, :w].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        res[i] = cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA) if up > 1 else o
        if n % 10 == 0:
            say(f"[onnx{up}] {n}/{len(targets)}")
    return res


rep = {"clip": REL, "roi": [X0, Y0, X1, Y1], "configs": {}}
syn_all = {}
for ci, cfg in enumerate(CONFIGS):
    t0 = time.time()
    up = 2 if cfg.endswith("_up2") else 1
    try:
        syn = ncnn(up) if cfg.startswith("ncnn") else onnx(up)
    except Exception as e:
        rep["configs"][cfg] = {"error": str(e)[:500]}; say(cfg, "ERROR", e); continue
    E = np.stack([block_err(frames[i], syn[i]) for i in targets])
    r0, r1, c0, c1 = Y0 // 32, Y1 // 32, X0 // 32, X1 // 32
    R = E[:, r0:r1, c0:c1]
    worst_i = targets[int(R.mean(axis=(1, 2)).argmax())]
    rep["configs"][cfg] = {"secs": round(time.time() - t0, 1), "mean": round(float(E.mean()), 2),
                           "roi_mean": round(float(R.mean()), 2), "roi_p99": round(float(np.percentile(R, 99)), 2),
                           "roi_bad_blocks_10pct": int(((R > 12).mean(axis=0) > 0.10).sum()),
                           "whole_bad_blocks_10pct": int(((E > 12).mean(axis=0) > 0.10).sum()), "roi_worst_frame": worst_i}
    syn_all[cfg] = syn
    say(f"PROGRESS {5 + 85 * (ci + 1) // len(CONFIGS)}", json.dumps({cfg: rep["configs"][cfg]}))

# side-by-side ROI crops on the frames that were worst for the baseline: real | each config
base = next((c for c in CONFIGS if c in syn_all), None)
if base:
    picks = sorted(set([rep["configs"][base]["roi_worst_frame"]] + [rep["configs"][c]["roi_worst_frame"] for c in syn_all]))[:4]
    rows = []
    for i in picks:
        crop = lambda f: cv2.resize(f[Y0:Y1, X0:X1], ((X1 - X0) * 2 // 2, (Y1 - Y0) * 2 // 2))
        tiles = [crop(frames[i])] + [crop(syn_all[c][i]) for c in syn_all]
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(OUT, "roi-compare.jpg"), np.vstack(rows))
    rep["compare_cols"] = ["real"] + list(syn_all); rep["compare_rows"] = picks
shutil.rmtree(tmp, ignore_errors=True)
json.dump(rep, open(os.path.join(OUT, "check2.json"), "w"), indent=1)
say(json.dumps(rep))
