"""Spike (2026-11-06): where does RIFE warp? Self-check on the ORIGINAL clips of a test workspace copy:
drop every other real frame, let RIFE re-create it, compare with the real one, per 32 px block.
A block RIFE gets wrong at a 1/12 s gap is where it warps at 1/24 s too (the Denza grille).

usage: python fps_check.py TEST_WS ORIG_REF OUT_DIR [config ...]
config = model[:tta]  e.g. rife-v4.6  rife-v4.6:x  rife-v4.6:xz   (default: every rife-v4* model in RIFE_DIR, plus :x on v4.6)
"""
import glob, json, os, shutil, subprocess, sys, time
import cv2, numpy as np

WS, REF, OUT = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
assert os.path.normcase(WS).startswith(os.path.normcase(r"C:\rp-workspace-tests")), "test workspaces only"
RIFE_DIR = os.environ.get("RIFE_DIR", r"C:\Coding\tools\rife-ncnn-vulkan")
RIFE = os.path.join(RIFE_DIR, "rife-ncnn-vulkan.exe")
os.makedirs(OUT, exist_ok=True)
say = lambda *a: print(*a, flush=True)
FF = shutil.which("ffmpeg") or glob.glob(os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"), "**", "ffmpeg.exe"), recursive=True)[0]

models = sorted(d for d in os.listdir(RIFE_DIR) if d.startswith("rife") and os.path.isdir(os.path.join(RIFE_DIR, d)))
cache = glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "rife", "*"))
say(json.dumps({"rife_dir_models": models, "onnx_cache": cache}))
configs = sys.argv[4:] or ([m for m in models if m.startswith("rife-v4")] + ["rife-v4.6:x"])


def stream(cmd, tag):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        if line.strip():
            tail = (tail + [line.rstrip()])[-10:]
            say(f"[{tag}] {line.rstrip()[:160]}")
    if p.wait() != 0:
        raise RuntimeError(f"{tag} rc={p.returncode}: " + "\n".join(tail))


def block_err(a, b, bs=32):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    h, w = (d.shape[0] // bs) * bs, (d.shape[1] // bs) * bs
    return d[:h, :w].reshape(h // bs, bs, w // bs, bs).mean(axis=(1, 3))


changed = subprocess.run(["git", "-C", WS, "diff", "--name-only", REF, "HEAD"], capture_output=True, text=True).stdout.split()
clips = [c for c in changed if c.lower().endswith((".mp4", ".mov"))]
say("PROGRESS 3", json.dumps({"clips": clips, "configs": configs}))
rep = {"models": models, "onnx_cache": cache, "configs": configs, "clips": {}}
tmp = os.path.join(OUT, "tmp")
for ci, rel in enumerate(clips):
    shutil.rmtree(tmp, ignore_errors=True)
    src, ev = os.path.join(tmp, "src"), os.path.join(tmp, "even")
    os.makedirs(src); os.makedirs(ev)
    orig = os.path.join(tmp, "orig.mp4")
    with open(orig, "wb") as fh:
        fh.write(subprocess.run(["git", "-C", WS, "show", f"{REF}:{rel}"], capture_output=True).stdout)
    stream([FF, "-v", "error", "-i", orig, "-map", "0:v:0", "-fps_mode", "passthrough", os.path.join(src, "%08d.png")], "extract")
    names = sorted(os.listdir(src))
    for k, n in enumerate(names[::2]):
        shutil.copy2(os.path.join(src, n), os.path.join(ev, f"{k + 1:08d}.png"))
    m = len(names[::2])
    mid = cv2.imread(os.path.join(src, names[len(names) // 2]))
    base = os.path.splitext(os.path.basename(rel))[0][-12:]
    cv2.imwrite(os.path.join(OUT, f"thumb-{base}.jpg"), cv2.resize(mid, (mid.shape[1] // 4, mid.shape[0] // 4)))
    entry = {}
    for cfg in configs:
        model, _, tta = cfg.partition(":")
        out = os.path.join(tmp, "out-" + cfg.replace(":", "_"))
        os.makedirs(out)
        cmd = [RIFE, "-i", ev, "-o", out, "-n", str(2 * m), "-m", os.path.join(RIFE_DIR, model), "-g", "0", "-j", "2:4:4", "-f", "%08d.png"]
        cmd += ["-x"] if "x" in tta else []
        cmd += ["-z"] if "z" in tta else []
        t0 = time.time()
        stream(cmd, cfg)
        outs = sorted(os.listdir(out))
        errs, worst = [], (0, None, None)
        for i in range(1, len(outs), 2):          # output i sits at source position i/2 of the even set = real frame i
            if i >= len(names):
                break
            real = cv2.imread(os.path.join(src, names[i])); syn = cv2.imread(os.path.join(out, outs[i]))
            be = block_err(real, syn)
            errs.append(be)
            if be.max() > worst[0]:
                worst = (float(be.max()), i, np.unravel_index(int(be.argmax()), be.shape))
        E = np.stack(errs)
        # blocks that are bad in many frames = a structural failure (grille), not a one-off
        bad = (E > 12).mean(axis=0)
        entry[cfg] = {"mean": round(float(E.mean()), 2), "p99_block": round(float(np.percentile(E, 99)), 2),
                      "max_block": round(worst[0], 1), "worst_frame": worst[1], "worst_block_rc": [int(x) for x in worst[2]],
                      "blocks_bad_10pct_frames": int((bad > 0.10).sum()), "secs": round(time.time() - t0, 1)}
        i, (r, c) = worst[1], worst[2]
        real = cv2.imread(os.path.join(src, names[i])); syn = cv2.imread(os.path.join(out, outs[i]))
        y0, x0 = max(0, r * 32 - 96), max(0, c * 32 - 96)
        crop = lambda f: cv2.resize(f[y0:y0 + 224, x0:x0 + 224], (448, 448), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, f"worst-{base}-{cfg.replace(':', '_')}.jpg"), np.hstack([crop(real), crop(syn)]))
        heat = cv2.applyColorMap(np.clip(bad * 255 * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (real.shape[1] // 4, real.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, f"heat-{base}-{cfg.replace(':', '_')}.jpg"),
                    cv2.addWeighted(cv2.resize(real, heat.shape[1::-1]), 0.5, heat, 0.5, 0))
        say(json.dumps({rel: {cfg: entry[cfg]}}))
    rep["clips"][rel] = entry
    say(f"PROGRESS {3 + 95 * (ci + 1) // len(clips)}")
shutil.rmtree(tmp, ignore_errors=True)
json.dump(rep, open(os.path.join(OUT, "check.json"), "w"), indent=1)
say("done")
