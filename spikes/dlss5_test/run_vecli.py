"""Spike (2026-10-24): drive Merserk Visual Enhancer v18 (VE_CLI.exe) on the PC from a farm python job.

usage: python run_vecli.py OUT_DIR SPEC.json
SPEC: [{"label": "x", "args": ["render", "{IN:spikes/dlss5_test/in/a.mp4}", "--preset", "{IN:spikes/dlss5_test/p.json}",
        "-o", "{OUT}/x.mp4"]}, ...]   {IN:rel} -> absolute path in the checkout, {OUT} -> OUT_DIR,
        {PREV} -> the previous case's newest output file (for chained passes)
"""
import glob, json, os, subprocess, sys, time

VE = os.environ.get("VE_CLI", r"C:\tools\VisualEnhancer-v18\VE_CLI.exe")
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
spec = json.load(open(sys.argv[2]))
report, prev = [], None


def sub(a):
    if a.startswith("{IN:") and a.endswith("}"):
        return os.path.abspath(a[4:-1])
    return a.replace("{OUT}", out).replace("{PREV}", prev or "")


for i, s in enumerate(spec):
    before = set(glob.glob(os.path.join(out, "**", "*.*"), recursive=True))
    cmd = [VE] + [sub(a) for a in s["args"]]
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=s.get("timeout", 1800), cwd=os.path.dirname(VE))
    new = sorted(set(glob.glob(os.path.join(out, "**", "*.*"), recursive=True)) - before, key=os.path.getmtime)
    vids = [f for f in new if f.lower().endswith((".mp4", ".mkv", ".mov", ".png"))]
    if vids:
        prev = vids[-1]
    rep = {"label": s["label"], "rc": p.returncode, "seconds": round(time.time() - t0, 1), "new_files": new,
           "stdout": p.stdout[-6000:], "stderr": p.stderr[-4000:]}
    report.append(rep)
    print(json.dumps({k: rep[k] for k in ("label", "rc", "seconds", "new_files")}), flush=True)
    print(f"PROGRESS {int(100 * (i + 1) / len(spec))}", flush=True)
json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
