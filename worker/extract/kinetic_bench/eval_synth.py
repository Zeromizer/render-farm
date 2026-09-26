"""Score kinetic.analyse_event against the synthetic ground truth.

  python gen_synth.py [ffmpeg]   # once: writes synth/*.mp4 + truth.json
  python eval_synth.py [case ...]

Baseline 44/46 (the two misses are ease-out durations: the measure reports
the visible move, shorter than the tween's nominal length). Run it with the
extract venv's python (cv2 + numpy + Pillow) after any change to kinetic.py."""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kinetic  # noqa: E402  (worker/extract/kinetic.py)

HERE = os.path.dirname(os.path.abspath(__file__))
truth = json.load(open(f"{HERE}/synth/truth.json"))
only = set(sys.argv[1:])
os.makedirs(f"{HERE}/strips", exist_ok=True)

score = total = 0
for case in truth:
    if only and case["name"] not in only:
        continue
    r = kinetic.analyse_event(f"{HERE}/synth/{case['file']}", case["block"], 30.0, 3.4)
    t = case["truth"]
    e, x = r.get("entry", {}), r.get("exit", {})
    checks = [("entry", e.get("type"), t["entry"])]
    if "entry_from" in t:
        checks.append(("entry_from", e.get("from"), t["entry_from"]))
    checks.append(("exit", x.get("type"), t["exit"]))
    if "exit_to" in t:
        checks.append(("exit_to", x.get("to"), t["exit_to"]))
    if "entry_easing" in t:
        checks.append(("entry_easing", e.get("easing"), t["entry_easing"]))
    if "exit_easing" in t:
        checks.append(("exit_easing", x.get("easing"), t["exit_easing"]))
    if t["entry_dur"] > 0:
        checks.append(("entry_dur", round(e.get("duration_s", -1), 2), f"~{t['entry_dur']}"))
    line = []
    for k, got, want in checks:
        if k.endswith("_dur"):
            ok = abs(got - float(want[1:])) <= max(0.1, 0.35 * float(want[1:]))
        else:
            ok = got == want
        score += ok
        total += 1
        line.append(f"{k}={got}{'' if ok else f' (want {want})'}")
    print(f"{case['name']:<12} " + "  ".join(line))
    if "error" in r:
        print("   ERROR", r["error"])
    if os.environ.get("DEBUG"):
        print("   entry", {k: v for k, v in e.items() if k != "_strip"})
        print("   exit ", {k: v for k, v in x.items() if k != "_strip"})
    s_in, s_out = r.get("_strip", (None, None))
    if s_in is not None:
        cv2.imwrite(f"{HERE}/strips/{case['name']}_in.jpg", s_in)
print(f"\nSCORE {score}/{total}")
