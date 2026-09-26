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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kinetic  # noqa: E402  (worker/extract/kinetic.py)

HERE = os.path.dirname(os.path.abspath(__file__))
truth = json.load(open(f"{HERE}/synth/truth.json"))
only = set(sys.argv[1:])
os.makedirs(f"{HERE}/strips", exist_ok=True)

score = total = 0

# the true tween of each move whose curve is worth checking: (side, progress fn)
lin = lambda p: p
out3 = lambda p: 1 - (1 - p) ** 3
in3 = lambda p: p ** 3


def back_out(p, k=2.2):
    p -= 1
    return p * p * ((k + 1) * p + k) + 1


CURVES = {"slide_left": [("entry", out3), ("exit", lin)], "rise": [("entry", out3), ("exit", lin)],
          "fade": [("entry", lin), ("exit", lin)], "typewriter": [("entry", lin)],
          "wipe_up": [("entry", out3), ("exit", in3)], "pop": [("entry", back_out), ("exit", in3)]}
curve_ok = curve_n = 0


def curve_error(bez, fn, t_nom0, dur, m_t0, m_t1):
    """Max gap between the fitted bezier and the true tween, both over the
    MEASURED window: the fit is normalised to it, and a truncated ease-out
    tail is invisible, so the shape an ideal fit sees is the true tween
    re-normalised to that window."""
    tn = np.linspace(0.05, 0.95, 19)
    y_fit = kinetic._bezier_y(np.array([bez], dtype=float), tn)[0]
    tt = m_t0 + tn * (m_t1 - m_t0)
    f = lambda t: fn(min(1.0, max(0.0, (t - t_nom0) / dur)))
    a, b = f(m_t0), f(m_t1)
    y_true = np.array([(f(t) - a) / (b - a) if b != a else 0 for t in tt])
    return float(np.max(np.abs(y_fit - y_true)))
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
    for side, fn in CURVES.get(case["name"], []):
        m = e if side == "entry" else x
        t_nom0 = t["entry_start"] if side == "entry" else t["exit_start"]
        dur = t["entry_dur"] if side == "entry" else t["exit_dur"]
        curve_n += 1
        if "bezier" not in m:
            print(f"   curve {side}: none fitted")
            continue
        err = curve_error(m["bezier"], fn, t_nom0, dur, m["t_start"], m["t_end"])
        curve_ok += err <= 0.15
        print(f"   curve {side}: {m['bezier']} rmse {m['curve_rmse']} max-gap-to-truth {err:.2f}{'' if err <= 0.15 else '  (MISS)'}")
    s_in, s_out = r.get("_strip", (None, None))
    if s_in is not None:
        cv2.imwrite(f"{HERE}/strips/{case['name']}_in.jpg", s_in)
print(f"\nSCORE {score}/{total}   CURVES {curve_ok}/{curve_n}")
