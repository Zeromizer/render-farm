"""Conform a YuE2 instrumental plan (official planner + Vocal->Ins transfer) to a REFERENCE's
clock: its tempo, its length, and the moment its music lifts.

trim_score.py (spikes/yue2_official) cuts a plan to ad length and lets the tempo drift up to 4%.
Here the tempo is the reference's, exactly (rounded to the integer ABC allows), and the first
chorus is placed on the reference's main lift, so a cut timed to the reference's music still
lands on this bed. Everything played is YuE2's own writing; only bar order and tempo change.

Shape:
  run-in   the bars that lead into the plan's first chorus (the last `intro_n` of them, so the
           build into the chorus stays contiguous), intro_n = round(lift_s / bar_s)
  body     from the first chorus onward in plan order (intro/outro skipped, wrapping round)
  ending   the plan's own last `outro_bars` bars (its outro if it has one)
Total = round(seconds / bar_s) whole bars; the generator trims and fades to `seconds`.

Usage: python conform_score.py in.abc out.abc --seconds 24.4 --bpm 103.4 --lift 8.2 --first-beat 0.07
Prints a JSON report (same keys comfy_render reads: sections, bar_seconds, nominal_seconds).
"""
import argparse
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "yue2_official"))
from trim_score import CHORD_ANY, CHORD_AT_START, full_rest_units, header_value, parse_plan  # noqa: E402
from abc_tools import parse_abc, report  # noqa: E402  (trim_score put skill/scripts on the path)
from compile_score import compress  # noqa: E402


def conform(text, seconds, bpm, lift_s=None, outro_bars=2, first_beat=None):
    header, sections = parse_plan(text)
    q_plan = int(header_value(header, "Q:").split("=")[1])
    n, d = (int(x) for x in header_value(header, "M:").split("/"))
    quarters_per_bar = Fraction(4 * n, d)
    q = int(round(bpm))
    bar_s = float(quarters_per_bar) * 60 / q

    flat, chord = [], None
    for si, (label, bars) in enumerate(sections):
        for v, i in bars:
            start_chord = chord
            found = CHORD_ANY.findall(v)
            if found:
                chord = found[-1]
            flat.append({"label": label, "v": v, "i": i, "chord_in": start_chord})
    if not flat:
        raise ValueError("empty plan")
    # Head trim x (0 <= x < one bar, cut off the rendered audio): puts the bed's beats on the
    # reference's beat phase (first_beat) and, with the run-in length, its first chorus as close
    # to the reference's lift as whole beats allow. YuE2 opens quiet, so losing the head helps.
    beat_s = 60.0 / bpm
    beats_per_bar = max(1, int(round(bar_s / beat_s)))
    phases = [((j * beat_s - (first_beat or 0.0)) % bar_s) for j in range(beats_per_bar)] if first_beat is not None else [0.0]
    choices = []
    for x in phases:
        for k in range(0, 64):
            err = abs(k * bar_s - x - lift_s) if lift_s is not None else abs(k - 2)
            choices.append((round(err, 3), x, k))
    _, head_trim, intro_guess = min(choices)
    total = max(6, int(-(-(seconds + head_trim) // bar_s)))
    labels = [s[0] for s in sections]
    if "outro" in labels:
        ending = [b for b in flat if b["label"] == "outro"][-outro_bars:]
    else:
        ending = flat[-outro_bars:]
    ending_ids = {id(b) for b in ending}
    first_chorus = next((k for k, b in enumerate(flat) if b["label"] == "chorus"), None)
    if first_chorus is None:   # no chorus: the body starts after the intro
        first_chorus = next((k for k, b in enumerate(flat) if b["label"] != "intro"), 0)
    lead = [b for b in flat[:first_chorus] if id(b) not in ending_ids]
    body_pool = [b for b in flat[first_chorus:] if b["label"] not in ("intro", "outro") and id(b) not in ending_ids]
    if not body_pool:
        body_pool = [b for b in flat if id(b) not in ending_ids] or flat
    intro_n = intro_guess
    intro_n = max(0, min(intro_n, total - len(ending) - 2))
    if intro_n and not lead:
        lead = [b for b in body_pool if b["label"] != "chorus"] or body_pool
    # the LAST intro_n lead-in bars; a short lead-in is cycled from its start, keeping its tail last
    run_in = [lead[(len(lead) - intro_n + k) % len(lead)] for k in range(intro_n)] if intro_n else []
    body_n = total - intro_n - len(ending)
    body = [body_pool[k % len(body_pool)] for k in range(body_n)]

    picked = [("intro" if b["label"] in ("intro", "outro") else b["label"], b) for b in run_in] + \
             [(b["label"], b) for b in body] + [("outro", b) for b in ending]
    full = full_rest_units(header)
    out_sections = []
    for k, (label, b) in enumerate(picked):
        v, i = b["v"], b["i"]
        nxt = picked[k + 1][1] if k + 1 < len(picked) else None
        if i.endswith("-") and not (nxt is not None and flat.index(nxt) == flat.index(b) + 1):
            i = i[:-1]
        prev = picked[k - 1][1] if k else None
        if not (prev is not None and flat.index(b) == flat.index(prev) + 1) and not CHORD_AT_START.match(v) and b["chord_in"]:
            v = f'"{b["chord_in"]}"z{full}' if v == "Z" else f'"{b["chord_in"]}"' + v
        if not out_sections or out_sections[-1][0] != label:
            out_sections.append([label, []])
        out_sections[-1][1].append((v, i))

    lines = [l if not l.startswith("Q:") else f"Q:1/4={q}" for l in header]
    for label, bars in out_sections:
        lines.append(f"% {label}")
        for g in range(0, len(bars), 4):
            group = bars[g:g + 4]
            lines += ["V: Vocal", compress([v for v, _ in group]), "V: Ins", compress([i for _, i in group])]
    result = "\n".join(lines) + "\n"
    rep = report(parse_abc(result))
    key = next((l[2:].strip() for l in header if l.startswith("K:")), None)
    info = {"plan_bars": len(flat), "plan_sections": labels, "plan_tempo": q_plan, "tempo": q, "plan_key": key,
            "bars": len(picked), "run_in_bars": intro_n, "head_trim_s": round(head_trim, 3),
            "chorus_at_s": round(intro_n * bar_s - head_trim, 2),
            "lift_s": lift_s, "nominal_seconds": round(rep["nominal_duration_seconds"], 2),
            "bar_seconds": bar_s, "sections": [[l, len(b)] for l, b in out_sections]}
    return result, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--lift", type=float, default=None)
    ap.add_argument("--outro-bars", type=int, default=2)
    ap.add_argument("--first-beat", type=float, default=None)
    a = ap.parse_args()
    with open(a.src, encoding="utf-8") as f:
        text = f.read()
    result, info = conform(text, a.seconds, a.bpm, a.lift, a.outro_bars, a.first_beat)
    from instrumental import validate_score  # the skill's own gate
    validate_score(result)
    with open(a.dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(result)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
