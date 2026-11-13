"""Cut a YuE2 instrumental score (after the official skill's Vocal -> Ins transfer) to ad length.

The official recipe lets YuE2 plan the whole piece; for energetic styles that is 3-5 minutes,
which is too long for an ad and runs a 16 GB card out of memory. YuE2 plays a fixed score to
its end (audio ~95-98% of the score), so cutting the SCORE is how length is controlled while
keeping a real, played ending. Everything kept is YuE2's own writing; nothing is composed here.

Shape, for a short ad:
  intro    the plan's first `intro_bars` bars (0 = start on the hook)
  body     from the plan's first chorus, then onward in the plan's order (intro/outro skipped,
           wrapping round if the plan is short), cut on a phrase boundary where possible
  outro    the plan's own last `outro_bars` bars: its outro section if it has one, else the
           bars it ended on
The tempo (Q) is then moved by at most `max_tempo_change` (4%) so the whole bars land on target_s;
if that is not enough the score is left at the nearest whole-bar length.

Usage: python trim_score.py in.abc out.abc --seconds 30 [--intro-bars 2] [--outro-bars 2]
Prints a JSON report. Checks the result with the skill's own validate_score.
"""
import argparse
import json
import os
import re
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "skill", "scripts"))
from abc_tools import parse_abc, report  # noqa: E402
from compile_score import compress  # noqa: E402

CHORD_AT_START = re.compile(r'^"[^"]*"')
CHORD_ANY = re.compile(r'"([^"]*)"')


def split_bars(line):
    """'Z2|"Dm"z32|' -> ['Z', 'Z', '"Dm"z32'] (multi-bar rests expanded to one Z per bar)."""
    out = []
    for token in line.rstrip().rstrip("|").split("|"):
        m = re.fullmatch(r"Z([0-9]*)", token)
        if m:
            out += ["Z"] * int(m.group(1) or 1)
        else:
            out.append(token)
    return out


def parse_plan(text):
    """Header lines, and sections as [label, [(vocal_bar, ins_bar), ...]] in order."""
    lines = text.splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith("% "))
    header, sections, voice, pending = lines[:first], [], None, {}
    for line in lines[first:]:
        if line.startswith("% "):
            sections.append([line[2:].strip(), []])
        elif line.startswith("V: "):
            voice = line[3:].strip()
        elif line.startswith(("M:", "K:")):
            raise ValueError("mid-score meter/key change: not handled by this cutter")
        elif line.strip():
            pending[voice] = split_bars(line)
            if "Vocal" in pending and "Ins" in pending:
                v, i = pending.pop("Vocal"), pending.pop("Ins")
                if len(v) != len(i):
                    raise ValueError("voices disagree on bar count in a group")
                sections[-1][1].extend(zip(v, i))
    return header, sections


def header_value(header, key):
    return next(l for l in header if l.startswith(key))[len(key):]


def full_rest_units(header):
    n, d = (int(x) for x in header_value(header, "M:").split("/"))
    unit = int(header_value(header, "L:").split("/")[1])
    return n * unit // d


def cut(text, seconds=30.0, intro_bars=2, outro_bars=2, max_tempo_change=0.04):
    header, sections = parse_plan(text)
    q = int(header_value(header, "Q:").split("=")[1])
    n, d = (int(x) for x in header_value(header, "M:").split("/"))
    quarters_per_bar = Fraction(4 * n, d)
    bar_s = float(quarters_per_bar) * 60 / q

    # Every bar with its section and the chord sounding at its start (for splices).
    flat, chord = [], None
    for si, (label, bars) in enumerate(sections):
        for bi, (v, i) in enumerate(bars):
            start_chord = chord
            found = CHORD_ANY.findall(v)
            if found:
                chord = found[-1]
            flat.append({"si": si, "label": label, "v": v, "i": i, "chord_in": start_chord,
                         "ties_out": i.endswith("-"), "last_in_section": bi == len(bars) - 1})
    labels = [s[0] for s in sections]
    intro = [b for b in flat if b["label"] == "intro"][:intro_bars]
    if "outro" in labels:
        ending = [b for b in flat if b["label"] == "outro"][-outro_bars:] if outro_bars else []
    else:
        ending = flat[-outro_bars:] if outro_bars else []
    ending_ids = {id(b) for b in ending}
    pool = [b for b in flat if b["label"] not in ("intro", "outro") and id(b) not in ending_ids]
    if not pool:
        raise ValueError("plan has no body bars")
    start = next((k for k, b in enumerate(pool) if b["label"] == "chorus"), 0)
    order = pool[start:] + pool[:start]

    # Pick the body length. In order of importance: keep the tempo within max_tempo_change of
    # the plan (a beat-synced edit is cut to it), end the body on a 4-bar phrase, land exactly
    # on seconds. If no tempo nudge fits exactly, keep the plan's tempo and accept up to
    # `slack_s` short (the played ending then finishes a little early).
    slack_s = 3.0
    best = None
    for body_n in range(1, 4 * int(seconds / bar_s) + 8):
        total = len(intro) + body_n + len(ending)
        exact_q = total * float(quarters_per_bar) * 60 / seconds
        phrase_penalty = 0 if body_n % 4 == 0 else 0.02
        if abs(exact_q - q) / q <= max_tempo_change:
            cost, new_q = abs(exact_q - q) / q + phrase_penalty, round(exact_q)
        else:
            length = total * float(quarters_per_bar) * 60 / q
            if not seconds - slack_s <= length <= seconds:
                continue
            cost, new_q = max_tempo_change + (seconds - length) / seconds + phrase_penalty, q
        if best is None or cost < best[0]:
            best = (cost, body_n, new_q)
    if best is None:
        raise ValueError("no body length fits the target")
    _, body_n, new_q = best
    body = [order[k % len(order)] for k in range(body_n)]

    # Assemble with splice repairs: drop a tie that leads out of a kept bar into a bar that is
    # not the next one; give a spliced-in bar the chord that was sounding at that point.
    picked = [("intro", b) for b in intro] + [(b["label"], b) for b in body] + [("outro", b) for b in ending]
    full = full_rest_units(header)
    out_sections = []
    for k, (label, b) in enumerate(picked):
        v, i = b["v"], b["i"]
        nxt = picked[k + 1][1] if k + 1 < len(picked) else None
        contiguous_next = nxt is not None and flat.index(nxt) == flat.index(b) + 1
        if i.endswith("-") and not contiguous_next:
            i = i[:-1]
        prev = picked[k - 1][1] if k else None
        contiguous_prev = prev is not None and flat.index(b) == flat.index(prev) + 1
        if not contiguous_prev and not CHORD_AT_START.match(v) and b["chord_in"]:
            v = f'"{b["chord_in"]}"z{full}' if v == "Z" else f'"{b["chord_in"]}"' + v
        if not out_sections or out_sections[-1][0] != label:
            out_sections.append([label, []])
        out_sections[-1][1].append((v, i))

    lines = [l if not l.startswith("Q:") else f"Q:1/4={new_q}" for l in header]
    for label, bars in out_sections:
        lines.append(f"% {label}")
        for g in range(0, len(bars), 4):
            group = bars[g:g + 4]
            lines += ["V: Vocal", compress([v for v, _ in group]), "V: Ins", compress([i for _, i in group])]
    result = "\n".join(lines) + "\n"
    score = parse_abc(result)
    rep = report(score)
    info = {"plan_bars": len(flat), "plan_seconds": round(len(flat) * bar_s, 1), "plan_sections": labels,
            "kept": [f"{l}:{len(b)}" for l, b in out_sections], "bars": len(picked),
            "tempo": [q, new_q], "nominal_seconds": round(rep["nominal_duration_seconds"], 2)}
    return result, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--intro-bars", type=int, default=2)
    ap.add_argument("--outro-bars", type=int, default=2)
    ap.add_argument("--max-tempo-change", type=float, default=0.04)
    a = ap.parse_args()
    with open(a.src, encoding="utf-8") as f:
        text = f.read()
    result, info = cut(text, a.seconds, a.intro_bars, a.outro_bars, a.max_tempo_change)
    from instrumental import validate_score  # the skill's own gate
    validate_score(result)
    with open(a.dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(result)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
