"""Free steps of a session (no keys, no GPU), all writing under <d>/case:
    ingest       source span -> case/{cond.mp4 (draft), cond720.mp4 (final), audio.m4a, nf, cuts.json, meta.json} + empty
                 plan/case/where files
    strips       one row of 4 frames per detector shot -> case/look/strips_<p>.jpg
    frames_cmd   exact frames (up to 12), 2 per row at 960 wide, from case/cond720.mp4
    check        validates plan_grid.json / case.json / where.json, writes paint/src + paint/inv.json, lists the shots
                 with people to replace that have no keyframe yet
    windows_cmd  render windows ending on cuts + their keyframes -> case/refs.json; a shot longer than one window is
                 continued by hops (see windows())
    strip        review strip of a window: src row + one row per current take -> case/review/
    finish       kept take per window -> case/final/full_<canvas>.mp4, then 1-2 frame cut flashes at window boundaries
                 are fixed (deflash)
    peek         contact sheet + cut times of any source span (before ingest)
"""
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ARM, FPS, LENS, P, find_cuts, find_strobes, frames, grid, jr, jw, label, pick, probe, seg, ticks


def windows(cuts, n, maxlen, maxcuts):
    """Windows end on cuts, at most `maxlen` frames and, where the cuts allow, at most `maxcuts` cuts inside the render
    (the model drops and reorders short shots in a window with many cuts); `render` is 17k+5 long and may run past `own`
    (dropped frames). A shot longer than one window is not split into independent windows (the seam jumps the face)
    but continued: the next window is a hop, "prefix_of" the window before it, whose take's frames [prefix_at,
    prefix_at + P) are this render's first P frames (render[0] = own[0] - P). The last window is aligned to the clip
    end; when it is a hop the window before it gives up the frames it needs."""
    ls = [x for x in LENS if x <= maxlen]
    rnd = lambda m: min(x for x in ls if x >= m)  # noqa: E731
    out, s, p = [], 0, 0
    while True:
        r = s - p
        # a render that would run past the clip end is no candidate (the cond would come out shorter than num_frames)
        cand = [e for e in cuts if r + ls[0] <= e <= r + ls[-1] and s < e < n and r + rnd(e - r) <= n]
        if n - r <= ls[-1] and (not cand or sum(s < c < n for c in cuts) <= maxcuts):
            break
        ok = [e for e in cand if sum(r < c < r + rnd(e - r) for c in cuts) <= maxcuts]
        e = max(ok) if ok else min(cand) if cand else r + ls[-1]
        out.append({"render": [r, r + rnd(e - r)], "own": [s, e],
                    **({"prefix_of": len(out) - 1, "prefix_at": r - out[-1]["render"][0]} if p else {})})
        s, p = e, 0 if e in cuts else P
    ln = rnd(max(n - s + p, ls[0]))
    if p:  # the hop starts P frames into its render: own[0] moves back to n - ln + P, out of the window before it
        s = out[-1]["own"][1] = n - ln + P
        out.append({"render": [n - ln, n], "own": [s, n], "prefix_of": len(out) - 1, "prefix_at": n - ln - out[-1]["render"][0]})
    else:
        out.append({"render": [n - ln, n], "own": [s, n]})
    return out


def nframes(d):
    return int((d / "case" / "nf").read_text())


AMAX = 960 / 416  # the widest aspect rendered (2.31:1); a wider source is squeezed to it
AREA = {"480": 832 * 480, "720": 1280 * 736}
X264 = ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]


def fit(a, area):
    """The canvas for aspect a = w/h: multiples of 32, area in [0.8, 1] x `area`, the largest within 3% of a (16:9 gives
    832x480 / 1280x736), else the closest. Frames are stretched onto it, never cropped; finish stretches them back."""
    c = [(abs(math.log(W / H / a)), W * H, W, H) for H in range(32, 2049, 32)
         for W in {32 * math.floor(H * a / 32), 32 * math.ceil(H * a / 32)} if W and 0.8 * area <= W * H <= area]
    ok = [x for x in c if x[0] <= 0.03]
    return list(max(ok, key=lambda x: (x[1], -x[0]))[2:] if ok else min(c)[2:])


def size(d, canvas):
    m = jr(d, "meta.json")
    return m["draft"] if str(canvas) == "480" else m["final"]


def out_size(d, canvas):
    """The final's size: the canvas's long edge at the source's aspect (even, for yuv420p)."""
    (W, H), a = size(d, canvas), jr(d, "meta.json")["aspect"]
    return (W, 2 * round(W / a / 2)) if W >= H else (2 * round(H * a / 2), H)


def cond(d, canvas):
    return d / "case" / ("cond.mp4" if str(canvas) == "480" else "cond720.mp4")


def shot_bounds(d):
    b = [0] + jr(d, "cuts.json") + [nframes(d)]
    return list(zip(b, b[1:]))


def ingest(d, src, start=0.0, dur=None, crop=None, force=False):
    C = d / "case"
    if (C / "meta.json").exists():
        assert force, ("case/ already holds an ingested clip. Frame numbers change with the span, so force=true discards "
                       "the plan, where.json, paints, prompts and takes (case.json and the character sheets are kept).")
        for p in C.iterdir():
            if p.name not in ("chars", "case.json"):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    src = Path(src).resolve()
    w, h, length, has_audio = probe(src)
    if crop:
        w, h = (int(x) for x in crop.split(":")[:2])
    a = min(max(w / h, 1 / AMAX), AMAX)
    draft, final = fit(a, AREA["480"]), fit(a, AREA["720"])
    want = int(min(dur or 1e9, length - start) * FPS)
    C.mkdir(exist_ok=True)
    cut = ["-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-an", "-vf"]
    vf = lambda W, H: (f"crop={crop}," if crop else "") + f"fps={FPS},scale={W}:{H},setsar=1"  # noqa: E731
    for name, (W, H) in (("cond.mp4", draft), ("cond720.mp4", final)):
        subprocess.run(["ffmpeg", "-y", *cut, vf(W, H), "-frames:v", str(want), *X264, str(C / name)], check=True)
    v = frames(C / "cond.mp4", *draft)
    real, cuts, strobes = len(v), find_cuts(v), find_strobes(v)
    assert real >= LENS[0], f"only {real} frames: the renderer needs >= {LENS[0]} ({LENS[0] / FPS:.2f} s)"
    # a clip of one render must be 17k+5 long: the last shot plays on backwards and forwards up to the next one, and
    # finish cuts that off again
    nf = real if real > LENS[-1] else min(x for x in LENS if x >= real)
    if nf > real:
        lo = cuts[-1] if cuts else 0
        back = list(range(real - 2, lo - 1, -1)) + list(range(lo + 1, real)) or [real - 1]
        idx = list(range(real)) + [back[k % len(back)] for k in range(nf - real)]
        for name, (W, H) in (("cond.mp4", draft), ("cond720.mp4", final)):
            raw = subprocess.run(["ffmpeg", *cut, vf(W, H), "-frames:v", str(real), "-f", "rawvideo", "-pix_fmt", "rgb24",
                                  "-"], capture_output=True, check=True).stdout
            x = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
            assert len(x) == real, (name, len(x), real)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                            str(FPS), "-i", "-", *X264, str(C / name)], input=x[idx].tobytes(), check=True)
    # the soundtrack of the real frames, then silence over the padding
    aud = ["-ss", f"{start:.3f}", "-t", f"{real / FPS:.3f}", "-i", str(src)] if has_audio else [
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *aud, "-af", "apad", "-t", f"{nf / FPS:.3f}", "-vn", "-c:a", "aac",
                    "-b:a", "192k", str(C / "audio.m4a")], check=True)
    # the agent writes the soundscape from this line: a muted stream reads as music otherwise, and one run (ingested before
    # its audio was fixed) said "silent" in all 35 prompts while the pin carried the live vocals
    vd = subprocess.run(["ffmpeg", "-v", "info", "-t", f"{real / FPS:.3f}", "-i", str(C / "audio.m4a"), "-af", "volumedetect",
                         "-f", "null", "-"], capture_output=True, text=True).stderr
    db = float(vd.split("mean_volume:")[1].split()[0]) if "mean_volume:" in vd else -91.0
    has_audio = has_audio and db > -60
    (C / "nf").write_text(str(nf))
    jw(d, "cuts.json", cuts)
    jw(d, "meta.json", {"src": str(src), "start": start, "crop": crop, "draft": draft, "final": final, "aspect": w / h,
                        "frames": real, "audio": has_audio, "strobes": strobes})
    jw(d, "asr.json", {"skipped": True, "segments": []})
    if not (C / "plan_grid.json").exists():
        jw(d, "plan_grid.json", {"people": [], "shots": [{"start": s, "end": e, "people": [], "desc": ""} for s, e in shot_bounds(d)]})
    if not (C / "case.json").exists():
        jw(d, "case.json", {"chars": {}, "outfits": {}})
    if not (C / "where.json").exists():
        jw(d, "where.json", {})
    st = final[0] / final[1] / (w / h) - 1
    return (f"{real} frames ({real / FPS:.1f} s) at {draft[0]}x{draft[1]} (final {final[0]}x{final[1]}), "
            f"audio {f'yes (mean {db:.0f} dB)' if has_audio else 'none (silent track)'}, {len(cuts) + 1} detector shots:"
            + (f"\n(the {w}x{h} source is stretched {st:+.1%} in width onto the canvas"
               + (f", being wider than the {AMAX:.2f}:1 the model takes" if a != w / h else "")
               + "; finish stretches it back)" if abs(st) > 0.005 else "")
            + (f"\n(frames {real}-{nf - 1} are padding - the last shot played backwards - to a renderable 17k+5 length; "
               "finish cuts them off: put no keyframe there)" if nf > real else "")
            + "\n" + " ".join(f"S{k}[{s},{e})" for k, (s, e) in enumerate(shot_bounds(d)))
            + (f"\n{strobe_note(d)}" if strobes else ""))


def strobe_note(d):
    """One run painted a frame as the close-up its where.json said and stopped on "the frame tool returns inconsistent
    pictures": two frames 11 apart of one detector shot are different pictures."""
    s = jr(d, "meta.json").get("strobes", [])
    return (f"strobe {', '.join(f'f{a}-{b - 1}' for a, b in s)}: the frames alternate between two pictures every frame "
            f"(f{s[0][0]} is one, f{s[0][0] + 1} the other), so frames of that shot show different things. List the people "
            "of both pictures in the shot and write each where.json entry from its own frame. Renders keep the "
            "alternation with a few frames off the beat (7-9 of 31 in four runs): MINOR, a reseed does not fix it"
            ) if s else ""


def strips(d):
    """-> (text, [image paths])"""
    W, H = size(d, 480)
    cw, ch = (352, round(352 * H / W)) if W >= H else (round(352 * W / H), 352)
    v = frames(d / "case" / "cond.mp4", cw, ch)
    (d / "case" / "look").mkdir(exist_ok=True)
    rows, out, txt = [], [], []
    for k, (s, e) in enumerate(shot_bounds(d)):
        fs = [s + (e - s) * (2 * j + 1) // 8 for j in range(4)]
        rows.append(grid([Image.fromarray(v[f]) for f in fs], 4, [f"S{k} f{f}" + (f" [{s},{e})" if j == 0 else "")
                                                                for j, f in enumerate(fs)]))
    for p in range(0, len(rows), 10):
        out.append(d / "case" / "look" / f"strips_{p // 10}.jpg")
        grid(rows[p:p + 10], 1).save(out[-1], quality=82)
        txt.append(f"case/look/{out[-1].name} shots S{p}..S{min(p + 10, len(rows)) - 1}")
    return "\n".join(txt), out


def frames_cmd(d, fs):
    """-> (text, image path)"""
    nf = nframes(d)
    bad = [f for f in fs if not 0 <= f < nf]
    assert not bad, f"frames {bad} are outside the clip [0, {nf})"
    txt = f"only the first 12 are drawn; call again for {fs[12:]}\n" if len(fs) > 12 else ""
    fs = fs[:12]
    W, H = size(d, 720)
    cw, ch = (960, round(960 * H / W)) if W >= H else (round(960 * W / H), 960)
    v = pick(d / "case" / "cond720.mp4", fs, cw, ch)  # the same frame numbers as everything else in the case
    (d / "case" / "look").mkdir(exist_ok=True)
    out = d / "case" / "look" / f"frames_{'_'.join(map(str, fs))}.jpg"
    grid([label(ticks(Image.fromarray(v[f])), f"f{f}") for f in fs], 2).save(out, quality=86)
    return txt + f"case/look/{out.name}", out


def shot_of(plan, f):
    return next(s for s in plan["shots"] if s["start"] <= f < s["end"])


def check(d):
    """-> (errors, report lines, plan, where, people). With errors nothing is written."""
    C = d / "case"
    plan, case, where = jr(d, "plan_grid.json"), jr(d, "case.json"), jr(d, "where.json", {})
    people = {p["id"]: p for p in plan["people"]}
    nf, err, warn = nframes(d), [], []
    for c in {*case["chars"], *case.get("outfits", {})}:
        if not re.fullmatch(r"[\w \-]+", c):
            err.append(f"character name {c!r}: letters, digits, space, - and _ only (it becomes a file name)")
    for p in plan["people"]:
        if "replace_with" not in p:
            err.append(f"person {p['id']}: no replace_with (a character name, or \"none\" for people who stay)")
        r = p.get("replace_with", "none")
        if r != "none" and r not in case["chars"]:
            err.append(f"person {p['id']} -> {r}: no case.json chars[{r!r}]")
        if r != "none" and r not in case.get("outfits", {}):
            err.append(f"character {r}: no case.json outfits[{r!r}]")
    sh = sorted(plan["shots"], key=lambda s: s["start"])
    if [s["start"] for s in sh] != [0] + [s["end"] for s in sh[:-1]] or sh[-1]["end"] != nf:
        err.append(f"plan_grid.json shots must tile [0, {nf}) without gaps or overlaps")
    for s in sh:
        for i in s["people"]:
            if i not in people:
                err.append(f"shot [{s['start']},{s['end']}) lists unknown person {i!r}")
    for f, wf in where.items():
        f = int(f)
        if not 0 <= f < nf:
            err.append(f"where.json frame {f} outside the clip")
            continue
        if f >= jr(d, "meta.json")["frames"]:
            err.append(f"where.json frame {f} is in the padding after the real clip (finish cuts it off): pick an "
                       "earlier frame")
            continue
        for i in wf:
            if not i.startswith("_") and i not in people:
                err.append(f"where.json f{f}: unknown person {i!r}")
            elif not i.startswith("_") and i not in shot_of(plan, f)["people"]:
                warn.append(f"where.json f{f}: {i} is not in that shot's people in plan_grid.json")
    for i in people:  # the painter puts in everyone listed, where the text says: one text copied to every keyframe says
        # neither (one run copied "pink-clad lead, wherever visible" to 63 keyframes: one got a whole person where the source shows a hat brim)
        same = {}
        for f, wf in where.items():
            if i in wf:
                same.setdefault(wf[i], []).append(f)
        for t, fs in same.items():
            if len(fs) > 3:
                err.append(f"where.json {i}: one text in {len(fs)} keyframes (f{' f'.join(fs[:4])} ...: {t[:60]!r}). "
                           f"Write it for each frame from that frame - where {i} is and how much shows (\"left edge, "
                           f"only the hat brim\", \"centre, full figure\") - and leave {i} out of a frame they are not in")
    if err:
        return err, [], plan, where, people
    src = C / "paint" / "src"
    src.mkdir(parents=True, exist_ok=True)
    (C / "paint" / ARM).mkdir(exist_ok=True)
    W, H = size(d, 720)  # keyframes are painted and kept at the final canvas: a draft copy, upscaled, is soft at 720
    need = [int(f) for f in where if not (src / f"f{f}.png").exists() or Image.open(src / f"f{f}.png").size != (W, H)]
    if need:
        v = pick(C / "cond720.mp4", need, W, H)
        for f in need:
            Image.fromarray(v[f]).save(src / f"f{f}.png")
    inv = {}  # the prompt writer's per-frame inventory: who is visible in each keyframe
    for f, wf in where.items():
        ids = shot_of(plan, int(f))["people"]
        inv[f] = {"people": [{"id": i, "desc": people[i]["desc"]} for i in ids],
                  "inv": {i: {"visible": i in wf, "face_clear": i in wf, "where": wf.get(i, ""), "votes": 1} for i in ids}}
    jw(d, "paint/inv.json", inv)
    swap = {i for i, p in people.items() if p.get("replace_with", "none") != "none"}
    kf = {int(f) for f in where}
    bare = [s for s in sh if set(s["people"]) & swap and not any(s["start"] <= f < s["end"] for f in kf)]
    unpainted = sorted(f for f in kf if not (C / "paint" / ARM / f"f{f}.png").exists())
    lines = [f"warning: {w}" for w in warn]
    lines.append(f"ok: {len(people)} people ({len(swap)} replaced), {len(sh)} plan shots, {len(kf)} keyframes in where.json, "
                 f"{len(kf) - len(unpainted)} painted")
    if strobe_note(d):
        lines.append(strobe_note(d))
    if bare:
        lines.append("shots with people to replace and NO keyframe (pick one frame each, add it to where.json):\n  "
                     + " ".join(f"[{s['start']},{s['end']})" for s in bare))
    if unpainted:
        lines.append(f"keyframes not painted yet: {unpainted}")
    return [], lines, plan, where, people


def report(d):
    """check() as text: its errors, or its report."""
    err, lines, *_ = check(d)
    return "ERRORS:\n  " + "\n  ".join(err) if err else "\n".join(lines)


def pick_refs(plan, where, people, w, painted):
    """Keyframes of one window: one painted frame per shot in `own` (the one showing most people to replace), then any
    replaced character still unshown. The renderer takes 4: over that, first a frame for every character, then the
    longest shots. A short shot without its own keyframe is where a wrong identity or a missed replacement shows up,
    so a window that needs more than 4 should be made shorter."""
    swap = {i for i, p in people.items() if p.get("replace_with", "none") != "none"}
    rep = lambda f: {people[i]["replace_with"] for i in where[str(f)] if i in swap}  # noqa: E731
    a, b = w["own"]
    shots = [s for s in plan["shots"] if s["start"] < b and s["end"] > a]
    need = {people[i]["replace_with"] for s in shots for i in s["people"] if i in swap}
    own = [f for f in painted if a <= f < b]
    pick = [max(fs, key=lambda f: (len(rep(f)), -f)) for s in shots
            if (fs := [f for f in own if max(s["start"], a) <= f < min(s["end"], b)])]
    left = need - {c for f in pick for c in rep(f)}
    while left and (cand := [f for f in own if rep(f) & left and f not in pick]):
        f = max(cand, key=lambda f: len(rep(f) & left))
        pick.append(f)
        left -= rep(f)
    if len(pick) > 4:
        ln = lambda f: (lambda s: s["end"] - s["start"])(shot_of(plan, f))  # noqa: E731
        keep, left = [], set(need)
        while left and (cand := [f for f in pick if rep(f) & left]):
            f = max(cand, key=lambda f: (len(rep(f) & left), ln(f)))
            keep.append(f)
            left -= rep(f)
        pick = (keep + sorted((f for f in pick if f not in keep), key=lambda f: -ln(f)))[:4]
    return sorted(pick), need


def windows_cmd(d, maxlen, maxcuts):
    err, _, plan, where, people = check(d)
    assert not err, "ERRORS:\n  " + "\n  ".join(err)
    nf = nframes(d)
    kf = sorted(int(f) for f in where)
    painted = [f for f in kf if (d / "case" / "paint" / ARM / f"f{f}.png").exists()]
    cuts = jr(d, "cuts.json")
    ws = windows(cuts, nf, maxlen, maxcuts)
    assert all(w["render"][0] >= 0 for w in ws), ws
    swapped = lambda s: any(people[p].get("replace_with", "none") != "none" for p in s["people"])  # noqa: E731
    out = []
    for i, w in enumerate(ws):
        refs, need = pick_refs(plan, where, people, w, painted)
        a0, b0 = w["own"]
        if "prefix_of" in w and ws[w["prefix_of"]]["passthrough"]:  # nothing rendered to continue: an ordinary window
            del w["prefix_of"], w["prefix_at"]
        hop = "prefix_of" in w
        w.update(refs=refs, why="kit windows", reflect=False, passthrough=not need)
        has = lambda s, fs: any(max(s["start"], a0) <= f < min(s["end"], b0) for f in fs)  # noqa: E731
        # a hop's first shot carries on from the prefix and needs no keyframe of its own (only for a face flip or a
        # new person)
        sw = [s for s in plan["shots"] if s["start"] < b0 and s["end"] > a0 and swapped(s) and not (hop and s["start"] < a0)]
        span = lambda ss: " ".join(f"[{max(s['start'], a0)},{min(s['end'], b0)})" for s in ss)  # noqa: E731
        nokf = [s for s in sw if not has(s, kf)]
        todo = [s for s in sw if has(s, kf) and not has(s, painted)]
        drop = [s for s in sw if has(s, painted) and not has(s, refs)]
        r0, r1 = w["render"]
        out.append(f"w{i:2d} own [{a0},{b0}) render [{r0},{r1}) {r1 - r0:3d} fr, {sum(r0 < c < r1 for c in cuts)} cuts, "
                   + (f"hop of w{w['prefix_of']}, " if hop else "")
                   + ("PASSTHROUGH (nobody to replace)" if not need else f"refs {refs} chars {sorted(need)}")
                   + (f"\n      !! no keyframe in where.json for {span(nokf)}" if nokf else "")
                   + (f"\n      keyframe chosen but not painted yet for {span(todo)}" if todo else "")
                   + (f"\n      !! {len(sw)} shots with people to replace but the renderer takes 4 keyframes: {span(drop)} "
                      f"left without one. Lower maxlen / maxcuts, or accept a weak window" if drop else ""))
    jw(d, "refs.json", {"windows": ws, "anchors": []})
    if not any(p.get("replace_with", "none") != "none" for p in people.values()):
        out.insert(0, "!! NOBODY TO REPLACE YET: plan_grid.json has no person with a replace_with, so every window is a "
                      "passthrough. This is a planning gap, not a finished job: save_case the people and their shots, "
                      "pick keyframes, paint, then run windows again.")
    out.append(f"{len(ws)} windows -> case/refs.json ({sum(not w['passthrough'] for w in ws)} to render); "
               f"maxlen {maxlen} maxcuts {maxcuts}")
    return "\n".join(out)


def take_fp(d, i, w, seed, canvas):
    """A take is keyed by everything it was rendered from, so a repaint or a new prompt never reuses a stale take."""
    C = d / "case"
    # the audio is not hashed, only the mute that changes it (a window without one keeps its old fingerprint)
    h = hashlib.sha1(json.dumps([w["render"], w["own"], w["refs"], seed, "pin", str(canvas)]
                                + ([sorted(w["mute"])] if w.get("mute") else [])).encode())
    h.update((C / f"win{i}/prompt.txt").read_bytes())
    for f in w["refs"]:
        h.update((C / "paint" / ARM / f"f{f}.png").read_bytes())
    if "prefix_of" in w:  # a hop continues the same seed's take of the window before it: a new take there makes it stale
        j = w["prefix_of"]
        h.update(json.dumps([w["prefix_at"], prefix_take(d, j, seed, canvas).name]).encode())
    return h.hexdigest()[:10]


def prefix_take(d, j, seed, canvas):
    """The take of window j a hop of it continues (it may not exist yet: the worker holds the hop until it does)."""
    return d / "case" / "takes" / f"w{j}_{canvas}_s{seed}_{take_fp(d, j, jr(d, 'refs.json')['windows'][j], seed, canvas)}.mp4"


def takes(d, i, w, canvas):
    """{seed: path} of the takes of window i at `canvas` that match its current inputs."""
    out = {}
    for p in (d / "case" / "takes").glob(f"w{i}_{canvas}_s*.mp4"):
        s = int(p.stem.split("_")[2][1:])
        if (d / "case" / f"win{i}/prompt.txt").exists() and p.stem.endswith(take_fp(d, i, w, s, canvas)):
            out[s] = p
    return dict(sorted(out.items()))


def cut_drift(src, take, r0, a0, z0, tol=8, known=()):
    """Cuts of the source inside [a0, z0) that the take puts >= tol frames off, and cuts it adds there; src and take are
    the window's frames at one size. A strip samples 1-2 frames per shot, so it cannot show a cut that lands late or
    early (one window: a whip pan 17 frames early in seed 0 of two runs, on the beat in a third). Matching uses the
    whole render span: a take that cuts a frame early at the end of own is not an extra cut (3 of 10 flags on 111
    takes). A source cut no take cut answers is listed only when the take hardly changes there (under 0.4x the
    source's jump): the detector misses same-palette cuts that the take does make, but a dropped cut is a still take
    (one take: two cuts of a 6-frame montage played on as one group shot, d 17-18 against the source's 63-64).
    A take cut no source cut claims is extra once it is 3+ frames from every source cut, near one or not: one take
    left the shot 13 frames early and cut again a frame after the source cut, that second cut claimed it, and the old 24-frame rule hid the
    early exit, so codex picked s1 as the fix for s0's -16. Not tol: the 4 unclaimed cuts 5-6 frames from a source
    cut over 526 takes are 2 strobes the mirror test drops and 2 real faults, a ghost 6 before a whip pan
    and pillarbox bars that pop in 6 after the cut into a pillarboxed shot; none is nearer. The mirror test keeps it to a source that moves less (under
    0.65x the take's jump): a whip-spin the take follows, or a strobe, reads as a cut in the take only (0.83-
    0.94 over 504 takes; a doodle overlay 0.70, a lightning flash 0.75). Real extras: early exits 0.16-0.32, and
    0.46-0.57 where the take dissolves four dancers into ghosts before it leaves (0.4 missed those)."""
    # known: cuts.json's cuts, which the session may have corrected: the detector misses same-palette cuts that codex
    # adds there (a dark-on-dark cut), and the take that makes such a cut was then flagged as extra
    sc = sorted({r0 + c for c in find_cuts(src[:len(take)])} | {c for c in known if r0 < c < r0 + min(len(src), len(take))})
    tc = [r0 + c for c in find_cuts(take)]
    hit = {}  # each take cut answers one source cut, closest pairs first so a rapid pair is not matched crosswise
    for g, s, t in sorted((abs(t - s), s, t) for s in sc for t in tc):
        if g <= 24 and s not in hit and t not in hit.values():
            hit[s] = t
    jump = lambda v, c: np.abs(np.diff(v[max(c - 3, 0):c + 3, ::4, ::4].astype(np.int16), axis=0)).mean(axis=(1, 2, 3)).max()  # noqa: E731
    # a late or early cut leaves the take still where the source cuts (0.19x the source's jump on one cut); a
    # take that jumps there too made that cut, and the pair is two weak candidates matched (a light flash
    # read as a cut, paired with an in-shot jump 14 frames earlier, both seeds 0.85x - a seed 1 and a BAD for nothing)
    # the cut that ends own counts when the take leaves early, inside own (one window rendered 7 frames past own's end:
    # seed 0 cut 17 frames early and its rerender 15, both unflagged, and the final shows the next shot 15 frames early);
    # a late one there is the next window's frames, never seen
    out = [f"cut f{s} at f{t} ({t - s:+d})" for s, t in sorted(hit.items())
           if (a0 <= s < z0 or s == z0 > t) and abs(t - s) >= tol and jump(take, s - r0) < 0.4 * jump(src, s - r0)]
    # not within 3 of r0: the window opens on a cut that ingest's detector put a frame early, and a take that starts on
    # the new shot cuts 1-2 frames early, not never (both seeds were flagged, the final is on the cut)
    out += [f"cut f{s} dropped (the take plays on through it)" for s in sc
            if a0 <= s < z0 and s - r0 >= 3 and s not in hit and jump(take, s - r0) < 0.4 * jump(src, s - r0)]
    for t in tc:
        if (a0 <= t < z0 and t not in hit.values() and all(abs(t - s) >= 3 for s in sc)
                and jump(src, t - r0) < 0.65 * jump(take, t - r0)):
            s = min(sc, key=lambda s: abs(t - s), default=t + 99)
            out.append(f"extra cut f{t}" + (f" ({t - s:+d} from the source cut f{s})" if abs(t - s) <= 24 else ""))
    return out


def edge_bands(src, take, r0, a0, z0, run=6):
    """Runs of >= run frames inside [a0, z0) where an edge of the take grows a sharp-edged black band the source lacks:
    >15 % of the line black for 4+ lines from the edge, then <5 % at once, within 35 % of the frame. The cause so far was
    always a keyframe that carried the next grid cell's top rows (3 of 213 takes on one film, all true);
    the video model renders the leak on every seed. Frames within 4 of a source cut are skipped."""
    m = min(len(src), len(take))
    sc = [r0 + c for c in find_cuts(src[:m])]
    out = []
    for e, name in zip("TBLR", ("top", "bottom", "left", "right")):
        k, ks = take[:m].max(axis=3) < 40, src[:m].max(axis=3) < 40
        if e in "LR":
            k, ks = k.transpose(0, 2, 1), ks.transpose(0, 2, 1)
        if e in "BR":
            k, ks = k[:, ::-1], ks[:, ::-1]
        p, ps = k.mean(2), ks.mean(2)
        D = p.shape[1]
        hit = []
        for j in range(m):
            lo = np.nonzero(p[j, :int(0.35 * D) + 1] < 0.05)[0]
            if (len(lo) and lo[0] >= 4 and (p[j, :lo[0]] > 0.15).all() and ps[j, :lo[0]].mean() < 0.05
                    and a0 <= r0 + j < z0 and all(abs(r0 + j - c) > 4 for c in sc)):
                hit.append(r0 + j)
        a = None
        for x, y in zip(hit, hit[1:] + [None]):
            a = x if a is None else a
            if y != x + 1:
                if x - a + 1 >= run:
                    out.append(f"{name} edge band f{a}-{x}")
                a = None
    return out


def strip(d, wins, canvas):
    """-> (text, [image paths])"""
    ws = jr(d, "refs.json")["windows"]
    txt, paths = [], []
    for i in wins:
        w = ws[i]
        (r0, r1), (a0, z0) = w["render"], w["own"]
        fs = []
        for s, e in shot_bounds(d):  # 1-2 frames per shot; a keyframe of the shot replaces the sample nearest to it
            s, e = max(s, a0), min(e, z0)
            if e > s:
                smp = [s + (e - s) // 2] if e - s < 48 else [s + (e - s) // 4, s + 3 * (e - s) // 4]
                ks = [f for f in w["refs"] if s <= f < e]
                for k in ks[:1]:
                    smp[min(range(len(smp)), key=lambda j: abs(smp[j] - k))] = k
                # a painted frame always looks right: also show the frame of the shot farthest from its keyframes, where
                # a take drifts (one 46-frame orbit, keyframe on the face, the back of the head turned human)
                if ks:  # a shot without its own keyframe (a rapid-cut repeat of a keyframed setup) has none to be far from
                    far = max(range(min(s + 2, e - 1), max(e - 2, s + 1)), key=lambda f: min(abs(f - k) for k in ks),
                              default=None)
                    if far is not None and min(abs(f - far) for f in smp) >= 12:
                        smp = sorted(smp + [far])
                fs += smp
        W, H = size(d, canvas)
        cw, ch = (320, round(320 * H / W) // 2 * 2) if W >= H else (round(320 * W / H) // 2 * 2, 320)
        src = pick(cond(d, canvas), fs, cw, ch)
        tk = takes(d, i, w, canvas)
        rows, drift, bands = [("src", lambda f: src[f])], [], []
        whole = seg(cond(d, canvas), r0, r1, cw, ch) if tk else None
        for s, p in tk.items():
            v = frames(p, cw, ch)
            rows.append((f"s{s}", lambda f, v=v: v[min(f - r0, len(v) - 1)]))
            drift += [f"s{s} {x}" for x in cut_drift(whole, v, r0, a0, z0, known=jr(d, "cuts.json"))]
            bands += [f"s{s} {x}" for x in edge_bands(whole, v, r0, a0, z0)]
        # the same cut 8+ frames early on 2+ seeds is the input, not the seed (three cuts of one film: -15/-15, -21/-19,
        # -10/-10, and codex re-rolled each for nothing); a muted second before it: mean 14 -> 7, landed 0 -> 5 of 8.
        # Seed 0 alone already tells: 5 of 6 cuts it put 8+ early were early on seed 1 too (-14 / -8,
        # -18 / -18), and seed 1 used up the window's one fix round, so the pin never came
        er = [(x.split()[0], int(x.split()[2][1:])) for x in drift if re.match(r"s\d+ cut f\d+ at f\d+ \(-", x)]
        early = sorted({c for _, c in er if len({s for s, c2 in er if c2 == c}) == len(tk)} - set(w.get("mute", [])) - set(w.get("pinned", [])))
        blocks, cols = [], min(6, len(fs))
        for c in range(0, len(fs), 6):  # 6 columns per block: wider and the cells are too small to judge a face
            part, ims, labs = fs[c:c + 6], [], []
            for name, get in rows:
                for f in part:
                    ims.append(Image.fromarray(get(f)))
                    labs.append(f"{name} f{f}" + (" KEY" if name == "src" and f in w["refs"] else ""))
                ims += [Image.new("RGB", (cw, ch))] * (cols - len(part))
                labs += [" "] * (cols - len(part))
            blocks.append(grid(ims, cols, labs, (cw, ch)))
        (d / "case" / "review").mkdir(exist_ok=True)
        out = d / "case" / "review" / f"{canvas}_w{i}.jpg"
        grid(blocks, 1).save(out, quality=86)
        paths.append(out)
        txt.append(f"case/review/{out.name} own [{a0},{z0}) keyframes {w['refs']} takes s{list(tk)}"
                   + (f" ({len(blocks)} blocks of rows)" if len(blocks) > 1 else "")
                   + (f"\n  cut timing: {'; '.join(drift)}" if drift else "")
                   + (f"\n  early on every seed: {early} - pin_before_cuts({i}, {early}), write_prompts([{i}]), then seed 0 again" if early else "")
                   + (f"\n  edge bands: {'; '.join(bands)}" if bands else ""))
    return "\n".join(txt), paths


def finish(d, picks, rest_source=False, canvas="720"):
    ws = jr(d, "refs.json")["windows"]
    picks = {int(k): int(v) for k, v in picks.items()}
    W, H = size(d, canvas)
    todo = {}
    for i, w in enumerate(ws):
        if w.get("passthrough") or (rest_source and i not in picks):
            continue
        tk = takes(d, i, w, canvas)
        assert i in picks, f"window {i} has no pick (current takes: seeds {list(tk)}); rest_source=true keeps the source there"
        assert picks[i] in tk, f"window {i}: no current {canvas} take of seed {picks[i]} (current: seeds {list(tk)})"
        j = w.get("prefix_of")
        assert j is None or picks.get(j) == picks[i], f"window {i} continues window {j}'s take: pick the same seed for both"
        todo[i] = tk[picks[i]]
    assert todo, ("nothing to finish: no window has a picked take, so the result would be the source unchanged. That is "
                  "not a deliverable. Plan the people (save_case), paint the keyframes, render and review, then finish.")
    jw(d, f"picks_{canvas}.json", {str(i): picks[i] for i in todo})
    (d / "case" / "final").mkdir(exist_ok=True)
    out = d / "case" / "final" / f"full_{canvas}.mp4"
    v = []
    for i, w in enumerate(ws):  # each window gives its own frames; a window with nobody to replace gives the source's
        a0, z0 = w["own"]
        v.append(seg(cond(d, canvas), a0, z0, W, H) if i not in todo else frames(todo[i], W, H)[a0 - w["render"][0]:z0 - w["render"][0]])
        assert len(v[-1]) == z0 - a0, (i, len(v[-1]), z0 - a0)
    v = np.concatenate(v)
    fixed = deflash(d, v, ws, canvas)
    unbarred = unbar(d, v, canvas)
    real, (ow, oh) = jr(d, "meta.json")["frames"], out_size(d, canvas)  # without the padding, at the source's aspect
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", str(d / "case" / "audio.m4a"), "-map", "0:v", "-map", "1:a", "-vf",
                           f"scale={ow}:{oh}:flags=lanczos,setsar=1", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    ff.stdin.write(v[:real].tobytes())
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg failed writing the final"
    return (f"{out} ({ow}x{oh}, {real} frames) kept: "
            + ", ".join(f"w{i} s{picks[i]}" if i in todo else f"w{i} source" for i in range(len(ws)))
            + f" | cut flashes fixed: {fixed or 'none'} | frames whose letterbox was put back: {unbarred}")


def unbar(d, v, canvas):
    """Each frame gets the source's letterbox / pillarbox back where the render drew picture into it: a keyframe
    painted over the bars makes the take fill them for seconds, and the frame shape jumps mid-shot. The bars are the
    whole source's (paint.vbars), and a frame is blacked there only when its own source frame is dark there too.
    -> frames changed"""
    from .paint import vbars
    n, H, W = v.shape[:3]
    W0, H0 = jr(d, "meta.json")["final"]
    l, t, r, b = vbars(d)
    side = {"t": (t * H // H0, lambda m: f"{W}:{m}:0:0", lambda x, m: x[:m]),
            "b": ((H0 - b) * H // H0, lambda m: f"{W}:{m}:0:{H - m}", lambda x, m: x[H - m:]),
            "l": (l * W // W0, lambda m: f"{m}:{H}:0:0", lambda x, m: x[:, :m]),
            "r": ((W0 - r) * W // W0, lambda m: f"{m}:{H}:{W - m}:0", lambda x, m: x[:, W - m:])}
    changed = set()
    for m, crop, part in side.values():
        if not m:
            continue
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(cond(d, canvas)), "-vf", f"scale={W}:{H},crop={crop(m)},"
                              "scale=48:48,format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        dark = np.percentile(np.frombuffer(raw, np.uint8).reshape(-1, 48 * 48)[:n], 98, axis=1) < 14
        for i in np.flatnonzero(dark):
            q = part(v[i], m)
            if q.mean() > 12:
                q[...] = 0
                changed.add(int(i))
    return len(changed)


def deflash(d, v, ws, canvas):
    """A window whose own span starts on a cut renders a few frames of the shot before as context, and the model may
    put the cut up to 2 frames off, so the final flashes 1-2 frames of the wrong shot there. Each frame within 2 of such
    a cut gets the side of the cut the source has, by copying its neighbour on that side. Edits v in place; returns the
    frames it replaced."""
    cuts, (W, H) = set(jr(d, "cuts.json")), size(d, canvas)

    def sides(x):  # x = frames b-4 .. b+3; True = looks like the shot before the cut, for frames b-2 .. b+1
        x = x[:, ::4, ::4].astype(np.float32)
        return [np.abs(x[k] - x[0]).mean() < np.abs(x[k] - x[7]).mean() for k in range(2, 6)]

    todo = []  # (frame, +-1 = copy from the next / previous frame)
    for w in ws[1:]:
        b = w["own"][0]
        if b not in cuts or b < 4 or b + 4 > len(v) or any(0 < abs(b - c) < 8 for c in cuts):
            continue
        want, got = sides(seg(cond(d, canvas), b - 4, b + 4, W, H)), sides(v[b - 4:b + 4])
        fix = {}
        for k in range(4):  # pre-cut frames copy the frame before them, post-cut frames the frame after them
            if want[k] and not got[k]:
                fix[b - 2 + k] = -1
        for k in range(3, -1, -1):
            if not want[k] and got[k]:
                fix[b - 2 + k] = 1
        todo += sorted(fix.items(), key=lambda t: -t[0] * t[1])  # a copy's source is fixed before it is copied
    for f, s in todo:
        v[f] = v[f + s]
    return [f for f, _ in todo]


def peek(src, start, dur, out, every=0.5, cols=6, w=384):
    """Timestamped contact sheet of a span of any video + the span's cut times (ffmpeg scene score > 0.3) and its
    size/fps/duration."""
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(src)],
                                   capture_output=True, text=True, check=True).stdout)
    st = pr["streams"][0]
    with tempfile.TemporaryDirectory() as t:
        r = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(dur), "-i", str(src), "-filter_complex",
                            f"[0:v]split[a][b];[a]fps=1/{every},scale={w}:-2[x];[b]select='gt(scene,0.3)',showinfo[y]",
                            "-map", "[x]", f"{t}/%04d.jpg", "-map", "[y]", "-f", "null", "-"], capture_output=True, text=True)
        cuts = [round(start + float(x), 2) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]
        ims = [Image.open(f).convert("RGB") for f in sorted(Path(t).glob("*.jpg"))]
        assert ims, f"{src}: no frames in [{start}, {start + dur})"
        grid(ims, cols, [f"{start + k * every:.1f}s" for k in range(len(ims))]).save(out, quality=85)
    return {"sheet": str(out), "size": [st["width"], st["height"]], "fps": st["r_frame_rate"],
            "duration": round(float(pr["format"]["duration"]), 1), "cuts_s": cuts}
