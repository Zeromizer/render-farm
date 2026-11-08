"""Viggle Remake MCP: whole-person swap and object/background edit of a video, one session per video under
$REMAKE_HOME/sessions/<sid>, as a stdio MCP server for an agent (Claude Code, codex, ...). Two session kinds:
  swap  replace people (face, hair, body and clothes) with characters from sheets
  edit  remove or replace an object, a set or the background
Every tool takes the session id and works on that dir only (no cwd, no globals per session). Renders go through the
queue in $REMAKE_HOME/queue to viggle-remake-worker processes (jobs.py). AGENTS.md (in the package data) is the
agent's manual.
"""
import base64
import functools
import io
import difflib
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent
from PIL import Image

from . import jobs, kit, paint as pt, prompts
from .common import ARM, CHARS, FPS, LENS, SESSIONS, STOP, jr, jw, kind
from .prompts import DATA

NAME = re.compile(r"[A-Za-z0-9-]{1,48}")
server = MCPServer(name="viggle-remake", instructions=(
    f"Viggle Remake: whole-person swap and object/background edit videos, one session per video under {SESSIONS}/<sid>. "
    f"Read {DATA / 'AGENTS.md'} before the first call; status(sid) is the ground truth when you resume a session."
    + (" CODEX IMAGE MODE: sheet/paint/epaint do not call an image API; their first call returns a CODEX IMAGE STEP "
       "that you carry out with your own image_gen tool, then you call the same tool again (AGENTS.md, codex image mode)."
       if pt.CODEX else "")
    + (f" POOL IMAGE MODE: sheet/paint/epaint paint in one call (a helper codex); paint_many paints up to {pt.POOL_N} "
       "keyframe groups at once - use it for every batch of keyframes." if pt.POOL else "")
    + (" PROMPT POOL MODE: write_prompts writes and stores the prompts itself (a helper codex per window, all at once) "
       "and returns their lint results; to fix one, call it again for that window with notes=..." if prompts.PPOOL else "")
    + ("" if prompts.LLM else " Prompts are yours to write: write_prompts returns a brief per window, set_prompt stores "
       "what you wrote.")))


def tool(description):
    """server.tool whose errors reach the agent (the SDK shows any exception but ToolError as a bare 'Error executing
    tool'); in codex image mode an image tool's first call returns the CODEX IMAGE STEP."""
    def deco(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            try:
                return fn(*a, **k)
            except pt.Brief as b:
                return [pt.step(b.args[0], fn.__name__)]
            except Exception as e:
                raise ToolError(f"{type(e).__name__}: {e}") from e
        return server.tool(structured_output=False, description=description)(w)
    return deco


def img(path, width=1600):
    """JPEG image block; codex imageDetail=original keeps frame labels readable."""
    im = Image.open(path)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=88)
    return ImageContent(type="image", data=base64.b64encode(buf.getvalue()).decode(), mimeType="image/jpeg",
                        _meta={"codex/imageDetail": "original"})


RUN_SIDS = os.environ.get("REMAKE_RUN_SIDS")


def sess(sid):
    d = SESSIONS / sid
    assert NAME.fullmatch(sid) and d.is_dir(), f"no session {sid!r}: new_session makes one"
    # the workers write under case/ outside any agent sandbox: a link planted there would carry writes out of the session
    C = d / "case"
    bad = [str(p.relative_to(d)) for p in [C, *C.glob("*"), *C.glob("*/*")] if p.is_symlink()]
    assert not bad, f"symlinks under case/ are not allowed: {bad}"
    if RUN_SIDS:  # codex_exec.sh resumes a session this run touched but did not deliver
        with open(RUN_SIDS, "a+") as f:
            f.seek(0)
            if sid not in f.read().split():
                f.write(sid + "\n")
    return d


def path(d, p):
    """A path argument: absolute, or relative to the session dir."""
    return Path(p) if Path(p).is_absolute() else d / p


def into_session(d, p):
    """A picture the painter gets must be a file of the session dir: copy any other into photos/."""
    p = path(d, p)
    assert p.is_file(), f"{p}: no such file"
    if p.resolve().is_relative_to(d.resolve()):
        return str(p.resolve().relative_to(d.resolve()))
    (d / "photos").mkdir(exist_ok=True)
    shutil.copy(p, d / "photos" / p.name)
    return f"photos/{p.name}"


def need(d, k):
    assert kind(d) == k, f"this tool is for {k} sessions; {d.name} is a {kind(d)} session"


@tool("Create a session dir $REMAKE_HOME/sessions/<sid>. kind: 'swap' (whole-person replacement) or 'edit' (remove/"
      "replace an object or the background). The kind is fixed for the session. sid: letters, digits and '-' only.")
def new_session(sid: str, kind: str) -> str:
    assert NAME.fullmatch(sid), "sid: letters, digits and - only (the job queue matches on _<sid>_)"
    assert kind in ("swap", "edit"), "kind is 'swap' or 'edit'"
    d = SESSIONS / sid
    if d.exists():
        k = (d / "kind").read_text().strip()
        assert k == kind, f"{sid} exists as {'an' if k == 'edit' else 'a'} {k} session"
        return f"{d} exists ({kind}); status() shows where it stands"
    d.mkdir(parents=True)
    (d / "kind").write_text(kind)
    return f"{d} created ({kind})"


@tool("Where a session stands: case files, keyframes painted or not, windows with refs/hops, prompts, current takes, "
      "this session's queue and image calls, the live workers, and the case check's report. Call it first when you "
      "resume.")
def status(sid: str) -> str:
    d = sess(sid)
    out = [f"{d} {kind(d)}"]
    if not (d / "case" / "meta.json").exists():
        return out[0] + "\nnot ingested yet (ingest)"
    nf, cuts, plan, where = kit.nframes(d), jr(d, "cuts.json"), jr(d, "plan_grid.json"), jr(d, "where.json", {})
    painted = sorted(int(f) for f in where if (d / f"case/paint/{ARM}/f{f}.png").exists())
    out.append(f"{nf} frames ({nf / FPS:.1f} s), cuts {cuts}")
    out.append("people: " + ", ".join(f"{p['id']}->{p.get('replace_with', '?')}" for p in plan["people"]))
    out.append(f"keyframes (where.json) {sorted(int(f) for f in where)}, painted {painted}")
    for i, w in enumerate(jr(d, "refs.json", {}).get("windows", [])):
        win = d / f"case/win{i}"
        out.append(f"w{i} render {w['render']} own {w['own']} " + ("PASSTHROUGH" if w.get("passthrough") else
                   f"refs {w['refs']}" + (f" hop of w{w['prefix_of']}" if "prefix_of" in w else "")
                   + (f" (why: {w['why']})" if w.get("why", "kit windows") != "kit windows" else "")
                   + f" | prompt {'yes' if (win / 'prompt.txt').exists() else 'no'}"
                   + f" | 720 takes seeds {list(kit.takes(d, i, w, '720'))}"))
    st = jobs.state(sid)
    out.append(f"queue: {st['queued']} queued, {st['running']} running, {st['done']} done, {st['failed']} failed"
               + (f", hops waiting for their prefix {st['hops_waiting']}" if st["hops_waiting"] else "")
               + (f", HOPS STUCK (nothing pending makes their prefix) {st['hops_stuck']}" if st["hops_stuck"] else "")
               + (f", ORPHANED (worker died) {st['orphaned']}" if st["orphaned"] else ""))
    out.append("workers: " + ("; ".join(f"{hb['host']}:{hb['gpu']} {hb['job'] or 'idle'} max_frames {hb['max_frames']}"
                                        for hb in st["workers"]) or "none live"))
    out.append(f"image calls {pt.calls(d)}/{pt.limit(d)}" + (f" | PAINT STOPPED: {STOP.read_text()}" if STOP.exists() else ""))
    fin = sorted((d / "case/final").glob("full_*.mp4"))
    out.append("final: " + (", ".join(map(str, fin)) or "none"))
    out.append("result.json: " + ("yes" if (d / "result.json").exists() else "no"))
    out.append("check:\n" + kit.report(d))
    return "\n".join(out)


@tool("Scout a source clip before ingesting: contact sheet every 0.5 s of [start, start+dur) with timestamps, plus "
      "its size, fps and cut times. src = a local video path (absolute, or relative to the session dir).")
def peek(sid: str, src: str, start: float = 0.0, dur: float = 20.0) -> list:
    d = sess(sid)
    (d / "scout").mkdir(exist_ok=True)
    out = d / "scout" / f"peek_{start:g}_{dur:g}.jpg"
    r = kit.peek(path(d, src), start, dur, out)
    return [json.dumps(r), img(out)]


@tool("Ingest the source span -> case/ (24 fps; the draft and final canvases follow the source's aspect, 832x480 / "
      "1280x736 for 16:9, and finish restores it), cuts, empty plan/case/where files, then strips (one row of 4 frames "
      "per detector shot). A span of at most 345 frames that is not 17k+5 long is padded with its last shot played "
      "backwards; finish cuts the padding off. force=true re-ingests and discards plan, where.json, paints, prompts and "
      "takes (case.json and sheets are kept). crop = 'w:h:x:y' source px.")
def ingest(sid: str, src: str, start: float = 0.0, dur: float | None = None, crop: str | None = None,
           force: bool = False) -> list:
    d = sess(sid)
    (d / "case").mkdir(exist_ok=True)
    txt = kit.ingest(d, path(d, src), start, dur, crop, force)
    return [txt] + strips(sid)


# codex drops a tool result over 16 MiB on one line and then waits out the tool timeout: one film (233 shots, 24 strip
# pages, ~28 MB of base64) hung in ingest
MAX_B64 = 12_000_000


@tool("case/look/strips_<p>.jpg, one row of 4 frames per detector shot (labels S<k> f<frame>), from page first_page on, "
      "as many pages as fit in one result. Re-run after fixing cuts.json.")
def strips(sid: str, first_page: int = 0) -> list:
    d = sess(sid)
    txt, paths = kit.strips(d)
    out, n = [txt], 0
    for k in range(first_page, len(paths)):
        b = img(paths[k])
        n += len(b.data)
        if n > MAX_B64 and len(out) > 1:
            out[0] += (f"\nPages {first_page}..{k - 1} are below; pages {k}..{len(paths) - 1} did not fit in this result: "
                       f"call strips(sid, first_page={k}) to see them.")
            return out
        out.append(b)
    return out


@tool("Up to 12 exact frames, large (960 wide, from the final-canvas cond) -> one image.")
def frames(sid: str, frames: list[int]) -> list:
    d = sess(sid)
    txt, p = kit.frames_cmd(d, frames)
    return [txt, img(p, 1920)]


@tool("Write the case files you pass (whole files, each replaces the old one): plan_grid (people + shots), case (chars, "
      "outfits, and for edit sessions 'edit'), where (keyframe -> who is where), cuts (frame list; fix it together with "
      "the plan's shots). Then runs the case check (validates, writes paint/src, lists shots still missing a keyframe).")
def save_case(sid: str, plan_grid: dict | None = None, case: dict | None = None, where: dict | None = None,
              cuts: list[int] | None = None) -> str:
    d = sess(sid)
    assert (d / "case/meta.json").exists(), "ingest first"
    for name, obj in (("plan_grid.json", plan_grid), ("case.json", case), ("where.json", where), ("cuts.json", cuts)):
        if obj is not None:
            jw(d, name, obj)
    warn = []
    where, nf = jr(d, "where.json", {}), kit.nframes(d)
    bounds = sorted({0, nf, *jr(d, "cuts.json")})
    # one line, not one per keyframe: on one film every save_case repeated 11-13 of these (8-13 calls per run)
    near = [f for f in sorted(int(f) for f in where) if any(0 <= f - b < 3 or 0 < b - f <= 3 for b in bounds[1:-1])]
    if near:
        warn.append(f"keyframes within 3 frames of a cut: f{' f'.join(map(str, near))} - keep 3 frames of its own shot "
                    "between it and the cut (f >= cut + 3 after it, f <= cut - 4 before it; cut = the new shot's first "
                    "frame). A shot under 7 frames has no such frame: leave it")
    if kind(d) == "edit":
        c = jr(d, "case.json")
        if not c.get("edit"):
            warn.append("edit session: case.json has no 'edit' sentence")
        if any(v != "none" for v in c.get("outfits", {}).values()):
            warn.append("edit session: outfits are 'none' for edited objects")
    return "\n".join(["warning: " + w for w in warn] + [kit.report(d)])


@tool("Render windows (ending on cuts, a longer shot continued by hops) and each window's keyframes (<= 4) -> "
      "case/refs.json. maxlen: frames per window, default = the most the live workers take at `canvas` (345 when no "
      "worker is up); a larger maxlen is lowered to that. Refs you set by hand (set_refs) are kept for windows whose "
      "span did not change. Run after every change of where.json or the paints, before prompts.")
def windows(sid: str, maxlen: int | None = None, maxcuts: int = 3, canvas: str = "720") -> str:
    d = sess(sid)
    assert canvas in ("480", "720"), "canvas 480 | 720"
    tier = 480 if canvas == "480" else 736
    live, note = jobs.max_frames(kind(d), tier), ""
    if live is None:
        cap, note = LENS[-1], f"\nwarning: no live worker; windows sized for {LENS[-1]} frames (run windows again once one is up)"
    else:
        fit = [x for x in LENS if x <= live]
        assert fit, f"the live workers take at most {live} frames at tier {tier}: below the shortest render {LENS[0]}"
        cap = fit[-1]
    if maxlen and maxlen > cap:
        note += f"\nmaxlen {maxlen} lowered to {cap}: the most the live workers take at tier {tier}"
    maxlen = min(maxlen or cap, cap)
    old = jr(d, "refs.json", {}).get("windows", [])
    txt = kit.windows_cmd(d, maxlen, maxcuts)
    hand = {json.dumps([w["render"], w["own"]]): w for w in old if w.get("why", "kit windows") != "kit windows"}
    rf, where, kept = jr(d, "refs.json"), jr(d, "where.json", {}), []
    mutes = {json.dumps([w["render"], w["own"]]): w["mute"] for w in old if w.get("mute")}
    for i, w in enumerate(rf["windows"]):
        if json.dumps([w["render"], w["own"]]) in mutes:
            w["mute"] = mutes[json.dumps([w["render"], w["own"]])]
        h = hand.get(json.dumps([w["render"], w["own"]]))
        if h and not w["passthrough"] and all(str(f) in where for f in h["refs"]):
            w.update(refs=h["refs"], why=h["why"], **({"pinned": h["pinned"]} if h.get("pinned") else {}))
            kept.append(f"w{i} refs {h['refs']} ({h['why']})")
            txt = re.sub(rf"^(w{i:2d} own .*?)refs \[[^\]]*\]", lambda m: f"{m[1]}refs {h['refs']} (hand-set)", txt,
                         count=1, flags=re.M)
    jw(d, "refs.json", rf)
    return txt + note + ("\nkept hand-set refs (shown above as hand-set): " + "; ".join(kept) if kept else "")


@tool("Set one window's keyframes by hand (e.g. a second keyframe in a shot: facing flip or a new replaced person, or an "
      "edit shot > 8 s). Frames must be in where.json and inside the window's own span, at most 4; a hop may have none. "
      "why: the reason, kept in refs.json. Write the window's prompt again afterwards.")
def set_refs(sid: str, window: int, refs: list[int], why: str) -> str:
    d = sess(sid)
    rf, where, plan = jr(d, "refs.json"), jr(d, "where.json", {}), jr(d, "plan_grid.json")
    w = rf["windows"][window]
    a, b = w["own"]
    assert not w.get("passthrough"), f"w{window} is a passthrough window (nobody to replace)"
    assert why.strip() and why.strip() != "kit windows", "write why these keyframes"
    assert len(refs) <= 4 and (refs or "prefix_of" in w), "1-4 keyframes (a hop may have 0)"
    for f in refs:
        assert str(f) in where, f"f{f} is not in where.json"
        assert a <= f < b, f"f{f} is outside w{window}'s own span [{a},{b})"
    for s in plan["shots"]:
        fs = sorted(f for f in refs if s["start"] <= f < s["end"])
        assert all(y - x >= 72 for x, y in zip(fs, fs[1:])), (f"keyframes {fs} in one shot must be >= 3 s (72 frames) apart. Two camera setups the detector merged: add the "
                                                        "cut (save_case cuts + plan_grid shots). A flash of a few frames gets no keyframe of its own")
    w.update(refs=sorted(refs), why=why.strip())
    jw(d, "refs.json", rf)
    return f"w{window} refs {w['refs']} ({w['why']})"


@tool("Silence the second before the given source cuts in the audio the video model hears for this window (the "
      "final keeps the real soundtrack), or clear it with cuts=[]. For a cut seed 0 put 8+ frames early "
      "(`early on every seed`), when pin_before_cuts cannot (no paint left): the next speaker's voice starting over the outgoing shot pulls the cut to the voice. Changes "
      "the window's takes (submit seeds 0 and 1 again, keep the one whose cut lands); the prompt stays valid.")
def mute_before_cuts(sid: str, window: int, cuts: list[int]) -> str:
    d = sess(sid)
    rf = jr(d, "refs.json")
    w = rf["windows"][window]
    a, b = w["own"]
    bad = [c for c in cuts if c not in jr(d, "cuts.json") or not a < c < b]
    assert not bad, f"{bad}: not source cuts inside w{window}'s own span ({a},{b})"
    w.pop("mute", None)
    if cuts:
        w["mute"] = sorted(set(cuts))
    jw(d, "refs.json", rf)
    prompts.cut_window(d, window, w)
    return f"w{window} mute {w.get('mute', [])}: case/win{window}/audio.m4a re-cut; submit renders it as new takes"


@tool("Swap sessions: for a cut seed 0 already put 8+ frames early (`early on every seed`), paint a keyframe of the OUTGOING "
      "shot 6 frames before each cut, so the take holds that shot up to the cut (measured on 4 cuts x 2 seeds: mean 14 frames "
      "early -> 2.75, all 8 within 5; mute_before_cuts only reached 7). Its where entry is copied from that shot's "
      "nearest keyframe; a keyframe of the shot within 72 frames of it leaves the window. One image call per cut; a shot "
      "with nobody to replace gets its source frame, no call (17 frames early -> 0). Then "
      "write_prompts for the window and submit seed 0 again. Returns the src|paint comparisons: read them.")
def pin_before_cuts(sid: str, window: int, cuts: list[int], quality: str = "low") -> list:
    d = sess(sid)
    need(d, "swap")
    assert quality in ("low", "medium"), "quality low | medium (never high)"
    rf, where, plan = jr(d, "refs.json"), jr(d, "where.json", {}), jr(d, "plan_grid.json")
    w = rf["windows"][window]
    a, b = w["own"]
    cuts = sorted(set(cuts))
    assert cuts, "no cuts"
    swap = {p["id"] for p in plan["people"] if p.get("replace_with", "none") != "none"}
    refs, new, bare = list(w["refs"]), [], []
    for c in cuts:
        f = c - 6
        # a cut on the window's end counts while the render runs past it: one window's render ran 9 frames past own, and the take
        # left the shot before its last cut 18 frames early
        assert c in jr(d, "cuts.json") and a <= f and c <= b and c < w["render"][1], \
            f"f{c}: not a source cut inside w{window}'s render with 6 frames of its own span [{a},{b}) before it"
        shot = next(s for s in plan["shots"] if s["start"] <= f < s["end"])
        assert shot["end"] == c and f - shot["start"] >= 3, f"the shot before f{c} is under 9 frames: mute_before_cuts({window}, [{c}]) instead"
        near = sorted((int(x) for x in where if shot["start"] <= int(x) < shot["end"] and int(x) != f), key=lambda x: abs(x - f))
        if not set(shot["people"]) & swap:
            # nobody to replace in the outgoing shot: its source frame is the keyframe, no image call (two such shots on one
            # film: seed 0 left them 7 and 17 frames early, pinned -5 and 0)
            where[str(f)] = {}
            bare.append(f)
        else:
            assert near, f"the shot before f{c} has no keyframe in where.json: give it one first"
            # the copied text names its own frame ("as seen at frame 1574"); a text without one gets one, or the same
            # text would land in a 4th keyframe and save_case refuses it
            where[str(f)] = {k: v.replace(f"frame {near[0]}", f"frame {f}") if f"frame {near[0]}" in v else f"{v} (frame {f})"
                             for k, v in where[str(near[0])].items()}
            new.append(f)
        refs = [x for x in refs if not (shot["start"] <= x < shot["end"] and abs(x - f) < 72)] + [f]
    assert len(refs) <= 4, f"w{window} would have {len(refs)} keyframes (max 4): drop one with set_refs first"
    assert pt.calls(d) + len(new) <= pt.limit(d), \
        f"{len(new)} pins would pass the session's {pt.limit(d)} image calls ({pt.calls(d)} used)"
    # unpin puts the window back as it was: one window, pinned -6 before an early cut, kept the cut on time but seed 0 then
    # played on through the 6-frame shot after it, and the takes before the pin had no way back
    if not (d / "case" / f"win{window}/prepin.json").exists():  # kept until unpin: a repeat call (codex image mode
        # returns a brief first, after refs/where changed) or a second pin must not overwrite the state before the first
        jw(d, f"win{window}/prepin.json", {"window": {k: w.get(k) for k in ("refs", "why", "pinned")},
                                           "pins": new + bare, "texts": {n: (d / "case" / f"win{window}" / n).read_text()
                                           for n in ("prompt.txt", "prompt.json", "stamp.json")
                                           if (d / "case" / f"win{window}" / n).exists()}})
    out = [save_case(sid, where=where)]
    rf = jr(d, "refs.json")  # refs first: in codex image mode the paint below returns the image step instead
    w = rf["windows"][window]
    w.update(refs=sorted(refs), why=f"pin before early cuts {cuts}", pinned=sorted(set(w.get("pinned", [])) | set(cuts)))
    jw(d, "refs.json", rf)
    out.append(f"w{window} refs {w['refs']} ({w['why']}): write_prompts([{window}]), then submit seed 0")
    for f in bare:
        shutil.copy(d / "case/paint/src" / f"f{f}.png", d / "case/paint" / ARM / f"f{f}.png")
        out.append(f"f{f}: nobody to replace in that shot, the source frame is its keyframe (no image call)")
    if not new:
        return out
    with ThreadPoolExecutor(len(new)) as ex:
        rs = list(ex.map(lambda f: pt.paint(d, [f], "the last frames of this shot before the cut", True, quality), new))
    for r in rs:
        out.append(json.dumps(r, ensure_ascii=False))
        out.append(img(d / r["cmp"]))
    return out


@tool("Swap sessions: undo the last pin_before_cuts of a window when the pinned take came out worse than the one "
      "before (a shot dropped, a new early cut, a worse face). Puts back the window's keyframes and prompt, so the takes "
      "rendered before the pin count again (no render, no image call) and the pin's keyframes leave where.json. Hop windows "
      "that continue this window's take need submitting again.")
def unpin(sid: str, window: int) -> str:
    d = sess(sid)
    need(d, "swap")
    bk = jr(d, f"win{window}/prepin.json")
    assert bk, f"w{window}: no pin_before_cuts to undo"
    rf = jr(d, "refs.json")
    w = rf["windows"][window]
    for k, v in bk["window"].items():
        w.pop(k, None) if v is None else w.update({k: v})
    jw(d, "refs.json", rf)
    for n, t in bk["texts"].items():
        (d / "case" / f"win{window}" / n).write_text(t)
    save_case(sid, where={k: v for k, v in jr(d, "where.json", {}).items() if int(k) not in bk["pins"]})
    (d / "case" / f"win{window}/prepin.json").unlink()
    hops = [i for i, x in enumerate(rf["windows"]) if x.get("prefix_of") == window]
    return (f"w{window} refs {w['refs']} as before the pin; 720 takes that count again: seeds {list(kit.takes(d, window, w, '720'))}"
            + (f"; hop windows {hops} continue this window's take: submit them again" if hops else ""))


@tool("Swap sessions: a character sheet case/chars/NAME_sheet.png. source='lib': copy $REMAKE_HOME/chars/NAME_sheet.png "
      "(free; fills case.json chars/outfits from $REMAKE_HOME/chars/lib_*.json if missing). source='photo': from a "
      "picture (absolute or session-relative path; copied into photos/), kind person|anthro|creature (one image call). "
      "source='text': from `text` only (an invented animal/creature; one image call). photo and text both need "
      "case.json outfits[NAME] (incl. shoes) first. Image calls count against the session's limit and USD cap.")
def sheet(sid: str, name: str, source: str, photo: str | None = None, text: str | None = None,
          kind: str = "person") -> list:
    d = sess(sid)
    need(d, "swap")
    assert source in ("lib", "photo", "text"), "source: lib | photo | text"
    txt = pt.sheet(d, name, source, into_session(d, photo) if source == "photo" else None, text, kind)
    return [txt, img(d / "case/chars" / f"{name}_sheet.png")]


@tool("Swap sessions: paint 1 or 2 keyframes (2 = one grid sharing one look: same characters; the two keyframes of one "
      "shot always go together) from the sheets and real photos. quality: 'medium' when a character comes with an identity photo, else 'low'. force=true repaints "
      "(the old paint goes to paint/g2/old/); note = what was wrong. Returns the src|paint comparison: read it.")
def paint(sid: str, frames: list[int], note: str = "", force: bool = False, quality: str = "low") -> list:
    d = sess(sid)
    need(d, "swap")
    assert quality in ("low", "medium"), "quality low | medium (never high)"
    r = pt.paint(d, frames, note, force, quality)
    return [json.dumps(r), img(d / r["cmp"])]


if pt.POOL:
    @tool(f"Swap sessions, pool image mode: paint several keyframe groups at once ({pt.POOL_N} in parallel). groups: a "
          "list of `frames` lists as `paint` takes them (1-2 frames each, the two keyframes of one shot together); no "
          "frame twice. notes: optional {first frame of a group: note}. Same rules, limit and comparisons as `paint`; a "
          "group that fails is reported and the others still land. Read every comparison it returns.")
    def paint_many(sid: str, groups: list[list[int]], notes: dict[str, str] | None = None, force: bool = False,
                   quality: str = "low") -> list:
        d = sess(sid)
        need(d, "swap")
        fl = [f for g in groups for f in g]
        assert groups and len(fl) == len(set(fl)), "groups: a non-empty list of frame lists, no frame twice"
        notes = notes or {}
        split = [g for g in groups if len(g) == 2 and pt.square_pair(d, g)]
        notes.update({str(g[1]): notes[str(g[0])] for g in split if str(g[0]) in notes and str(g[1]) not in notes})
        groups = [h for g in groups for h in ([[f] for f in g] if g in split else [g])]
        assert pt.calls(d) + len(groups) <= pt.limit(d), \
            f"{len(groups)} groups would pass the session's {pt.limit(d)} image calls ({pt.calls(d)} used)"

        def one(g):
            try:
                return pt.paint(d, g, notes.get(str(g[0]), ""), force, quality)
            except Exception as e:
                return {"frames": g, "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(pt.POOL_N) as ex:
            rs = list(ex.map(one, groups))
        out = [f"painted one frame per call: {split} (a 2-frame grid here is near 1:1, which the image model returns at 5:4)"] if split else []
        for r in rs:
            out.append(json.dumps(r, ensure_ascii=False))
            if "cmp" in r:
                out.append(img(d / r["cmp"]))
        return out


@tool("Edit sessions: paint one keyframe = the image model's edit of the source frame by `edit` (what changes and what "
      "fills the space). refs: optional pictures of what the edit brings in (absolute or session-relative paths, copied "
      "into photos/). quality low by default. force=true repaints. Returns the src|edit comparison: read it.")
def epaint(sid: str, frame: int, edit: str, refs: list[str] | None = None, force: bool = False,
           quality: str = "low") -> list:
    d = sess(sid)
    need(d, "edit")
    assert quality in ("low", "medium"), "quality low | medium (never high)"
    r = pt.epaint(d, frame, edit, [into_session(d, x) for x in refs or []], force, quality)
    return [json.dumps(r), img(d / r["cmp"])]


@tool("Swap sessions: a second keyframe for a shot, cut from a finished take instead of painted from the source - for a "
      "part the shot's keyframe does not show (legs, shoes, a hem, a carried bag) that the take gets wrong for >= 2 s. "
      "Frame `frame` of the latest `seed` take of `window` becomes keyframe f<frame>; with `fix` (what is wrong and what "
      "it should be, e.g. 'her skirt ends at the knee; it is a floor-length white gown') and `box` ([x0, y0, x1, y1], "
      "fractions of the frame around the wrong part; read it off `frames`, whose frames carry .1 marks on the bottom and left "
      "edges, and the returned take panel outlines it), one image call repairs only inside the box and the take's pixels "
      "stay everywhere else. Without `fix` it goes in as it is (free). who: where.json's entry for it (default: the "
      "shot's keyframe's). It joins where.json and the window's refs; then write_prompts + set_prompt for that window "
      "and submit. Returns take|edit|keyframe: read it.")
def take_keyframe(sid: str, window: int, frame: int, fix: str | None = None, box: list[float] | None = None,
                  who: dict | None = None, seed: int = 0, canvas: str = "720") -> list:
    d = sess(sid)
    need(d, "swap")
    plan, where, rf = jr(d, "plan_grid.json"), jr(d, "where.json", {}), jr(d, "refs.json")
    w = rf["windows"][window]
    (a, b), (r0, r1) = w["own"], w["render"]
    assert a <= frame < b, f"f{frame} is outside w{window}'s own span [{a},{b})"
    shot = next(s for s in plan["shots"] if s["start"] <= frame < s["end"])
    own = [f for f in w["refs"] if shot["start"] <= f < shot["end"] and f != frame]
    # a hop's first shot carries on from the prefix with no keyframe of its own, and drift there has no other fix: on one
    # film, 27 s into a 68 s dance shot, black trousers turned blue jeans and the review could only report it
    carried = "prefix_of" in w and shot["start"] < a
    near = sorted((int(f) for f in where if shot["start"] <= int(f) < shot["end"] and int(f) != frame), key=lambda f: abs(f - frame))
    assert own or (carried and near), f"f{frame}'s shot has no keyframe in w{window}: give the shot its own painted keyframe first"
    fs = sorted(own + [frame])
    assert all(y - x >= 72 for x, y in zip(fs, fs[1:])), f"keyframes {fs} in one shot must be >= 72 frames apart"
    refs = sorted(set(w["refs"]) | {frame})
    assert len(refs) <= 4, f"w{window} would have {len(refs)} keyframes (max 4)"
    assert not fix or (box and len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1), \
        "a fix needs box = [x0, y0, x1, y1], fractions of the frame (0..1)"
    takes = sorted((d / "case/takes").glob(f"w{window}_{canvas}_s{seed}_*.mp4"), key=lambda p: p.stat().st_mtime)
    assert takes, f"no {canvas} seed-{seed} take of w{window} yet: submit and wait first"
    who = who or where[str(own[0] if own else near[0])]
    for i, t in who.items():  # save_case refuses one text in 4+ keyframes: say so before the paint, not at write_prompts
        n = sum(1 for f, wf in where.items() if f != str(frame) and wf.get(i) == t)
        assert n < 3, (f"{i}'s where text {t[:60]!r} is already in {n} keyframes and save_case refuses a 4th: pass who= "
                       f"with where each person is in f{frame} and how much shows")
    people = {p["id"]: p for p in plan["people"]}
    chars = list(dict.fromkeys(people[i]["replace_with"] for i in who
                               if not i.startswith("_") and people[i].get("replace_with", "none") != "none"))
    r = pt.take_paint(d, takes[-1], frame - r0, frame, chars, fix, box)
    where[str(frame)] = who
    chk = save_case(sid, where=where)  # writes paint/src + the paint inventory for the new frame
    w.update(refs=refs, why=(w.get("why", "") + f"; f{frame} from take {takes[-1].name}" + (f", fixed: {fix}" if fix else "")).strip("; "))
    jw(d, "refs.json", rf)
    return [json.dumps(r, ensure_ascii=False) + f"\nw{window} refs {refs}\n" + chk[-600:], img(d / r["cmp"])]


@tool("Swap sessions: the six-section render prompts for all windows or the given ones; cuts case/win<i>/ first. By "
      "default (no server LLM configured) returns one brief per window - its facts, the rules, the guide and example "
      "paths, its <Picture k> images in order and its contact sheet - and you write each prompt and store it with "
      "set_prompt. Briefs past ~32K characters wait: the result's first and last lines name the windows NOT RETURNED; "
      "call again for them. With REMAKE_LLM_MODEL set the server writes them and returns the warnings and every prompt; "
      "in PROMPT POOL mode (codex-pool) one helper codex per window writes and stores them and you get only each window's "
      "lint result - notes= passes what to fix (a review's repair, action timings) to the writer. Needed "
      "again after any repaint, keyframe or window change. The guide and format example come with the session's first "
      "call only; guide=true repeats them.")
def write_prompts(sid: str, windows: list[int] | None = None, guide: bool = False, notes: str = "") -> list:
    d = sess(sid)
    txt, paths = prompts.write_prompts(d, windows, guide, notes)
    return [txt] + [img(p) for p in paths]


@tool("Store the prompt you wrote for one window (either session kind): six sections in order, every <Picture k> of the "
      "window's refs named. Binds it to the window, its keyframes and case.json like write_prompts does, so submit "
      "takes it; any later repaint, keyframe or window change makes it stale. The window must be cut for its current "
      "span (write_prompts / cut_windows).")
def set_prompt(sid: str, window: int, text: str) -> str:
    return prompts.set_prompt(sess(sid), window, text)


@tool("Edit sessions: cut case/win<i>/ (cond.mp4, audio.m4a) for all or the given windows and return each window's "
      "contact sheet (every 0.5 s, timestamps from the window's first frame), render span and keyframes - what you "
      "write the prompt from. Never touches a prompt.")
def cut_windows(sid: str, windows: list[int] | None = None) -> list:
    d = sess(sid)
    txt, paths = prompts.cut_windows(d, windows)
    return [txt] + [img(p) for p in paths]


@tool("Replace case/win<i>/prompt.txt with `text` (six sections in order, every <Picture k> of the window's refs named). "
      "Edit sessions: the same as set_prompt (binds it to the window and its paints, required before submit). Swap "
      "sessions: after write_prompts / set_prompt, to fix the text (e.g. explicit action timings) without binding it "
      "again.")
def edit_prompt(sid: str, window: int, text: str) -> str:
    return prompts.edit_prompt(sess(sid), window, text)


@tool("Queue renders for the workers (720 canvas, seed 0 by default; a fix round: seeds [1] for one window, once). "
      "Refuses windows with unpainted keyframes or stale prompts, windows longer than the live workers take, and the "
      "720 canvas when no live worker renders it. Hops are queued at once (a worker holds one until the take it "
      "continues exists). Existing takes / queued jobs are skipped; a queued job of the same window, seed and canvas "
      "from older inputs is withdrawn.")
def submit(sid: str, seeds: list[int] = [0], windows: list[int] | None = None, canvas: str = "720") -> str:
    d = sess(sid)
    assert canvas in ("480", "720"), "canvas 480 | 720"
    return json.dumps(jobs.submit(d, sid, seeds, windows, canvas), ensure_ascii=False)


@tool("Block until none of this session's jobs is queued or running (a hop whose prefix nothing pending makes counts "
      "as stuck, not queued), no worker is live, or max_s seconds (<= 1800). The queue is shared: call again if it is "
      "not done.")
def wait(sid: str, max_s: int = 540) -> str:
    sess(sid)
    return json.dumps(jobs.wait(sid, max_s), ensure_ascii=False)


@tool("Review strip per window - the src row and one row per current take, 1-2 frames per shot, KEY = keyframe. "
      "Returns the images: compare every column with src.")
def strip(sid: str, windows: list[int] | None = None, canvas: str = "720") -> list:
    d = sess(sid)
    ws = windows or [i for i, w in enumerate(jr(d, "refs.json")["windows"]) if not w.get("passthrough")]
    txt, paths = kit.strip(d, ws, canvas)
    return [txt] + [img(p, 1920) for p in paths]


@tool("The picked take per window ({window: seed}; a hop must use its prefix window's seed) -> "
      "case/final/full_<canvas>.mp4 with the source audio; 1-2 frame cut flashes at window seams are fixed inside. "
      "rest_source=true keeps the source footage for windows without a pick.")
def finish(sid: str, picks: dict[str, int], rest_source: bool = False, canvas: str = "720") -> str:
    return kit.finish(sess(sid), picks, rest_source, canvas)


@tool("This session's image calls and estimated spend vs its limits, failed calls, whether painting is STOPPED, and the "
      "paint mode.")
def budget(sid: str) -> str:
    return json.dumps(pt.budget(sess(sid)))


@tool("Look at an image of the session (path relative to the session dir, e.g. case/chars/X_sheet.png, "
      "case/paint/g2/f120.png) or of the character library $REMAKE_HOME/chars/.")
def view(sid: str, path: str) -> list:
    d = sess(sid)
    p = (Path(path) if Path(path).is_absolute() else d / path).resolve()
    assert p.is_relative_to(d.resolve()) or p.is_relative_to(CHARS.resolve()), "only files of the session or $REMAKE_HOME/chars"
    if not p.exists():  # codex guessed cmp_g1193.jpg for the group file cmp_g1193_1254.jpg
        near = difflib.get_close_matches(p.name, [q.name for q in p.parent.glob("*")], 6, 0.5) if p.parent.is_dir() else []
        raise FileNotFoundError(f"{path} does not exist" + (f"; in {p.parent.name}/: {', '.join(near)}" if near else ""))
    return [str(p), img(p, 1920)]


@tool("Write result.json (the deliverable): status done|partial|failed, final, seconds, cast, caption, picks, windows "
      "[{w, grade, note}], overall OK|MINOR|BAD, image_calls, changed, problems. sid/kind are filled in.")
def save_result(sid: str, result: dict) -> str:
    d = sess(sid)
    assert result.get("status") in ("done", "partial", "failed"), "status: done | partial | failed"
    if result["status"] != "failed":
        # twice codex read a long to-do list in `status`, tried to save a partial result and stopped
        assert Path(result.get("final") or "").is_file(), (
            "final: the absolute path of case/final/full_720.mp4, which `finish` writes. The job is not done: there is "
            "no time or token limit and nobody is waiting. Go on now with the next step `status` lists (paint, "
            "write_prompts, submit, wait, strip, finish) and save the result after `finish`.")
    r = {"sid": sid, **result, "kind": kind(d)}
    (d / "result.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    return f"{d / 'result.json'} written"


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
