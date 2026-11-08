"""The render prompt of a window: six sections, as the prompt-writing guide (guide(), data/short_guide.md) describes, in
case/win<i>/prompt.txt.

Who writes it:
  - by default the client (the agent driving this MCP): write_prompts returns one brief per window - the facts of the
    window, the rules below, the guide and example paths, its pictures and contact sheet - and set_prompt stores the
    text the client wrote;
  - with REMAKE_LLM_MODEL set, the server, through an OpenAI-compatible chat API with images (REMAKE_LLM_BASE_URL /
    REMAKE_LLM_API_KEY, else OPENAI_BASE_URL / OPENAI_API_KEY), from the same brief.
Either way the prompt is stored with `fp`, the fingerprint of what it was written from (plan, window, ref paints, cast),
and stamp.json (render span + case.json), so submit refuses a prompt that no longer matches its paints.

case/win<i>/ also holds the window's cut: cond.mp4 (draft canvas), audio.m4a, sheet.jpg (contact sheet every 0.5 s)
and cut.json, the span it was cut from; a cut that no longer matches its window is wiped or re-cut before any prompt
is written for it, and the final-canvas cond720.mp4 goes with it.
"""
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from PIL import Image

from .common import ARM, FPS, STOP, contact, jr, kind, log, spent
from .kit import size
from .paint import MAX_USD

SECTIONS = ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:",
            "non_diegetic_music:"]
DATA = Path(__file__).resolve().parent / "data"
EXAMPLE, EDIT_EXAMPLE = DATA / "example_prompt.txt", DATA / "example_edit_prompt.txt"
LLM = os.environ.get("REMAKE_LLM_MODEL")
# REMAKE_LLM_MODEL=codex-pool: the server writes them with one helper `codex exec` per window (paint.py's pool
# helper: own CODEX_HOME, no MCP, read-only), from the same brief and pictures; free on the codex login
PPOOL = LLM == "codex-pool"
# the helper writes typographic punctuation the lint flags, and the agent cannot see the text to fix it (on one run two
# windows were written twice for it); folded on store
TO_ASCII = str.maketrans({"—": "-", "–": "-", "‑": "-", "−": "-", "‘": "'", "’": "'",
                          "“": '"', "”": '"', "…": "...", "×": "x", " ": " ", "→": "->"})
PROMPT_MODEL = os.environ.get("REMAKE_PROMPT_MODEL", "gpt-6.1-terra")
PROMPT_EFFORT = os.environ.get("REMAKE_PROMPT_EFFORT", "high")
# codex_exec.sh put the guide and examples at the end of AGENTS.md: re-read after every compaction, never resent
IN_AGENTS = os.environ.get("REMAKE_GUIDE_IN_AGENTS") == "1"
GUIDE_AT = "at the end of AGENTS.md" if IN_AGENTS else "at the top of the session's first write_prompts result"
# codex code mode keeps ~10K tokens (~40K chars) of one exec result and drops the middle: briefs past this many
# characters wait for the next call
BRIEF_CHARS = int(os.environ.get("REMAKE_BRIEF_CHARS", "32000"))


def guide():
    """The prompt-writing guide shipped in data/: a short summary of the six-section format."""
    return DATA / "short_guide.md"


RULES = (
    "Rules: describe the TARGET video in present tense, never edit instructions; every <Subject N> is bound to "
    "the picture(s) that show it and its descriptor must exclude every similar person in frame; "
    "retention_analysis: <Video 1> fully_preserved for shots, cuts (give the cut times), camera, motion, "
    "timing, background and unchanged people; each <Picture N> '(appears in [Shot K]): reference - a frame of "
    "the target video ...'; each <Subject N> fully_preserved with its look. Cut times use window time "
    "MM:SS.mmm. Burned-in subtitles and on-screen text stay: quote each one verbatim in the shot that shows it, "
    "even one that flashes for a few frames (read it off the contact sheet); text left unquoted comes back as "
    "garbled text. Never write 'dim' or 'dimly lit': it darkens the whole render. Plain ASCII punctuation.")
NO_ASR = ("No speech transcript is given and no spoken words are known. Where someone visibly talks or sings, write it "
          "as <d>[unclear]</d> with their speaker ID; never write or guess words.\n")


def fp(d, w):
    """What a window's prompt is written from: the plan's shots inside it and the people they or its cast name, the
    window, the bytes of its ref paints and its cast. Not the whole plan: a save_case that fixed another window's
    shot made every written prompt stale, and codex learnt to re-bind the unchanged prompt.txt (6 times in
    3 of 8 runs), which also clears a stale flag that is real."""
    C = d / "case"
    plan = jr(d, "plan_grid.json")
    sh, c = shots(d, plan, w), cast(d, plan, w)
    ids = {p for x in sh for p in x["people"]} | set(c[0])
    h = hashlib.sha1(json.dumps([sh, [p for p in plan["people"] if p["id"] in ids]], sort_keys=True).encode())
    h.update(json.dumps([w["render"], w["own"], bool(w.get("reflect")), w["refs"], c]).encode())
    for f in w["refs"]:
        h.update((C / "paint" / ARM / f"f{f}.png").read_bytes())
    return h.hexdigest()[:12]


def stamp(d, w):
    """What win<i>/'s prompt was written from besides fp: the render span and case.json."""
    return [w["render"], hashlib.sha1((d / "case" / "case.json").read_bytes()).hexdigest()[:12]]


def shots(d, plan, w):
    """The detector's shots inside the window, each with the plan shot holding its midpoint."""
    s, e = w["render"]
    last = w["own"][1] if w.get("reflect") else e  # a mirrored tail continues own's last shot
    b = [s] + [c for c in jr(d, "cuts.json") if s < c < last] + [e]
    return [dict(next(x for x in plan["shots"] if x["start"] <= (a + min(z, last)) // 2 < x["end"]), start=a, end=z)
            for a, z in zip(b, b[1:])]


def cast(d, plan, w):
    """-> (named, unpictured): {id: frames} of people to replace that the window holds (in its shots' frames inside
    `own`, so a stub of a neighbouring shot in the dropped head does not count, plus anyone the per-frame inventory
    sees in a ref frame), and those of them no ref frame shows."""
    swap = {p["id"] for p in plan["people"] if p["replace_with"] != "none"}
    inv = jr(d, "paint/inv.json")
    assert all(str(f) in inv for f in w["refs"]), f"refs {w['refs']} lack a paint inventory: run save_case"
    seen = {i for f in w["refs"] for i, v in inv[str(f)]["inv"].items() if v["visible"]} & swap
    o0, o1 = w["own"]
    n = {}
    for x in shots(d, plan, w):
        for i in set(x["people"]) & swap:
            n[i] = n.get(i, 0) + max(0, min(x["end"], o1) - max(x["start"], o0))
    in_own = {i: k for i, k in n.items() if k > 0}
    return {i: in_own.get(i, 0) for i in sorted(set(in_own) | seen)}, {i: k for i, k in sorted(in_own.items())
                                                                         if i not in seen}


def cut_key(w):
    return [w["render"], w["own"], bool(w.get("reflect"))]


def cut_window(d, i, w):
    """case/win<i>/{cond.mp4, audio.m4a, sheet.jpg, cut.json} for window w; drops a cond720.mp4 of an older cut."""
    C = d / "case"
    s, e = w["render"]
    z = w["own"][1]
    out = C / f"win{i}"
    out.mkdir(parents=True, exist_ok=True)
    # `reflect`: the frames past `own` (the next shot's first few, there only to reach 17n+5) become a mirror of own's
    # last frames, so the window has no stub of the next shot (a stub can pull the render's cut early and show the
    # next shot's unreplaced person)
    k = e - z if w.get("reflect") else 0
    fc = f"[0:v]select=between(n\\,{s}\\,{e - 1}),setpts=N/{FPS}/TB[v]"
    if k:
        fc = (f"[0:v]split[a][b];[a]select=between(n\\,{s}\\,{z - 1}),setpts=N/{FPS}/TB[x];"
              f"[b]select=between(n\\,{z - 1 - k}\\,{z - 2}),reverse,setpts=N/{FPS}/TB[y];[x][y]concat=n=2:v=1:a=0[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", C / "cond.mp4", "-filter_complex", fc, "-map", "[v]", "-r",
                    str(FPS), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", out / "cond.mp4"], check=True)
    # `mute`: source cuts whose last second the model hears as silence (mute_before_cuts). On one film the next
    # speaker's voice starts under the listener's still face, and every seed cut to the voice 10-21 frames early;
    # silence there: mean 14 frames early -> 7 over 4 cuts x 2 seeds. Only the model's audio: finish muxes case/audio.m4a
    m = "+".join(f"between(t,{max(c - FPS - s, 0) / FPS:.4f},{(c - s) / FPS:.4f})" for c in w.get("mute", []))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s / FPS:.4f}", "-i", C / "audio.m4a", "-t",
                    f"{(e - s) / FPS:.4f}", *(["-af", f"volume=enable='{m}':volume=0"] if m else []),
                    "-c:a", "aac", "-b:a", "192k", out / "audio.m4a"], check=True)
    contact(out / "cond.mp4", size(d, 480))[0].save(out / "sheet.jpg", quality=85)
    (out / "cond720.mp4").unlink(missing_ok=True)
    (out / "cut.json").write_text(json.dumps(cut_key(w)))
    return out


def cut_ok(d, i, w):
    p = d / "case" / f"win{i}" / "cut.json"
    return p.exists() and json.loads(p.read_text()) == cut_key(w)


def check_prompt(text, nrefs):
    pos = [text.find(x) for x in SECTIONS]
    assert pos[0] == 0 and -1 not in pos and pos == sorted(pos), "the six sections in order, starting with subject_definitions:"
    miss = [k + 1 for k in range(nrefs) if f"<Picture {k + 1}>" not in text]
    assert not miss, f"<Picture {miss}> never named: one per keyframe, in refs order"


def reference(example):
    """The guide and a format example, inlined: the client may run where the server's files are not."""
    return (f"=== GUIDE ===\n{guide().read_text()}\n\n=== FORMAT EXAMPLE (a different edit of a different clip - copy its "
            f"structure, tags and level of detail, not its content) ===\n{example.read_text()}\n\n")


def brief(d, i, w, client):
    """-> (task text, warnings, people to replace the plan puts outside the window). client=True: the text a client
    writes the prompt from (write_prompts puts the guide and example above it, the pictures and sheet after it);
    False: the server LLM's (guide and example inlined, the pictures attached in order, then the contact sheet)."""
    cfg, plan = jr(d, "case.json"), jr(d, "plan_grid.json")
    people = {p["id"]: p for p in plan["people"]}
    s, e = w["render"]
    fr = w["refs"]
    mmss = lambda f: f"00:{(f - s) / FPS:06.3f}"  # noqa: E731
    sh = shots(d, plan, w)
    named, unpictured = cast(d, plan, w)
    shot_txt = "\n".join(
        f"[Shot {k + 1}] frames {max(x['start'], s) - s}..{min(x['end'], e) - s - 1} of the window, starts at "
        f"{mmss(max(x['start'], s))}; people: "
        + ", ".join(f"{p} ({people[p]['desc']}; becomes {people[p]['replace_with']})" if p in named
                    else f"{p} ({people[p]['desc']}; unchanged)" if p in people else p
                    for p in x["people"]) + f"; {x['desc']}" for k, x in enumerate(sh))
    pics = "\n".join((f"<Picture {k + 1}> = the image returned as picture {k + 1} of this window" if client
                      else f"<Picture {k + 1}> = the image attached as picture {k + 1}")
                     + f": window frame {f - s} ({mmss(f)}), in "
                     f"[Shot {next(j for j, x in enumerate(sh) if x['start'] <= f < x['end']) + 1}]"
                     for k, f in enumerate(fr))
    who = "\n".join(f"- {people[p]['desc']} -> {people[p]['replace_with']}: {cfg['chars'][people[p]['replace_with']]}"
                    + (" (no reference picture shows this person; describe them from this text)"
                       if p in unpictured else "") for p in named)
    (W, H), (w0, h0) = size(d, 720), size(d, 480)
    head = (f"Write the six-section prompt for window {i} and store it with set_prompt(window={i}, text=...). Follow "
            f"the GUIDE {GUIDE_AT} exactly; the FORMAT EXAMPLE there is a "
            "different edit of a different clip - copy its structure, tags and level of detail, not its content. The "
            "text starts with "
            "'subject_definitions:' and ends with the non_diegetic_music section - no markdown fences, no comments.\n\n"
            if client else
            "Write the six-section rewrite prompt for one render of a video-editing model (full-reference mode). Follow "
            "the guide below exactly. Output ONLY the prompt text, starting with 'subject_definitions:' and ending with "
            "the non_diegetic_music section - no markdown fences, no comments.\n\n"
            + reference(EXAMPLE))
    task = (
        head + "=== THIS RENDER ===\n"
        f"<Video 1> = the source clip, {e - s} frames at {FPS} fps = {(e - s) / FPS:.3f} s, rendered at {W}x{H} "
        f"(drafts {w0}x{h0}) (contact sheet " + ("returned after this window's pictures" if client else "attached last")
        + ", frames labelled with window frame index and time). The target video is <Video 1> with these "
        "people replaced (whole-person replacement: face, hair, body and clothes):\n"
        + (who or "- nobody")
        + "\nEveryone else stays as in <Video 1>. Name only the characters listed here: no other character appears "
          "in this window.\nShots in this window (verify against the contact sheet; a "
          "shot shorter than ~10 frames at the start or end of the window is a stub of a neighbouring shot and "
          "still gets its own [Shot N]; two shots with the same description are one planner shot split by a "
          "real cut - describe each from the contact sheet):\n" + shot_txt + "\n"
        "Reference pictures (frames of the TARGET video, already repainted"
        + ("" if client else ", attached in this order before the contact sheet") + "):\n" + pics + "\n"
        + NO_ASR + RULES)
    warn = [f"{p} ({people[p]['desc']} -> {people[p]['replace_with']}) is in {k} of the window's own frames "
            f"by the plan but in no ref frame" for p, k in unpictured.items()]
    if "prefix_of" in w:  # a hop has no ref frame by design (its prefix carries the look): not a missing keyframe
        warn = [x for x in warn if not x.endswith("in no ref frame")]
    rest = sorted({p["id"] for p in plan["people"] if p["replace_with"] != "none"} - set(named))
    return task, warn, rest


def jpeg_part(path):
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=88)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def pool_ask(d, text, images, tag):
    """REMAKE_LLM_MODEL=codex-pool: one helper `codex exec` writes the prompt from the brief, the pictures attached in
    order (--image); its last message is the prompt. A reply without 'subject_definitions:' is asked once more."""
    from .paint import POOL_AUTH, POOL_CFG, POOL_DIR
    auth = POOL_AUTH
    said = ""
    for attempt in (0, 1):
        h = POOL_DIR / f"{d.name}_prompt_{tag}_{time.strftime('%H%M%S')}_{os.getpid()}_{attempt}"
        (h / "home").mkdir(parents=True)
        (h / "ws").mkdir()
        shutil.copy(auth, h / "home" / "auth.json")
        (h / "home" / "config.toml").write_text(POOL_CFG)
        t0 = time.time()
        try:
            r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "-C", str(h / "ws"), "-m", PROMPT_MODEL, "-c",
                                f'model_reasoning_effort="{PROMPT_EFFORT}"', "--json"] + [f"--image={p}" for p in images],
                               input=text + "\n\nDo not run commands or touch files: reply with the prompt text only.",
                               env={**os.environ, "CODEX_HOME": str(h / "home")}, capture_output=True, text=True, timeout=1200)
        finally:
            shutil.rmtree(h, ignore_errors=True)
        use = [json.loads(x)["usage"] for x in r.stdout.splitlines() if '"turn.completed"' in x]
        msgs = [json.loads(x)["item"].get("text", "") for x in r.stdout.splitlines() if '"agent_message"' in x]
        said = msgs[-1] if msgs else (r.stderr[-400:] or f"rc {r.returncode}, no reply")
        ok = "subject_definitions:" in said
        log(d, kind="llm", prov="pool", tag=f"prompt {tag}", usd=0.0, s=round(time.time() - t0, 1), attempt=attempt, ok=ok,
            **({k: use[-1].get(k, 0) for k in ("input_tokens", "cached_input_tokens", "output_tokens")} if use else {}))
        if ok:
            return said
        if any(q in said.lower() for q in ("usage limit", "rate limit", "429", "quota")):
            raise RuntimeError(f"codex limit: {said[:400]}. No prompt written for {tag}. Report back.")
    return said


def ask(text, images, d=None, tag=""):
    """One chat call: the task text, then the images in order."""
    if PPOOL:
        return pool_ask(d, text, images, tag)
    client = OpenAI(base_url=os.environ.get("REMAKE_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
                    api_key=os.environ.get("REMAKE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"), timeout=900)
    r = client.chat.completions.create(model=LLM, temperature=0.3, messages=[
        {"role": "user", "content": [{"type": "text", "text": text}] + [jpeg_part(p) for p in images]}])
    return r.choices[0].message.content or ""


def store(d, i, w, text, by):
    wd = d / "case" / f"win{i}"
    (wd / "prompt.txt").write_text(text.strip() + "\n")
    (wd / "prompt.json").write_text(json.dumps({"fp": fp(d, w), "by": by}))
    (wd / "stamp.json").write_text(json.dumps(stamp(d, w)))


def view(d, f):
    """The keyframe at the draft canvas, for a prompt writer: the kept one is the final canvas, ~2.4x the image tokens
    for nothing a prompt needs."""
    p, v = d / "case" / "paint" / ARM / f"f{f}.png", d / "case" / "look" / "kf" / f"f{f}.jpg"
    if not v.exists() or v.stat().st_mtime < p.stat().st_mtime:
        v.parent.mkdir(parents=True, exist_ok=True)
        Image.open(p).convert("RGB").resize(tuple(size(d, 480)), Image.LANCZOS).save(v, quality=92)
    return v


def write_prompts(d, windows=None, guide_again=False, notes=""):
    """Swap sessions. -> (text, [image paths]). Cuts each window first if its cut is missing or stale (a stale win<i>/
    is wiped whole: its prompt was written for another span). The guide and example (~30K chars) come with the
    session's first call only, or with guide_again."""
    assert kind(d) == "swap", "write_prompts is for swap sessions; edit sessions: cut_windows, then edit_prompt"
    ws = jr(d, "refs.json")["windows"]
    todo = [i for i in (windows or range(len(ws))) if not ws[i].get("passthrough")]
    bare = [f"w{i}" for i in todo if not ws[i]["refs"] and "prefix_of" not in ws[i]]  # a hop needs none
    assert not bare, f"{' '.join(bare)}: no painted keyframe yet (paint, then windows again)"
    unpainted = [f"w{i} f{f}" for i in todo for f in ws[i]["refs"] if not (d / f"case/paint/{ARM}/f{f}.png").exists()]
    assert not unpainted, f"keyframes not painted yet: {', '.join(unpainted)}"
    for i in todo:
        wd = d / "case" / f"win{i}"
        if not cut_ok(d, i, ws[i]):
            if wd.exists():
                shutil.rmtree(wd)
            cut_window(d, i, ws[i])
    jobs = {i: brief(d, i, ws[i], client=not LLM) for i in todo}
    warn = [f"warning: {x}" for i in todo for x in jobs[i][1]]
    pics = {i: [view(d, f) for f in ws[i]["refs"]] for i in todo}
    if not LLM:
        sent = d / "case" / "guide_sent"
        if guide_again or not (IN_AGENTS or sent.exists()):
            top = reference(EXAMPLE)
            sent.touch()
        else:
            top = (f"(GUIDE and FORMAT EXAMPLE: {GUIDE_AT}, not repeated here"
                   + ("" if IN_AGENTS else "; call write_prompts(guide=true) to get them again if they are no longer in "
                      "your context") + ".)\n")
        out, images, left = [top] + warn, [], []
        for i in todo:
            b = (f"\n===== brief for window {i} (render {ws[i]['render']}, refs {ws[i]['refs']}; not named, the plan "
                 f"puts them outside this window: {jobs[i][2]}) =====\n{jobs[i][0]}")
            if left or (images and sum(map(len, out)) + len(b) > BRIEF_CHARS):  # codex keeps ~10K tokens of one result
                left.append(i)
                continue
            out.append(b)
            images += pics[i] + [d / "case" / f"win{i}" / "sheet.jpg"]
        out.append(f"\n{len(todo) - len(left)} briefs. Write each prompt yourself and store it with set_prompt; the images "
                   "returned are, per window, its <Picture k> in order, then its contact sheet.")
        if left:  # said at both ends: a cut result keeps its head and tail
            note = (f"NOT RETURNED (one result holds about {BRIEF_CHARS} characters of briefs): windows {left}. "
                    f"Call write_prompts(windows={left}) for them.")
            out = [note] + out + [note]
        return "\n".join(out), images
    assert not STOP.exists(), f"STOPPED: {STOP.read_text()} (delete {STOP} to resume)"
    assert spent(d) + 0.003 * len(todo) <= MAX_USD, f"session spend {spent(d):.2f} USD would pass REMAKE_MAX_USD {MAX_USD}"

    def write(i):  # the windows are independent: one LLM call each, all at once
        w = ws[i]
        task = jobs[i][0] + (f"\n=== NOTES FROM THE AGENT RUNNING THIS JOB (apply them) ===\n{notes}\n" if notes else "")
        try:
            txt = ask(task, pics[i] + [d / "case" / f"win{i}" / "sheet.jpg"], d, f"w{i}").strip()
        except Exception as e:
            return f"== win{i} FAILED: {e}"
        if "subject_definitions:" not in txt:  # find() = -1 would keep only the last character
            return (f"== win{i} FAILED: the rewriter returned no 'subject_definitions:' (nothing written; call "
                    f"write_prompts again for this window): {txt[:300]!r}")
        txt = txt[txt.find("subject_definitions:"):].translate(TO_ASCII)
        miss =[k + 1 for k in range(len(w["refs"])) if f"<Picture {k + 1}>" not in txt]
        store(d, i, w, txt, "write_prompts")
        return (f"== win{i} render {w['render']} refs {w['refs']} chars {len(txt)} missing pictures {miss}"
                f"\n  not named (the plan puts them outside this window; check the sheet): {jobs[i][2]}")
    with ThreadPoolExecutor(8) as ex:
        lines = list(ex.map(write, todo))
    if not PPOOL:  # pool_ask logs each helper call itself
        log(d, kind="llm", tag=f"prompts x{len(todo)}", model=LLM, usd=round(0.003 * len(todo), 4))
    out = warn + lines
    for i in todo:
        p = d / "case" / f"win{i}" / "prompt.txt"
        if p.exists() and PPOOL:  # the text stays out of the agent's context: the lint result is what it acts on
            lw = lint(d, ws[i], p.read_text())
            out.append(f"  w{i} lint: " + (" | ".join(lw) if lw else "clean"))
        elif p.exists():
            out.append(f"\n===== case/win{i}/prompt.txt =====\n" + p.read_text())
    if PPOOL:
        out.append(f"\nStored (prompt pool; each is case/win<i>/prompt.txt under {d}). To change one, call write_prompts "
                   "again for that window with notes= saying what to fix; edit_prompt replaces a text you wrote yourself.")
    return "\n".join(out), []


def set_prompt(d, i, text):
    """Store a prompt the client wrote for window i (either session kind) and bind it to the window, like the server
    path, so submit takes it."""
    w = jr(d, "refs.json")["windows"][i]
    assert not w.get("passthrough"), f"w{i} is a passthrough window: nothing is rendered there"
    assert cut_ok(d, i, w), (f"w{i} is not cut for its current span: "
                             + ("write_prompts" if kind(d) == "swap" else "cut_windows") + f" [{i}] first")
    unpainted = [f for f in w["refs"] if not (d / f"case/paint/{ARM}/f{f}.png").exists()]
    assert not unpainted, f"w{i}: keyframes {unpainted} are not painted yet"
    check_prompt(text.strip(), len(w["refs"]))
    store(d, i, w, text, "client")
    return f"case/win{i}/prompt.txt written and bound to w{i}" + "".join(f"\nwarning: {x}" for x in lint(d, w, text))


def lint(d, w, text):
    """Warnings, never refusals: 'dim', non-ASCII, and in a swap the brief's binding rules a terse prompt drops. One run
    folded 4-5 people into one '<Subject 2> is the ensemble ...' in 14 of 22 windows and named characters the
    window does not hold in 14; three later runs did neither."""
    warn = [x for x, bad in (("'dim' darkens the whole render", re.search(r"\bdim(ly)?\b", text, re.I)),
                             ("non-ASCII characters (plain ASCII punctuation, outside <d> dialogue)",
                              not text.isascii())) if bad]
    if kind(d) != "swap":
        return warn
    plan = jr(d, "plan_grid.json")
    to = {p["id"]: p["replace_with"] for p in plan["people"] if p["replace_with"] != "none"}
    here = sorted({to[p] for p in cast(d, plan, w)[0]})
    subs = len(set(re.findall(r"<Subject (\d+)>", text)))
    if subs < len(here):
        warn.append(f"{len(here)} characters in this window ({', '.join(here)}) but {subs} <Subject N>: give each "
                    "its own <Subject N>, bound to the pictures that show it")
    out = sorted(n for n in set(to.values()) - set(here) if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text))
    if out:
        warn.append(f"names {', '.join(out)}, who are not in this window: name only the characters the brief lists")
    ret = text.split("retention_analysis:")[-1].split("detailed_description:")[0]
    nap = [k + 1 for k in range(len(w["refs"])) if not re.search(rf"<Picture {k + 1}>[^\n]*\[Shot \d+\]", ret)]
    if nap:
        warn.append(f"<Picture {nap}> not tied to a shot in retention_analysis: write '<Picture N> (appears in "
                    "[Shot K]): reference - a frame of the target video ...'")
    return warn


def edit_prompt(d, i, text):
    """Edit sessions: set_prompt. Swap sessions: replace a written prompt's text, keeping its binding (as before)."""
    if kind(d) == "edit":
        return set_prompt(d, i, text)
    w = jr(d, "refs.json")["windows"][i]
    check_prompt(text.strip(), len(w["refs"]))
    assert (d / f"case/win{i}/prompt.json").exists(), f"write_prompts [{i}] first: it binds the window and cuts it"
    (d / f"case/win{i}/prompt.txt").write_text(text.strip() + "\n")
    return f"case/win{i}/prompt.txt replaced; submit takes it as is (a later write_prompts / set_prompt overwrites it)"


def cut_windows(d, windows=None):
    """Edit sessions (swap sessions may use it too): (re)cut case/win<i>/ to write the prompt from. Never touches
    prompt.txt. -> (text, [sheet paths])"""
    ws = jr(d, "refs.json")["windows"]
    todo = windows or [i for i, w in enumerate(ws) if not w.get("passthrough")]
    out = [("(GUIDE and edit-session FORMAT EXAMPLE: at the end of AGENTS.md.) " if IN_AGENTS else reference(EDIT_EXAMPLE))
           + "Write each window's prompt by the GUIDE, in the structure of the FORMAT EXAMPLE."]
    for i in todo:
        cut_window(d, i, ws[i])
        out.append(f"w{i}: render {ws[i]['render']} ({(ws[i]['render'][1] - ws[i]['render'][0]) / FPS:.2f} s), "
                   f"<Picture k> = refs {ws[i]['refs']} in this order -> case/win{i}/sheet.jpg")
    return "\n".join(out), [d / "case" / f"win{i}" / "sheet.jpg" for i in todo]
