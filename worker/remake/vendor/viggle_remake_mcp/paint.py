"""The keyed steps: character sheets, keyframe paints (swap sessions) and edit paints (edit sessions), all through
one `edit()`. Three backends:
  - an OpenAI-compatible images API (OPENAI_API_KEY, OPENAI_BASE_URL, REMAKE_IMAGE_MODEL, default gpt-image-2);
    quality "low" by default, "medium" when the call asks for it, never higher; cost estimated from the reported usage;
  - REMAKE_PAINT=codex: codex's own image_gen paints, in two calls of the same tool (codex_edit); no API, no spend;
  - REMAKE_PAINT=pool: the same image_gen, but in a helper `codex exec` this process starts per picture (pool_edit), so
    one tool call paints and paint_many runs several at once (image_gen is serial inside one codex session).
Gates, in order: $REMAKE_HOME/STOP (a quota error writes it: every session stops), the session's image-call limit
(REMAKE_MAX_IMAGES, default 24, or 80 in codex mode; a `paint_limit` file in the session overrides it), the session's
estimated spend (REMAKE_MAX_USD, default 10). Every call is a row of the session's ledger.jsonl.

Paint follows one recipe: identity from the character sheet plus the character's real photo (identity before
alignment), every cell says who is where, two frames per grid so a pair shares one look. Source frames and
keyframes are at the final canvas (1280x736 class), a request is that aspect at ~1.6 MP (`req`); the worker
shrinks a keyframe for drafts.
"""
import base64
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .common import ARM, CHARS, HOME, STOP, grid, jr, ledger, log, pick, spent, split, ticks

CODEX = os.environ.get("REMAKE_PAINT") == "codex"
POOL = os.environ.get("REMAKE_PAINT") == "pool"
# a helper's home per picture (auth.json copied in, ~10 MB of codex state, removed after); keep it on a local disk
POOL_DIR = Path(os.environ.get("REMAKE_POOL_DIR", HOME / "pool"))
# the helpers' login: REMAKE_CODEX_AUTH, else the codex running this server (codex_exec.sh copies one in), else ~/.codex
POOL_AUTH = Path(os.environ.get("REMAKE_CODEX_AUTH") or next(
    (p for p in (Path(os.environ.get("CODEX_HOME", "/nonexistent")) / "auth.json",) if p.exists()),
    Path.home() / ".codex" / "auth.json"))
POOL_MODEL = os.environ.get("REMAKE_POOL_MODEL", "gpt-6.1-terra")
POOL_N = int(os.environ.get("REMAKE_POOL_N", "8"))  # paint_many's parallel helpers
POOL_CFG = """approval_policy = "never"
sandbox_mode = "read-only"
suppress_unstable_features_warning = true
[features]
plugins = false
remote_plugin = false
apps = false
[skills.bundled]
enabled = false
"""
IMAGES = os.environ.get("REMAKE_CODEX_IMAGES")  # codex's generated_images dir: the import finds the PNG itself
MODEL = os.environ.get("REMAKE_IMAGE_MODEL", "gpt-image-2")
MAX_USD = float(os.environ.get("REMAKE_MAX_USD", "10"))
# a python with onnxruntime + opencv: paint then checks every painted cast face with facecheck.py (unset: no check)
FACE_PY = os.environ.get("REMAKE_FACE_PY")
FACE_SEM = threading.Semaphore(2)  # paint_many runs 8 paints at once; 2 checks (4 threads each) at a time
QUOTA = ("spending cap", "insufficient", "more credits", "quota", "billing", "exceeded your current", "402")

RULES = ("Never add a head, face or body that the source does not show: someone seen only as an arm, a sleeve or a "
         "shoulder stays only that arm, sleeve or shoulder, in the character's clothes. The hands and all visible skin of a "
         "replaced person take that character's own skin tone, and the original's rings, bracelets and chains go. ")
WHOLE = ("Whole-person replacement: the replaced person's face, hair, skin tone, body shape and clothes all become "
         "those of the reference character; nothing of the original person's identity or outfit remains (props they "
         "hold stay). ")
SOFT = ("Sharpness follows the source person by person: someone out of focus or motion-blurred in the source is "
        "painted just as soft or smeared, never sharper; only people sharp in the source are sharp. Someone partly "
        "hidden behind another person or an object stays exactly as hidden: show only the parts of them that the "
        "source shows, never more of their head or body. ")
KEEP = ("Keep everything else exactly as in the source: same camera framing and crop, same background, same lighting "
        "and colour grade, same depth of field, motion blur and grain. " + SOFT + "Every replaced person keeps the original's "
        "pose, head position and head size, gaze, expression, mouth shape and action, and anything the scene does to "
        "them (makeup or creature effects, wounds, light on the face). People not named stay untouched. ALIGNMENT IS "
        "THE TOP PRIORITY: body keypoints stay at the same pixel positions as in the source - head centre, head "
        "size, shoulder line, elbows, hands, torso lean; never enlarge a person or change the crop. Objects they "
        "touch or point at stay at the same spot. ")
# with a real photo: keeping the original's head size and shoulder line kept the original's face and build, so
# identity comes first
IDKEEP = ("Keep everything else exactly as in the source: same camera framing and crop, same background, same lighting "
          "and colour grade, same depth of field, motion blur and grain. " + SOFT + "People not named stay untouched. Every replaced "
          "person keeps the original's pose, head position, gaze direction, expression, mouth shape and action. "
          "IDENTITY IS THE TOP PRIORITY: each replaced person must be recognisable at a glance as the real person in "
          "their photo - their own face shape, eyes, brows, nose, jaw, hairline and hair, and their own head size, neck "
          "and body build, not the original's. Where the character is slimmer or smaller than the original, draw them so "
          "and fill the freed space with the background. Objects they touch or point at stay at the same spot. ")
KINDS = {
    "person": ("Character reference sheet of the person in the photo, as a photorealistic studio photograph on a plain "
               "light-grey background, soft even lighting. Left third: a large head-and-shoulders portrait facing the "
               "camera. Right two thirds: the same person three times full body, head to shoes visible, standing upright "
               "with arms relaxed - front view, three-quarter view, back view. Exactly the same real face, hairstyle, "
               "skin tone, age and body build as the photo in every view. Outfit in every view: {outfit}. No text."),
    "anthro": ("Character reference sheet, photorealistic studio photograph on a plain light-grey background, soft even "
               "lighting. The character is an anthropomorphic animal: the real animal head, fur and markings of the animal "
               "in the photo on an upright human-proportioned body with human posture, furry hands, fully dressed. Left "
               "third: a large head-and-shoulders portrait facing the camera. Right two thirds: the same character three "
               "times full body, head to shoes visible, standing upright with arms relaxed - front view, three-quarter "
               "view, back view. Exactly the same head, fur colour and markings in every view. Outfit in every view: "
               "{outfit}. No text."),
    "creature": ("Character reference sheet of the character in the picture, as a photorealistic studio photograph of a "
                 "life-size version of it on a plain light-grey background, soft even lighting. Left third: a large "
                 "head-and-shoulders portrait facing the camera. Right two thirds: the same character three times full "
                 "body, head to feet visible, standing upright - front view, three-quarter view, back view. Exactly the "
                 "same shape, colours, face and proportions as the picture in every view. Outfit/details: {outfit}. No text."),
}
TEXT_SHEET = ("Character reference sheet, photorealistic studio photograph on a plain light-grey background, soft even "
              "lighting. The character: {text}. Left third: a large head-and-shoulders portrait facing the camera. "
              "Right two thirds: the same character three times full body, head to shoes visible, standing upright - front "
              "view, three-quarter view, back view. Identical in every view. Outfit in every view: {outfit}. No text.")
EDIT = ("The last image is a frame from a video. Edit it: {edit} Change nothing else: every person keeps exactly the "
        "same face, hair, body, clothes, pose, hands and position; the same background, lighting, colours, camera framing, "
        "composition and image grain. Photorealistic: the same photograph with only that edit.")


class Brief(Exception):
    """Codex mode, first call: nothing painted; args[0] is the brief (step() turns it into what codex must do)."""


def limit(d):
    p = d / "paint_limit"
    return int(p.read_text()) if p.exists() else int(os.environ.get("REMAKE_MAX_IMAGES", 80 if CODEX or POOL else 24))


def calls(d):
    return sum(1 for r in ledger(d) if r.get("kind") == "image" and r.get("ok"))


def gpt(images, prompt, size, quality):
    from openai import OpenAI
    c = OpenAI(timeout=300, max_retries=0)  # OPENAI_API_KEY / OPENAI_BASE_URL from the environment
    if images:
        r = c.images.edit(model=MODEL, image=[open(p, "rb") for p in images], prompt=prompt, size=size, quality=quality)
    else:
        r = c.images.generate(model=MODEL, prompt=prompt, size=size, quality=quality)
    u = r.usage.model_dump() if r.usage else {}
    t = u.get("input_tokens_details") or {}
    # gpt-image list prices per 1M tokens: text in $5, image in $10, out $40; a backend that reports no usage counts $0.15
    usd = (t.get("text_tokens", 0) * 5 + t.get("image_tokens", 0) * 10 + u.get("output_tokens", 0) * 40) / 1e6 if u else 0.15
    return base64.b64decode(r.data[0].b64_json), usd


def edit(d, images, prompt, size, out, tag, quality="low"):
    """-> (PIL image, usd). images: absolute paths, the frame to edit last."""
    if STOP.exists():
        raise RuntimeError(f"PAINT STOPPED for every session ({STOP}): {STOP.read_text()}\nDo not retry. Report back.")
    lim = limit(d)
    if calls(d) >= lim:
        raise RuntimeError(f"this session used its {lim} image calls. Stop painting, finish with what you have, and say "
                           "so in result.json.")
    if CODEX:
        return codex_edit(d, images, prompt, size, out, tag)
    if POOL:
        return pool_edit(d, images, prompt, size, out, tag)
    if spent(d) + 0.3 > MAX_USD:
        raise RuntimeError(f"session image budget reached: ${spent(d):.2f} of ${MAX_USD:.2f} (REMAKE_MAX_USD). Stop painting "
                           "and report back.")
    for attempt in (0, 1):
        try:
            raw, usd = gpt(images, prompt, size, quality)
        except Exception as e:  # the ledger keeps failed calls; a quota error stops every session
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            log(d, kind="image", prov="api", tag=tag, usd=0.0, ok=False, error=msg)
            low = msg.lower()
            if any(q in low for q in QUOTA):
                STOP.parent.mkdir(parents=True, exist_ok=True)
                STOP.write_text(f"quota/credit error: {msg}")
                raise RuntimeError(f"quota/credit error: {msg}\nPainting is stopped for every session (delete {STOP} to "
                                   "resume). Report back.") from e
            if attempt == 0 and ("moderation" in low or "safety" in low):
                time.sleep(45)  # a moderation refusal of an ordinary edit is often a passing state; rewording does not help
                continue
            raise RuntimeError(f"image call failed: {msg}") from e
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.save(out)
        log(d, kind="image", prov="api", tag=tag, usd=round(usd, 4), ok=True)
        return im, usd


def codex_cards(d, prompt, images, tag):
    """image_gen takes at most 5 reference pictures (a 3-person paint = 3 sheets + 3 photos + the frame is refused).
    Past 5, pack each character's sheet and photo into one card (two characters a card past 4) and rewrite the head of
    paint's prompt to match; the rest of the prompt and the frame, last, are as is."""
    case = jr(d, "case.json")
    C = d / "case" / "chars"
    chars = [Path(x).name[:-len("_sheet.png")] for x in images if x.endswith("_sheet.png")]
    photos = [c for c in chars if any(x.endswith(f"/{c}_photo.png") for x in images)]
    head = " ".join(f"Image {k + 1} is the character sheet of {c} - {case['chars'][c]}." for k, c in enumerate(chars))
    head += "".join(f" Image {len(chars) + k + 1} is a real photo of {c}: copy {c}'s exact face and build from it."
                    for k, c in enumerate(photos))
    if not (tag.startswith("paint_") and prompt.startswith(head) and len(images) == len(chars) + len(photos) + 1):
        raise RuntimeError(f"{tag}: {len(images)} pictures, image_gen takes 5 and only a paint's can be packed into cards")

    def row(c):
        s = Image.open(C / f"{c}_sheet.png").convert("RGB")
        if c not in photos:
            return s
        p = Image.open(C / f"{c}_photo.png").convert("RGB")
        p = p.resize((round(p.width * s.height / p.height), s.height), Image.LANCZOS)
        r = Image.new("RGB", (s.width + 32 + p.width, s.height), "white")
        r.paste(s)
        r.paste(p, (s.width + 32, 0))
        return r

    def say(c):
        if c not in photos:
            return f"the character sheet of {c} - {case['chars'][c]}."
        return (f"{c}: on the left the character sheet of {c} - {case['chars'][c].rstrip('.')}; on the right a real photo of {c}: "
                f"copy {c}'s exact face and build from it.")

    per = 1 if len(chars) <= 4 else 2
    cards, text = [], []
    for k in range(0, len(chars), per):
        g = chars[k:k + per]
        rows = [row(c) for c in g]
        im = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 32 * (len(rows) - 1)), "white")
        y = 0
        for r in rows:
            im.paste(r, (0, y))
            y += r.height + 32
        cards.append(d / "inbox" / "cards" / f"{'+'.join(g)}.png")  # one file per group, not per paint (~7 MB each)
        cards[-1].parent.mkdir(parents=True, exist_ok=True)
        tmp = cards[-1].with_name(f".{tag}_{cards[-1].name}")
        im.save(tmp, "PNG")
        os.replace(tmp, cards[-1])
        n = len(cards)
        text.append(f"Image {n} shows {say(g[0])}" if len(g) == 1 else
                    f"Image {n} has two rows. Top row shows {say(g[0])} Bottom row shows {say(g[1])}")
    return " ".join(text) + prompt[len(head):], [str(c) for c in cards] + [images[-1]]


def codex_edit(d, images, prompt, size, out, tag):
    """REMAKE_PAINT=codex: the image comes from codex's image_gen, in two calls of the same tool. The first writes
    inbox/<tag>.json (prompt + pictures) and raises Brief; codex generates, copies the PNG to inbox/<tag>.png and calls
    again; that call takes the PNG if the prompt is still the brief's."""
    inbox = d / "inbox"
    got, brief = inbox / f"{tag}.png", inbox / f"{tag}.json"
    inbox.mkdir(exist_ok=True)
    # on a network filesystem a file written from another host can stay invisible (cached "no such file") for ~30 s;
    # reading the dir makes it visible at once
    os.listdir(inbox)
    old = json.loads(brief.read_text())["prompt"] if brief.exists() else None
    W, H = map(int, size.split("x"))
    if not got.exists() and old == prompt and IMAGES:  # no copy: take the one PNG image_gen saved since the brief
        taken = inbox / "used" / "codex_taken.txt"
        seen = set(taken.read_text().split()) if taken.exists() else set()
        new = [p for p in Path(IMAGES).glob("*/*.png") if p.stat().st_mtime > brief.stat().st_mtime and str(p) not in seen]
        if len(new) == 1:
            shutil.copy(new[0], got)
            taken.parent.mkdir(exist_ok=True)
            with open(taken, "a") as f:
                f.write(f"{new[0]}\n")
    if got.exists() and old == prompt:
        im = Image.open(got).convert("RGB")
        (inbox / "used").mkdir(exist_ok=True)
        keep = inbox / "used" / f"{tag}_{time.strftime('%H%M%S')}"
        got.rename(keep.with_suffix(".png"))
        brief.rename(keep.with_suffix(".json"))
        if abs(im.width / im.height - W / H) > 0.03 * W / H:
            raise RuntimeError(f"{got.name} is {im.width}x{im.height}, not the {W}x{H} aspect: nothing imported. Call again "
                               "for a new brief and generate again.")
        im.save(out)
        log(d, kind="image", prov="codex", tag=tag, usd=0.0, ok=True)
        return im, 0.0
    note = (f"{got.name} was there but its brief had a different prompt (or none): discarded, generate again. "
            if got.exists() else "")
    got.unlink(missing_ok=True)
    cprompt, refs = codex_cards(d, prompt, images, tag) if len(images) > 5 else (prompt, images)
    b = {"tag": tag, "size": size, "prompt": prompt, "copy_to": str(got), "note": note, "codex_prompt": cprompt + shape(W, H),
         "referenced_image_paths": [str(Path(x).resolve()) for x in refs]}
    brief.write_text(json.dumps(b, ensure_ascii=False))
    raise Brief(b)


def req(W, H, px=1664 * 960):
    """The request size: W:H at the image model's ~1.6 MP (what image_gen returns whatever is asked), multiples of 16."""
    s = (px / (W * H)) ** .5
    return round(W * s / 16) * 16, round(H * s / 16) * 16


def square_pair(d, fs):
    """A 2-frame grid near 1:1, which image_gen returns at its own 5:4 (a 2.06:1 picture stacked to 1.02:1: 13 of
    20 tries came back 1402x1122 and were dropped by the aspect check; 0.87:1 and 1.38:1 grids never failed in ~1500)."""
    if not all((d / "case" / "paint" / "src" / f"f{f}.png").exists() for f in fs):
        return False  # paint says what is missing
    W, H = jr(d, "meta.json")["final"]
    fr = [Image.open(d / "case" / "paint" / "src" / f"f{f}.png").convert("RGB").resize((W, H), Image.LANCZOS) for f in fs]
    box = inner(d, fr)
    w, h = box[2] - box[0], box[3] - box[1]
    return 0.92 < (w / (2 * h) if W >= H else 2 * w / h) < 1.12


def bars(ims):
    """The picture inside the letterbox / pillarbox the frames share, as a PIL box (left, top, right, bottom) at their
    size: a line is bar when 98 % of it is dark in every frame, and a bar under 2 % of the side is none. The image model
    paints the whole canvas whatever the source shows (on one letterboxed film 62 of 76 keyframes lost their
    bars and the video model then drew picture into the bar for 4 s), so a paint sees only the picture and gets the
    bars back. A dark scene passes this test too: use box(), which also asks the whole source."""
    a = np.stack([np.asarray(im.convert("L"), np.float32) for im in ims]).max(0)
    H, W = a.shape

    def run(L):
        k = 0
        while k < len(L) // 3 and np.percentile(L[k], 98) < 14:
            k += 1
        return k if k >= 0.02 * len(L) else 0
    return run(a.T), run(a), W - run(a.T[::-1]), H - run(a[::-1])


def vbars(d):
    """The source's own letterbox at the final canvas, as a box: per side the bar that 90 % of 48 frames spread over
    the source have - a real bar is there all along, a dark scene is not (a dark right third read as a
    115 px pillarbox frame by frame). Cached in case/bars.json."""
    p = d / "case" / "bars.json"
    if not p.exists():
        W, H = jr(d, "meta.json")["final"]
        fs = np.linspace(0, int((d / "case" / "nf").read_text()) - 1, 48).astype(int).tolist()
        got = pick(d / "case" / "cond720.mp4", fs, W, H)
        s = np.array([(l, t, W - r, H - b) for l, t, r, b in (bars([Image.fromarray(x)]) for x in got.values())])
        l, t, r, b = np.percentile(s, 10, axis=0).astype(int).tolist()
        p.write_text(json.dumps([l, t, W - r, H - b]))
    return tuple(json.loads(p.read_text()))


def inner(d, ims):
    """The picture of these frames (at the final canvas) that a paint sees: bars only where the whole source has them
    and these frames are dark there too (a fade, a title card)."""
    v, f = vbars(d), bars(ims)
    return min(v[0], f[0]), min(v[1], f[1]), max(v[2], f[2]), max(v[3], f[3])


def shape(W, H):
    # image_gen has no size argument and picks its own aspect; saying it in the prompt is enough
    s = "PORTRAIT image, taller than wide" if H > W else "LANDSCAPE image, wider than tall" if W > H else "SQUARE image"
    return f" OUTPUT SIZE: a {s}, exact aspect ratio {W}:{H} ({W / H:.3f}:1)."


def pool_edit(d, images, prompt, size, out, tag):
    """REMAKE_PAINT=pool: one helper `codex exec` (own CODEX_HOME, no MCP, read-only, low effort) calls image_gen once
    with the pictures and the prompt; its PNG is the picture. A refusal or a missing PNG is asked once more, then
    raised. Cost: one ~31K-in / ~1K-out request per attempt, in the ledger."""
    W, H = map(int, size.split("x"))
    cprompt, refs = codex_cards(d, prompt, images, tag) if len(images) > 5 else (prompt, images)
    refs = [str(Path(x).resolve()) for x in refs]
    msg = (f"Call your image generation tool exactly once, with referenced_image_paths = {json.dumps(refs)} and the "
           "prompt below verbatim (no rewording, nothing added or dropped). Do nothing else: no files, no shell, no "
           "commentary. If the tool refuses, reply REFUSED and the tool's message, and stop.\n\nPROMPT:\n"
           + cprompt + shape(W, H))
    auth = POOL_AUTH
    why = ""
    for attempt in (0, 1):
        h = POOL_DIR / f"{d.name}_{tag}_{time.strftime('%H%M%S')}_{os.getpid()}_{attempt}"
        (h / "home").mkdir(parents=True)
        (h / "ws").mkdir()
        shutil.copy(auth, h / "home" / "auth.json")
        (h / "home" / "config.toml").write_text(POOL_CFG)
        t0 = time.time()
        try:
            r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "-C", str(h / "ws"), "-m", POOL_MODEL, "-c",
                                'model_reasoning_effort="low"', "--json", msg], env={**os.environ, "CODEX_HOME": str(h / "home")},
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)
            png = sorted((h / "home" / "generated_images").glob("*/*.png"))
            im = Image.open(png[-1]).convert("RGB") if png else None
        finally:
            shutil.rmtree(h, ignore_errors=True)
        use = [json.loads(x)["usage"] for x in r.stdout.splitlines() if '"turn.completed"' in x]
        said = " ".join(json.loads(x)["item"].get("text", "") for x in r.stdout.splitlines() if '"agent_message"' in x)
        row = dict(kind="image", prov="pool", tag=tag, usd=0.0, s=round(time.time() - t0, 1), attempt=attempt,
                   **({k: use[-1].get(k, 0) for k in ("input_tokens", "cached_input_tokens", "output_tokens")} if use else {}))
        if im:
            if abs(im.width / im.height - W / H) > 0.03 * W / H:
                log(d, ok=False, error=f"aspect {im.width}x{im.height}", **row)
                why = f"the picture came back {im.width}x{im.height}, not the {W}x{H} aspect"
                continue
            im.save(out)
            log(d, ok=True, **row)
            return im, 0.0
        why = (said or r.stderr[-400:] or f"rc {r.returncode}, no picture").strip()[:400]
        log(d, ok=False, error=why, **row)
        if any(q in why.lower() for q in ("usage limit", "rate limit", "429", "quota")):
            STOP.parent.mkdir(parents=True, exist_ok=True)
            STOP.write_text(f"codex pool: {why}")
            raise RuntimeError(f"codex limit: {why}\nPainting is stopped for every session (delete {STOP} to resume). "
                               "Report back.")
    raise RuntimeError(f"{tag}: no picture after 2 tries ({why}). Nothing was painted. Call the same tool again as is, "
                       "later in the job; a refused sheet never turns a cast person into none.")


def step(b, tool_name):
    """The text codex gets for a brief."""
    if IMAGES:
        return ("CODEX IMAGE STEP - nothing is painted yet.\n" + (f"NOTE: {b['note']}\n" if b.get("note") else "") +
                f"Call image_gen__imagegen once with the BRIEF_JSON object below as its arguments, as is, then call "
                f"{tool_name} again with exactly the same arguments: that call finds the new PNG, imports it and returns "
                "the comparison to judge (no copy needed). AGENTS.md has the one-cell recipe.\n"
                "BRIEF_JSON " + json.dumps({"referenced_image_paths": b["referenced_image_paths"], "prompt": b["codex_prompt"]},
                                           ensure_ascii=False))
    return ("CODEX IMAGE STEP - nothing is painted yet.\n" + (f"NOTE: {b['note']}\n" if b.get("note") else "") +
            "1. Call image_gen__imagegen once with\n"
            f"   referenced_image_paths = {json.dumps(b['referenced_image_paths'])}\n"
            "   prompt = the PROMPT below, verbatim (no rewording, nothing added or dropped).\n"
            f"2. Copy the generated PNG (the generated_images/ path the tool reports) to {b['copy_to']}\n"
            f"3. Call {tool_name} again with exactly the same arguments: that call imports the picture and returns "
            "the comparison to judge.\n\nPROMPT:\n" + b["codex_prompt"])


def sheet(d, name, source, photo=None, text=None, kind="person"):
    """-> text. photo: a path inside the session dir."""
    C = d / "case"
    (C / "chars").mkdir(exist_ok=True)
    out = C / "chars" / f"{name}_sheet.png"
    assert re.fullmatch(r"[\w \-]+", name), f"{name!r}: character names are letters, digits, space, - and _ only"
    case = jr(d, "case.json")
    if source == "lib":
        assert (CHARS / f"{name}_sheet.png").exists(), f"no {CHARS}/{name}_sheet.png"
        shutil.copy(CHARS / f"{name}_sheet.png", out)
        txt = f"{out.relative_to(d)} copied from {CHARS}"
        for lib in sorted(CHARS.glob("lib_*.json")):  # optional {"chars": {NAME: desc}, "outfits": {NAME: outfit}}
            L = json.loads(lib.read_text())
            if name in L["chars"] and name not in case["chars"]:
                case["chars"][name], case.setdefault("outfits", {})[name] = L["chars"][name], L["outfits"][name]
                (C / "case.json").write_text(json.dumps(case, indent=1, ensure_ascii=False))
                txt += f"\ncase.json chars/outfits[{name}] filled from {lib.name}"
        if (CHARS / f"{name}_photo.png").exists():
            txt += "\nthe library photo goes into case/chars at paint time"
        return txt
    assert name in case.get("outfits", {}), f"case.json outfits[{name!r}] is missing: write the outfit (incl. shoes) first"
    if source == "text":
        _, usd = edit(d, [], TEXT_SHEET.format(text=text, outfit=case["outfits"][name]), "1536x1024", out, f"sheet_{name}")
    else:
        assert kind in KINDS, f"kind: {' | '.join(KINDS)}"
        p = (d / photo).resolve()
        assert p.is_relative_to(d.resolve()) and p.is_file(), f"{photo}: not a file of the session dir"
        src = C / "chars" / f"{name}_photo.png"
        im = Image.open(p).convert("RGB")
        im.thumbnail((1024, 1024))
        im.save(src)
        _, usd = edit(d, [str(src)], KINDS[kind].format(outfit=case["outfits"][name]), "1536x1024", out, f"sheet_{name}")
    return json.dumps({"out": str(out.relative_to(d)), "usd": round(usd, 4), "spent": round(spent(d), 3)})


def cell(f, people, case, where, note):
    wf = where[str(f)]
    sw = [i for i in wf if not i.startswith("_") and people[i].get("replace_with", "none") != "none"]
    keep = [i for i in wf if not i.startswith("_") and i not in sw]
    who = lambda i: f"{people[i]['desc']} (in this frame: {wf[i]})"  # noqa: E731
    s = "; ".join(f"replace {who(i)} with {c}, wearing {c}'s outfit from the character sheet ({case['outfits'][c]}) with "
                  "none of the original clothes, hat or sunglasses left" for i in sw for c in [people[i]["replace_with"]])
    return s + (". Leave unchanged: " + "; ".join(who(i) for i in keep) if keep else "") + ". " + wf.get("_note", "") + " " + note


def recut(im, rows):
    """A 2-frame grid re-cut at the seam the image model actually drew, each part scaled back to half, and where that
    seam was (fraction along the stacking axis), or (im, None) when it is at the middle or no full-width line shows.
    The model often draws the first cell short (24 of 400 grids, up to 7 % of the grid, always the first
    cell); a cut at the middle then carries the next frame's top rows into this frame's far edge, and the video model
    renders them as a band."""
    a = im if rows == 2 else im.transpose(Image.Transpose.TRANSPOSE)
    g = np.asarray(a.convert("L").resize((256, round(256 * a.height / a.width)), Image.Resampling.BOX), float)
    j = (np.abs(g[2:] - g[:-2]) > 24).mean(1)  # j[k]: share of columns that jump across row k + 1
    R = len(g); lo, hi, m = int(0.35 * R), int(0.65 * R), R // 2
    y = lo + int(np.argmax(j[lo - 1:hi - 1]))
    if j[y - 1] < 0.5 or j[m - 3:m + 2].max() >= 0.5 or abs(y - m) < 0.008 * R:
        return im, None
    s, h = round(y / R * a.height), a.height // 2
    b = Image.new(a.mode, a.size)
    b.paste(a.crop((0, 0, a.width, s)).resize((a.width, h), Image.LANCZOS), (0, 0))
    b.paste(a.crop((0, s, a.width, a.height)).resize((a.width, a.height - h), Image.LANCZOS), (0, h))
    return (b if rows == 2 else b.transpose(Image.Transpose.TRANSPOSE)), y / R


def paint(d, fs, note="", force=False, quality="low"):
    """-> {frames, chars, cmp, usd, spent}"""
    C = d / "case"
    plan, case, where = jr(d, "plan_grid.json"), jr(d, "case.json"), jr(d, "where.json", {})
    people = {p["id"]: p for p in plan["people"]}
    assert 1 <= len(fs) <= 2, "paint takes 1 or 2 frames per call (2 = one grid, the pair shares one look)"
    src, out = C / "paint" / "src", C / "paint" / ARM
    for f in fs:
        assert str(f) in where, f"f{f} is not in where.json: say who is where in it first"
        assert (src / f"f{f}.png").exists(), f"f{f}: run save_case (its check writes paint/src) first"
        assert force or not (out / f"f{f}.png").exists(), \
            f"f{f} is already painted; pass force=true to repaint (the old one goes to paint/{ARM}/old/)"
    chars = list(dict.fromkeys(people[i]["replace_with"] for f in fs for i in where[str(f)]
                               if not i.startswith("_") and people[i].get("replace_with", "none") != "none"))
    assert chars, f"nobody to replace in {fs} per where.json"
    for c in chars:
        assert (C / "chars" / f"{c}_sheet.png").exists(), (
            f"no character sheet for {c}: call sheet first" + (f" (source='lib': {CHARS} has one)" if (CHARS / f"{c}_sheet.png").exists()
            else "; a refused sheet is asked again as is, it never drops the character"))
        if not (C / "chars" / f"{c}_photo.png").exists() and (CHARS / f"{c}_photo.png").exists():
            shutil.copy(CHARS / f"{c}_photo.png", C / "chars" / f"{c}_photo.png")  # a library sheet's photo
    head = " ".join(f"Image {k + 1} is the character sheet of {c} - {case['chars'][c]}." for k, c in enumerate(chars))
    imgs = [str(C / "chars" / f"{c}_sheet.png") for c in chars]
    photos = [c for c in chars if (C / "chars" / f"{c}_photo.png").exists()]  # a sheet drawn from text has none
    head += "".join(f" Image {len(chars) + k + 1} is a real photo of {c}: copy {c}'s exact face and build from it."
                    for k, c in enumerate(photos))
    imgs += [str(C / "chars" / f"{c}_photo.png") for c in photos]
    keep = IDKEEP if photos else KEEP
    tag = "g" + "_".join(map(str, fs))
    W, H = jr(d, "meta.json")["final"]  # the keyframe is kept at the final canvas
    land = W >= H
    fr = [Image.open(src / f"f{f}.png").convert("RGB").resize((W, H), Image.LANCZOS) for f in fs]
    box = inner(d, fr)  # the image model sees the picture without the bars; they are pasted back below
    w, h = box[2] - box[0], box[3] - box[1]
    if len(fs) == 1:
        fr[0].crop(box).resize(req(w, h), Image.LANCZOS).save(out / f"{tag}_src.png")
        p = head + f" The last image is a frame from a video. Edit it: {cell(fs[0], people, case, where, note)} " + WHOLE + keep + RULES
        size, cols, rows = "%dx%d" % req(w, h), 1, 1
    else:
        cols, rows = (1, 2) if land else (2, 1)
        grid([x.crop(box) for x in fr], cols, None, (w, h)).save(out / f"{tag}_src.png")
        pos = ("top", "bottom") if land else ("left", "right")
        cells = " ".join(f"Cell {k + 1} ({pos[k]}): {cell(f, people, case, where, note)}" for k, f in enumerate(fs))
        p = (head + f" The last image is a grid of 2 frames from the same video, {cols} column(s) x {rows} row(s), each cell "
             f"{w}x{h}, no gutters. Edit every cell. {cells} " + WHOLE + keep + RULES
             + "Your output must be the EXACT same grid: same number of cells at the same positions with the same cell "
               "boundaries; each cell reproduces its own source frame as a strict layout template. The cells are frames "
               "of the SAME video: every replaced person must look IDENTICAL across all cells - same face, hair, clothes, "
               "colours and details.")
        size = "%dx%d" % req(cols * w, rows * h)
    im, usd = edit(d, imgs + [str(out / f"{tag}_src.png")], p, size, out / f"{tag}_raw.png", f"paint_{tag}", quality)
    old = out / "old"
    for f in fs:  # after the call: a codex brief or a failed call leaves the current paint in place
        if (out / f"f{f}.png").exists():
            old.mkdir(exist_ok=True)
            v = len(list(old.glob(f"f{f}_v*.png")))
            (out / f"f{f}.png").rename(old / f"f{f}_v{v}.png")
    im, cut = recut(im, rows) if len(fs) == 2 else (im, None)
    if cut:
        log(d, event="recut", tag=tag, seam=round(cut, 3))
    if im.size != (cols * w, rows * h):
        im = im.resize((cols * w, rows * h), Image.LANCZOS)
    for f, x, c in zip(fs, fr, split(im, cols, rows)):
        x.paste(c, box[:2])
        x.save(out / f"f{f}.png")
    (out / f"{tag}_prompt.txt").write_text(p)
    cmp = grid([Image.open(x / f"f{f}.png") for f in fs for x in (src, out)], 2,
               [f"{k} f{f}" for f in fs for k in ("src", "paint")], (624, round(624 * H / W)) if land else (round(480 * W / H), 480))
    cmp_path = C / "paint" / f"cmp_{tag}.jpg"
    cmp.save(cmp_path, quality=86)
    kept = kept_source(d, fs, people, where, src, out)
    return {"frames": fs, "chars": chars, "cmp": str(cmp_path.relative_to(d)), "usd": round(usd, 4), "spent": round(spent(d), 3),
            **({"recut": f"the image model drew the seam at {cut:.1%}, not 50 %: the cells were cut there"} if cut else {}),
            **({"kept_source_face": kept} if kept else {})}


def kept_source(d, fs, people, where, src, out):
    """The painted face most like each cast person's photo, scored against the photo (t) and the source face under it (s).
    Flags t + .1 < s, .4 <= s < .9 (an untouched face is >= .9), width >= 100 px: the image model kept the source
    person's face under the new hair and clothes. On 17 films it flagged 8 of 355 such faces, all in two films'
    weak keyframes; three of them, repainted one frame per call, went from .14-.20 to .56-.71. A 48-99 px face needs
    t >= .1 as well: there the face most like the photo is mostly an unreplaced bystander, a statue or the back of a
    head (13 of 15 rule hits on 337 such faces, all t <= .08), while the two real ones were half-painted (t .37 s .63
    w 96, the source face under the new hair, carried into the render; another t .14). The first, repainted
    alone with a feature note went to t .77 s .15. An untouched face (s >= .9) is a bystander unless the body under it
    was repainted: one keyframe had the source face, untouched (s .945, 181 px), on the new character's clothes (torso change
    32 of 255), while the 15 untouched bystanders among 1196 checked faces (s >= .9, >= 100 px) were <= 18."""
    if not FACE_PY:
        return []
    msg = []
    for f in fs:
        cs = dict.fromkeys(people[i]["replace_with"] for i in where[str(f)]
                           if not i.startswith("_") and people[i].get("replace_with", "none") != "none")
        ph = [f"{c}={d / 'case/chars' / f'{c}_photo.png'}" for c in cs if (d / "case/chars" / f"{c}_photo.png").exists()]
        if not ph:
            continue
        try:
            with FACE_SEM:
                r = subprocess.run([FACE_PY, str(Path(__file__).with_name("facecheck.py")), str(src / f"f{f}.png"),
                                    str(out / f"f{f}.png"), *ph], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            log(d, event="facecheck", frame=f, err=str(e)[-300:])
            continue
        res = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        log(d, event="facecheck", frame=f, res=res, **({"err": r.stderr[-300:]} if r.returncode else {}))
        for c, v in res.items():
            if v and ((v[2] >= 100 or v[2] >= 48 and v[0] >= .1) and .4 <= v[1] < .9 and v[1] > v[0] + .1
                      or v[2] >= 100 and v[1] >= .9 and len(v) > 3 and v[3] >= 25):
                msg.append(f"f{f} {c}: the painted face ({v[2]} px) is still the source person - cos {v[1]} to the source "
                           f"face, {v[0]} to {c}'s photo. Repaint this frame alone with a note that names how {c}'s face differs from it "
                           f"(face shape, cheeks, nose, eyes, brows, mouth) and which of its features must go: "
                           f"paint(frames=[{f}], force=true, note=...). At most two such repaints per frame.")
    return msg


def epaint(d, frame, text, refs=(), force=False, quality="low"):
    """Edit sessions: one keyframe = the image model's edit of the source frame by `text`; no sheets. refs (paths inside
    the session) are pictures of what the edit brings in (a mount, a prop), put before the frame."""
    C = d / "case"
    src, out = C / "paint" / "src" / f"f{frame}.png", C / "paint" / ARM
    assert src.exists(), f"f{frame}: put it in where.json and run save_case first (its check writes paint/src)"
    assert force or not (out / f"f{frame}.png").exists(), \
        f"f{frame} is already painted; pass force=true to repaint (the old one goes to paint/{ARM}/old/)"
    W, H = jr(d, "meta.json")["final"]
    im = Image.open(src).convert("RGB").resize((W, H), Image.LANCZOS)
    tag = f"e{frame}"
    box = inner(d, [im])
    w, h = box[2] - box[0], box[3] - box[1]
    im.crop(box).resize(req(w, h), Image.LANCZOS).save(out / f"{tag}_src.png")
    refs = [str((d / r).resolve()) for r in refs]
    p = "".join(f"Image {k + 1} is a reference picture of what the edit brings into the frame. " for k in range(len(refs)))
    p += EDIT.format(edit=text.strip())
    res, usd = edit(d, refs + [str(out / f"{tag}_src.png")], p, "%dx%d" % req(w, h), out / f"{tag}_raw.png", f"epaint_{tag}",
                    quality)
    if (out / f"f{frame}.png").exists():
        (out / "old").mkdir(exist_ok=True)
        v = len(list((out / "old").glob(f"f{frame}_v*.png")))
        (out / f"f{frame}.png").rename(out / "old" / f"f{frame}_v{v}.png")
    kf = im.copy()
    kf.paste(res.resize((w, h), Image.LANCZOS), box[:2])
    kf.save(out / f"f{frame}.png")
    (out / f"{tag}_prompt.txt").write_text(p)
    cmp = grid([im, Image.open(out / f"f{frame}.png")], 2, [f"src f{frame}", "edit"],
               (624, round(624 * H / W)) if W >= H else (round(480 * W / H), 480))
    cmp.save(C / "paint" / f"cmp_{tag}.jpg", quality=86)
    return {"frame": frame, "cmp": f"case/paint/cmp_{tag}.jpg", "usd": round(usd, 4)}


def take_paint(d, take, i, frame, chars, fix=None, box=None, quality="low"):
    """A keyframe cut from a finished take (frame i of `take` = video frame `frame`), repaired by `fix` inside `box` only:
    the take's own pixels everywhere else, so it agrees with what the render already drew and adding it does not snap
    the shot (two independent paints of one shot do). -> {frame, cmp, usd}"""
    C = d / "case"
    out = C / "paint" / ARM
    W, H = jr(d, "meta.json")["final"]
    fr = Image.fromarray(pick(take, [i], W, H)[i])
    tag = f"t{frame}"
    res, ed, usd = fr, fr, 0.0
    fb = box if fix else None  # outlined on the take in cmp: where the repair went
    if fix:
        fr.resize(req(W, H), Image.LANCZOS).save(out / f"{tag}_src.png")
        case = jr(d, "case.json")
        p = " ".join(f"Image {k + 1} is the character sheet of {c} - {case['chars'][c]}." for k, c in enumerate(chars))
        p += (f" The last image is one frame of a video. Repair only this: {fix.strip()} Everything else stays exactly as "
              "it is in the last image: the framing, every pose and position, faces, fur, clothes, lighting, colours and "
              "background.")
        ed, usd = edit(d, [str(C / "chars" / f"{c}_sheet.png") for c in chars] + [str(out / f"{tag}_src.png")], p,
                       "%dx%d" % req(W, H), out / f"{tag}_raw.png", f"take_{tag}", quality)
        ed = ed.resize((W, H), Image.LANCZOS)
        x0, y0, x1, y1 = box
        m = Image.new("L", (W, H), 0)
        m.paste(255, (round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
        res = Image.composite(ed, fr, m.filter(ImageFilter.GaussianBlur(12 * W / 832)))
        (out / f"{tag}_prompt.txt").write_text(p)
    s = Image.fromarray(pick(C / "cond720.mp4", [frame], W, H)[frame])
    box = inner(d, [s])
    if box != (0, 0, W, H):  # the source's bars, whatever the take drew there
        s.paste(res.crop(box), box[:2])
        res = s
    if (out / f"f{frame}.png").exists():
        (out / "old").mkdir(exist_ok=True)
        v = len(list((out / "old").glob(f"f{frame}_v*.png")))
        (out / f"f{frame}.png").rename(out / "old" / f"f{frame}_v{v}.png")
    res.save(out / f"f{frame}.png")
    cell = (624, round(624 * H / W)) if W >= H else (round(480 * W / H), 480)
    cmp = grid([ticks(fr.resize(cell, Image.LANCZOS), fb), ed, res], 3, [f"take f{frame}", "edit", "keyframe"], cell)
    cmp.save(C / "paint" / f"cmp_{tag}.jpg", quality=86)
    return {"frame": frame, "cmp": f"case/paint/cmp_{tag}.jpg", "usd": round(usd, 4)}


def budget(d):
    rows = [r for r in ledger(d) if r.get("kind") == "image"]
    return {"session_usd": round(spent(d), 3), "usd_cap": MAX_USD, "calls_ok": sum(bool(r.get("ok")) for r in rows),
            "calls_failed": sum(not r.get("ok") for r in rows), "stopped": STOP.read_text() if STOP.exists() else False,
            "session_calls": calls(d), "session_limit": limit(d),
            "paint_mode": "codex" if CODEX else f"pool ({POOL_MODEL} x{POOL_N})" if POOL else f"api ({MODEL})"}
