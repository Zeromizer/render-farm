"""Kinetograph: draws a HyperFrames job about 2.5x faster, when it can.

Kinetograph is a Chrome-free Rust/GPU renderer (PolyForm Small Business licence).
We run our own fork, https://github.com/0xS4N4TG/kinetograph plus fx-library
spikes/kinetograph/fork/0001-0024, packed on the PC and installed with
  npm install <dist>\\*.tgz react@19.3.1 react-dom@19.3.1 --ignore-scripts

This is not an engine of its own: a job stays engine `hyperframes`, and
runners/hyperframes.py calls render() before it builds its own command. None
back from render() means HyperFrames draws the job, exactly as before, so every
way out of here (not eligible, no room, warned, crashed, hung, wrong output) ends
in the render the job would have had anyway. Only a cancel goes through.

Settings (.env; a worker restart reads them):
  KINETOGRAPH_CLI         the fork's dist/cli.js. Unset: this module does nothing.
  KINETOGRAPH_ROUTE       off (default) | test (only jobs with
                          params.renderer="kinetograph") | on (every eligible job)
  params.renderer         "hyperframes" keeps one job off this path in any mode

Who stays on HyperFrames, from the replay of 30 delivered videos (2026-10-31):
WebGL, WebGPU, three.js and iframes (not what this renderer was proved on); FX
modules that draw with those or with putImageData; data-fx-chain audio effects
(flaky on Windows, crash the worker's renders under repeats); full-frame
putImageData (correct but 2.4x slower); mix-blend-mode on a 3D layer (a known
silent failure in the fork's compositor); opacity or a filter on ::before and
::after (the fork draws the layer plain, with no warning); inline-block words after a
<br> (the fork loses the spaces between them, with no warning); emoji in a composition that names no
colour emoji font (the fork would draw its bundled monochrome one, where Chrome
draws colour, and --strict cannot see that); CJK text with no CJK font and vertical
text (--strict refuses both; this only saves the wasted attempt). `--strict` then
refuses whatever the
renderer knows it cannot draw. It cannot catch what it does not know about, which
is why the routing comes first and _check_output() comes after.

Job shapes never replayed stay on HyperFrames too. The CLI has no --fps and
no alpha output; resolution and variables have no equivalent; `gpu` asks for
NVENC; a non-root entry resolves its assets from another directory.

Third-party code: the render gets an allowlisted environment, because the
worker's own carries the Supabase service key.
"""
import json
import os
import re
import shutil
import subprocess
import time

import config
import locks
import proc

MIN_VERSION = (0, 8, 155)  # the HyperFrames behaviour the replay compared against
DRAFT_CRF = 28             # what `hyperframes render --quality draft` encodes at
STANDARD_CRF = 16          # what the runner sends a 0.8.155 mp4 (see hyperframes.py)

_ALLOW = ("PATH", "PATHEXT", "SystemRoot", "SystemDrive", "WINDIR", "COMSPEC", "USERPROFILE", "HOMEDRIVE",
          "HOMEPATH", "LOCALAPPDATA", "APPDATA", "ProgramData", "ProgramFiles", "ProgramFiles(x86)",
          "ProgramW6432", "NUMBER_OF_PROCESSORS", "OS", "PROCESSOR_ARCHITECTURE")
_PROGRESS_RE = re.compile(r"rendering frame (\d+)/(\d+)")
_TEXT_EXT = (".html", ".htm", ".js", ".css")
_SKIP_DIRS = {".git", "node_modules", "renders", "snapshots", ".hyperframes"}
_SKIP_FILES = {"gsap.min.js"}
_MAX_TEXT_BYTES = 3_000_000

# The bundle (fx-kit.js) carries every module's source, so it is read separately:
# what it holds says nothing about what this composition calls.
_DEFINE_RE = re.compile(r"FX\.define\(\s*[\"']([A-Z]{2,4}-\d{2})")
_GL_RE = re.compile(r"FXKit\.gl\(|getContext\(\s*['\"](?:webgl2?|experimental-webgl|webgpu)")
_PIXELS_RE = re.compile(r"putImageData")
_THREE_RE = re.compile(r"\bTHREE\.|three(?:\.module|\.min)?\.js|from\s+['\"]three['\"]")
_WEBGPU_RE = re.compile(r"shaders\.iife")
_BLEND_RE = re.compile(r"mix-blend-mode|mixBlendMode")
# A ::before drew opaque and sharp with opacity 0.38 or 0 and any filter, no warning; the same layer as
# a real element matched Chrome, SVG filter included (4e8e6e5e, fx-library blurtest.py, 2026-10-31).
_PSEUDO_EFFECT_RE = re.compile(r"::?(?:before|after)\b[^{}]*\{[^{}]*?(?<!\w)(?:opacity|filter|mix-blend-mode)\s*:", re.I)
# Past a <br>, the space between two inline-block words is lost, no warning: "isfullmostweeks," in
# 722fa49b, on 1553b79f and ecce9678. Plain spans, a div per line and &nbsp; drew right (build26.py spaces).
_WORD_GAP_RE = re.compile(r"<br\s*/?>(?:(?!</(?:div|p|h[1-6])\b).){0,4000}?</(?:span|b|i|em|strong)>\s+<(?:span|b|i|em|strong)\b", re.I | re.S)
_INLINE_BLOCK_RE = re.compile(r"display\s*:\s*inline-block", re.I)
_3D_RE = re.compile(r"preserve-3d|perspective\s*:|perspective\(|transformPerspective")
# Written out, or as the entity or escape a builder often uses (13c65c26 had &#x1F447;).
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
                       r"|&#x0*(?:1F[0-9A-F]{3}|2[67B][0-9A-F]{2}|FE0F);|&#1[23]\d{4};"
                       r"|\\u\{?(?:1F[0-9A-F]{3}|D83[C-E])", re.I)
# In a font-family, not just a link: Chrome only draws through the font the text names.
_COLOUR_EMOJI_FONT_RE = re.compile(r"font-family\s*:[^;}]*Noto[ +]Color[ +]Emoji", re.I)
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF]")
_CJK_FONT_RE = re.compile(r"Noto[ +](?:Sans|Serif)[ +](?:SC|TC|HK|JP|KR)\b|Noto[ +](?:Sans|Serif)[ +]CJK|Source[ +]Han|LXGW|ZCOOL|Ma[ +]Shan[ +]Zheng", re.I)
_VERTICAL_RE = re.compile(r"writing-mode\s*:\s*(?:vertical|sideways|tb)", re.I)


def _at_least(ver, want):
    try:
        return tuple(int(x) for x in re.split(r"[.+-]", ver)[:3]) >= want
    except ValueError:
        return False


def enabled(params):
    """Whether this job may try Kinetograph at all, before anything is read."""
    if not config.KINETOGRAPH_CLI or params.get("renderer") == "hyperframes":
        return False
    if config.KINETOGRAPH_ROUTE == "on":
        return True
    return config.KINETOGRAPH_ROUTE == "test" and params.get("renderer") == "kinetograph"


def crf_for(params, quality):
    """The CRF to encode at, or None for a quality with no agreed value."""
    if params.get("crf") is not None:
        return int(params["crf"])
    return {"draft": DRAFT_CRF, "standard": STANDARD_CRF}.get(quality)


def shape_reasons(params, entry, version, quality):
    """Why a job's parameters keep it on HyperFrames ([] when they don't)."""
    why = []
    if (params.get("format") or "mp4") != "mp4":
        why.append(f"format {params.get('format')}")
    for key in ("fps", "resolution", "variables", "gpu", "project_dir"):
        if params.get(key):
            why.append(key)
    if os.path.dirname(entry.replace("\\", "/")):
        why.append("entry outside the project root")
    if not _at_least(version, MIN_VERSION):
        why.append(f"hyperframes {version}")
    if crf_for(params, quality) is None:
        why.append(f"quality {quality} without crf")
    return why


def read_project(project_dir):
    """(composition text, FX bundle text), the way a router has to see them."""
    parts, bundle = [], []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            path = os.path.join(root, name)
            if not name.lower().endswith(_TEXT_EXT) or name in _SKIP_FILES:
                continue
            try:
                if os.path.getsize(path) > _MAX_TEXT_BYTES:
                    continue
                with open(path, encoding="utf-8", errors="replace") as f:
                    (bundle if "fx-kit" in name else parts).append(f.read())
            except OSError:
                continue
    return "\n".join(parts), "\n".join(bundle)


def hf_modules(bundle):
    """FX module ids whose own source needs HyperFrames.

    Read from the bundle the job carries rather than kept as a list here: a module
    added to the library next month is then routed correctly on its first use."""
    marks = list(_DEFINE_RE.finditer(bundle))
    found = set()
    for i, m in enumerate(marks):
        block = bundle[m.start(): marks[i + 1].start() if i + 1 < len(marks) else len(bundle)]
        if _GL_RE.search(block) or _PIXELS_RE.search(block):
            found.add(m.group(1))
    return found


def content_reasons(text, bundle):
    """Why a composition's content keeps it on HyperFrames ([] when it doesn't)."""
    why = []
    if _GL_RE.search(text):
        why.append("webgl")
    if _WEBGPU_RE.search(text):
        why.append("webgpu")
    if _THREE_RE.search(text):
        why.append("three")
    if "<iframe" in text.lower():
        why.append("iframe")
    if "data-fx-chain" in text:
        why.append("data-fx-chain")
    if _PIXELS_RE.search(text):
        why.append("putImageData")
    if _BLEND_RE.search(text) and _3D_RE.search(text):
        why.append("blend mode with 3D")
    if _PSEUDO_EFFECT_RE.search(text):
        why.append("effect on a pseudo-element")
    if _WORD_GAP_RE.search(text) and _INLINE_BLOCK_RE.search(text):
        why.append("word spans after a line break")
    if _EMOJI_RE.search(text) and not _COLOUR_EMOJI_FONT_RE.search(text):
        why.append("emoji without a colour emoji font")
    if _CJK_RE.search(text) and not _CJK_FONT_RE.search(text):
        why.append("CJK text without a CJK font")
    if _VERTICAL_RE.search(text):
        why.append("vertical text")
    # By mention, not by call syntax: the replay's call regex missed a TRAN-09 that
    # the delivery record shows. A false match here only costs the speed.
    used = sorted(hf_modules(bundle) & set(re.findall(r"\b[A-Z]{2,4}-\d{2}\b", text)))
    if used:
        why.append("fx " + ",".join(used))
    elif "FX." in text and not bundle:
        why.append("fx calls without a bundle to read")
    return why


def room():
    """(ok, reason). This renderer holds more VRAM than Chrome does (peak 3.9 GB over
    the idle box against 2.2), and light_lane's bars were set for Chrome."""
    import light_lane
    vram = light_lane._free_vram_mib()
    if vram is not None and vram < config.KINETOGRAPH_MIN_FREE_VRAM_MIB:
        return False, f"free VRAM {vram} MiB < {config.KINETOGRAPH_MIN_FREE_VRAM_MIB}"
    commit = light_lane._commit_free_gb()
    if commit < config.KINETOGRAPH_MIN_COMMIT_GB:
        return False, f"commit headroom {commit:.1f} GB < {config.KINETOGRAPH_MIN_COMMIT_GB:g}"
    return True, ""


def _env(tmp_dir, ffmpeg_dir):
    env = {k: os.environ[k] for k in _ALLOW if k in os.environ}
    env["TEMP"] = env["TMP"] = env["TMPDIR"] = tmp_dir
    if ffmpeg_dir:
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    return env


def expected_seconds(html):
    """The root composition's data-duration, or None when the file doesn't state one."""
    tag = re.search(r"<[^>]*\bdata-composition-id\b[^>]*>", html)
    m = tag and re.search(r"\bdata-duration=[\"']([\d.]+)[\"']", tag.group(0))
    return float(m.group(1)) if m else None


def _check_output(path, html, ffmpeg_dir):
    """A wrong file that exits 0 must not reach a client: name what's wrong, or None."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return "no output file"
    probe = os.path.join(ffmpeg_dir, "ffprobe.exe" if os.name == "nt" else "ffprobe") if ffmpeg_dir else "ffprobe"
    r = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    info = json.loads(r.stdout or "{}")
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    if "video" not in kinds:
        return "no video stream"
    if re.search(r"<audio\b", html) and "audio" not in kinds:
        return "composition has audio, output has none"
    want = expected_seconds(html)
    got = float((info.get("format") or {}).get("duration") or 0)
    if want and abs(got - want) > 0.5:
        return f"duration {got:.2f} s, composition says {want:g} s"
    return None


def render(job, project_dir, entry, version, quality, out_path, tmp_dir, ffmpeg_dir, heartbeat, log, run_kw):
    """Draw the job to out_path and return it, or None to leave it to HyperFrames."""
    params = job.get("params") or {}
    if not enabled(params):
        return None
    if not os.path.exists(config.KINETOGRAPH_CLI):
        log(f"kinetograph: {config.KINETOGRAPH_CLI} is missing, using hyperframes")
        return None

    text, bundle = read_project(project_dir)
    why = shape_reasons(params, entry, version, quality) + content_reasons(text, bundle)
    if why:
        log(f"kinetograph: hyperframes keeps this job ({'; '.join(why)})")
        return None
    with locks.limited("kinetograph", config.KINETOGRAPH_PARALLEL):
        # In here, so a render that was waiting reads the box as it is now.
        ok, reason = room()
        if not ok:
            log(f"kinetograph: hyperframes keeps this job ({reason})")
            return None
        return _draw(job, params, project_dir, entry, quality, out_path, tmp_dir, ffmpeg_dir, heartbeat, log, run_kw)


def _draw(job, params, project_dir, entry, quality, out_path, tmp_dir, ffmpeg_dir, heartbeat, log, run_kw):
    entry_path = os.path.join(project_dir, entry)
    cli = config.KINETOGRAPH_CLI
    tool = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cli))))  # <tool>\node_modules\kinetograph\dist\cli.js
    cmd = [shutil.which("node") or "node", cli, "render", entry_path, "-o", out_path, "--crf", str(crf_for(params, quality)),
           "--concurrency", str(config.KINETOGRAPH_CONCURRENCY), "--strict"]

    def on_line(raw):
        m = _PROGRESS_RE.search(raw)
        if m and int(m.group(2)):
            heartbeat.progress = 5 + int(85 * int(m.group(1)) / int(m.group(2)))
        elif raw.strip():
            log(f"  {raw[:400]}")

    limit = min(config.KINETOGRAPH_TIMEOUT_SECONDS, run_kw["timeout_seconds"] // 2)
    log(f"rendering: {' '.join(cmd)}")
    started = time.monotonic()
    try:
        proc.run_streaming(cmd, cwd=tool, env=_env(tmp_dir, ffmpeg_dir), on_line=on_line,
                           cancel_check=run_kw["cancel_check"], timeout_seconds=limit, watchdog=True)
        with open(entry_path, encoding="utf-8", errors="replace") as f:
            bad = _check_output(out_path, f.read(), ffmpeg_dir)
        if bad:
            raise RuntimeError(bad)
    except proc.Canceled:
        raise
    except (proc.TimedOut, RuntimeError, OSError, ValueError, subprocess.SubprocessError) as e:
        reason = (str(e).splitlines() or ["failed"])[0][:160]
        log(f"kinetograph: falling back to hyperframes after {time.monotonic() - started:.1f} s: {str(e)[:600]}")
        job["_done_phase"] = f"done: hyperframes (kinetograph: {reason})"
        heartbeat.progress = 3
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None
    seconds = time.monotonic() - started
    log(f"kinetograph: drawn in {seconds:.1f} s")
    job["_done_phase"] = f"done: kinetograph {seconds:.1f} s"
    return out_path
