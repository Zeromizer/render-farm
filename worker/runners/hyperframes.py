"""HyperFrames runner: `npx hyperframes render` (or `snapshot`) in the checked-out repo.

HyperFrames (HeyGen, Apache-2.0) renders an HTML composition deterministically:
headless Chrome seeks a paused GSAP timeline frame by frame, captures each frame
atomically, and the SYSTEM ffmpeg encodes and mixes audio. Nothing here is
Remotion's bundled compositor, so this engine is unaffected by the Smart App
Control block that periodically kills `remotion render` on Shawn's laptop.

params (jsonb):
  entry         composition file, repo-relative to project_dir (default index.html).
                Placements are separate root compositions in HyperFrames (variables
                cannot change width/height), so a 9:16 and a 4:5 render of one job
                are two jobs with two entries.
  project_dir   subdir of the repo holding hyperframes.json / index.html
  format        mp4 (default) | webm | mov | gif | png-sequence  (webm/mov carry alpha)
  quality       draft | standard | final (alias of standard) | high  — default standard.
                NOTE: HyperFrames' draft is a bitrate knob, not a speed knob (measured
                62 s vs 69 s on a 645-frame reel). For a fast preview send fps=15.
  fps           override the composition's data-fps (1-240)
  variables     object → written to variables.json and passed as --variables-file
  workers       parallel Chrome instances (default auto)
  gpu           true → --gpu (NVENC encode on the 4080). Off by default so a machine
                without NVENC still renders.
  resolution    supersample preset (1080p, 4k, portrait-4k ...) — optional
  crf           override CRF 0-51 — optional
  output_kind   "still" → `hyperframes snapshot` at params.at (seconds, default 0)
                and returns one PNG. Same name as the remotion runner uses, for the
                same reason: params is one shared jsonb and `output` belongs to the
                python engine.
  at            seconds into the composition for a still (default 0)
  assets        manifest — handled by render_worker before this runner

The CLI is not a project dependency: `hyperframes init` writes package.json scripts
that call `npx --yes hyperframes@X.Y.Z`, and there is no lockfile. So this runner
does NOT run npm ci; it reads the pinned version out of package.json (falling back
to HYPERFRAMES_VERSION, then a built-in default) and runs `npx --yes
hyperframes@<ver>`, which npm caches after the first download. Chrome headless
shell is fetched once per machine by `hyperframes browser ensure` into
~/.cache/hyperframes and reused; the runner calls ensure once per worker process.

Every runner here returns exactly one (path, ext, content_type); a png-sequence
is zipped, the way blender.py does.
"""
import json
import os
import re
import shutil
import zipfile

import proc

DEFAULT_VERSION = "0.8.26"
FORMAT_OUT = {
    "mp4": ("mp4", "video/mp4"),
    "webm": ("webm", "video/webm"),
    "mov": ("mov", "video/quicktime"),
    "gif": ("gif", "image/gif"),
}
QUALITIES = {"draft", "standard", "high"}
_CAPTURE_RE = re.compile(r"Capturing frame (\d+)/(\d+)")
_JSON_FRAMES_RE = re.compile(r'"framesCompleted":(\d+).{0,200}?"totalFrames":(\d+)|"totalFrames":(\d+).{0,200}?"framesCompleted":(\d+)')
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_BROWSER_READY = set()  # versions whose `browser ensure` already ran in this process
_BOX_RE = re.compile(r"[─-▟■-◿☀-➿←-⇿]")


def _clean(raw):
    """The CLI draws boxes and progress bars in Unicode; a cp1252 console cannot
    print them and the worker's stdout on Windows often is one. Strip colour
    codes and the box glyphs so every log line is plain text."""
    return _BOX_RE.sub("", _ANSI_RE.sub("", raw)).rstrip()


def _npx():
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError("npx not found on PATH — HyperFrames needs Node >= 22")
    return npx


def _pinned_version(project_dir):
    """The version the project's own scripts pin, so a re-render months later matches."""
    pkg = os.path.join(project_dir, "package.json")
    try:
        with open(pkg, encoding="utf-8") as f:
            scripts = (json.load(f).get("scripts") or {}).values()
        for s in scripts:
            m = re.search(r"hyperframes@(\d+\.\d+\.\d+[^\s]*)", s)
            if m:
                return m.group(1)
    except (OSError, ValueError):
        pass
    return os.environ.get("HYPERFRAMES_VERSION") or DEFAULT_VERSION


def _ffmpeg_dir():
    """The bin directory holding ffmpeg/ffprobe, without relying on the worker's PATH.

    The worker is launched from a Startup shortcut whose PATH predates the
    winget ffmpeg install, so `where ffmpeg` is empty there and a bare "ffmpeg"
    dies with WinError 2 — the way reference_extract did before it grew this
    same resolver (worker/extract/extract.py). Order: FFMPEG_DIR from .env,
    the winget Gyan.FFmpeg layout, then whatever PATH has.
    """
    import glob
    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "ffmpeg.exe")):
        return env_dir
    local = os.environ.get("LOCALAPPDATA", "")
    for hit in sorted(glob.glob(os.path.join(
            local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", "ffmpeg.exe")),
            reverse=True):
        return os.path.dirname(hit)
    found = shutil.which("ffmpeg")
    return os.path.dirname(found) if found else None


def _env():
    env = dict(os.environ)
    env["HYPERFRAMES_NO_TELEMETRY"] = "1"
    env["HYPERFRAMES_NO_UPDATE_CHECK"] = "1"
    env["CI"] = "1"  # never prompt, never open a browser tab
    d = _ffmpeg_dir()
    if d:
        # The CLI honours these two explicitly; PATH covers anything it spawns by name.
        env["HYPERFRAMES_FFMPEG_PATH"] = os.path.join(d, "ffmpeg.exe")
        env["HYPERFRAMES_FFPROBE_PATH"] = os.path.join(d, "ffprobe.exe")
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
    return env


def _ensure_browser(cli, project_dir, log, run_kw):
    ver = cli[-1]
    if ver in _BROWSER_READY:
        return
    log("hyperframes: browser ensure (cached after the first run)")
    proc.run_streaming(cli + ["browser", "ensure"], cwd=project_dir, env=_env(), **run_kw)
    _BROWSER_READY.add(ver)


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    """Render and return (local_output_path, ext, content_type)."""
    params = job.get("params") or {}
    project_dir = os.path.join(repo, params["project_dir"]) if params.get("project_dir") else repo
    entry = params.get("entry") or "index.html"
    if not os.path.exists(os.path.join(project_dir, entry)):
        raise RuntimeError(f"hyperframes entry not found in repo: {entry}")

    ver = _pinned_version(project_dir)
    cli = [_npx(), "--yes", f"hyperframes@{ver}"]
    run_kw = {"cancel_check": cancel_check, "timeout_seconds": timeout_seconds}
    _ensure_browser(cli, project_dir, log, dict(run_kw, on_line=lambda l: log(f"  {_clean(l)}")))
    heartbeat.progress = 3

    still = params.get("output_kind") == "still"
    if still:
        return _snapshot(cli, project_dir, entry, params, work_dir, heartbeat, log, run_kw)

    fmt = params.get("format") or "mp4"
    quality = params.get("quality") or "standard"
    if quality == "final":
        quality = "standard"
    if quality not in QUALITIES:
        raise RuntimeError(f"unsupported quality {quality!r}; expected draft|standard|high")

    if fmt == "png-sequence":
        out_local = os.path.join(work_dir, "frames")
    elif fmt in FORMAT_OUT:
        out_local = os.path.join(work_dir, f"out.{FORMAT_OUT[fmt][0]}")
    else:
        raise RuntimeError(f"unsupported format {fmt!r}; expected mp4|webm|mov|gif|png-sequence")

    cmd = cli + ["render", "--composition", entry, "--format", fmt, "--quality", quality,
                 "--output", out_local, "--json"]
    if params.get("fps"):
        cmd += ["--fps", str(int(params["fps"]))]
    if params.get("workers"):
        cmd += ["--workers", str(int(params["workers"]))]
    if params.get("gpu"):
        cmd.append("--gpu")
    if params.get("resolution"):
        cmd += ["--resolution", str(params["resolution"])]
    if params.get("crf") is not None:
        cmd += ["--crf", str(int(params["crf"]))]
    if params.get("variables"):
        vf = os.path.join(work_dir, "variables.json")
        with open(vf, "w", encoding="utf-8") as f:
            json.dump(params["variables"], f)
        cmd += ["--variables-file", vf, "--strict-variables"]

    def on_line(raw):
        line = _clean(raw)
        m = _CAPTURE_RE.search(line)
        if m and int(m.group(2)):
            heartbeat.progress = 5 + int(85 * int(m.group(1)) / int(m.group(2)))
        else:
            j = _JSON_FRAMES_RE.search(line)
            if j:
                done = int(j.group(1) or j.group(4))
                total = int(j.group(2) or j.group(3))
                if total:
                    heartbeat.progress = 5 + int(85 * done / total)
            elif "Encoding" in line or "Muxing" in line:
                heartbeat.progress = max(heartbeat.progress, 92)
        # progress bars repaint constantly; keep the log to state changes and JSON
        if line.strip() and not _CAPTURE_RE.search(line):
            log(f"  {line[:400]}")

    log(f"rendering: {' '.join(cmd)}")
    proc.run_streaming(cmd, cwd=project_dir, env=_env(), on_line=on_line, **run_kw)

    if fmt == "png-sequence":
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(out_local) for f in fs)
        if not files:
            raise RuntimeError("hyperframes reported success but the frame directory is empty")
        zip_path = os.path.join(work_dir, "frames.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, os.path.relpath(f, out_local))
        return zip_path, "zip", "application/zip"

    if not os.path.exists(out_local):
        raise RuntimeError("hyperframes reported success but output file missing")
    ext, content_type = FORMAT_OUT[fmt]
    return out_local, ext, content_type


def _snapshot(cli, project_dir, entry, params, work_dir, heartbeat, log, run_kw):
    """One PNG at params.at seconds. The still counterpart of `remotion still`."""
    at = float(params.get("at") or 0)
    out_dir = os.path.join(work_dir, "snap")
    # `snapshot` has no --composition flag (0.8.26): it takes the project DIR
    # and reads its index.html. A non-default entry (compositions/feed-4x5.html)
    # is therefore swapped into index.html for the duration of the call and put
    # back after — asset paths resolve from the project root either way, so the
    # copy renders identically. The checkout is the worker's own cache and is
    # reset on the next checkout, but restore anyway so a cancelled job leaves
    # the tree as it found it.
    cmd = cli + ["snapshot", project_dir, "--at", f"{at:g}", "--frames", "1",
                 "--no-end", "--output", out_dir, "--json"]
    index = os.path.join(project_dir, "index.html")
    backup = None
    swap = os.path.normcase(os.path.abspath(os.path.join(project_dir, entry))) != os.path.normcase(os.path.abspath(index))
    if swap:
        backup = os.path.join(work_dir, "index.html.bak")
        shutil.copy2(index, backup)
        shutil.copy2(os.path.join(project_dir, entry), index)
    log(f"snapshot: {' '.join(cmd)}" + (f" (entry {entry} swapped into index.html)" if swap else ""))
    try:
        proc.run_streaming(cmd, cwd=project_dir, env=_env(),
                           on_line=lambda l: log(f"  {_clean(l)[:400]}"), **run_kw)
    finally:
        if backup:
            shutil.copy2(backup, index)
    pngs = sorted(f for f in os.listdir(out_dir) if f.lower().endswith(".png")) if os.path.isdir(out_dir) else []
    if not pngs:
        raise RuntimeError("hyperframes snapshot produced no PNG")
    heartbeat.progress = 95
    return os.path.join(out_dir, pngs[0]), "png", "image/png"
