"""Footage timeline engine: probe, assemble and (later) H3 continuation.

Every footage task answers with a MANIFEST, never a media file — the platform
reads `renders/footage/<org>/<job>/<task>/manifest.json` and resolves the media
it references. That is why this engine exists separately from video_gen rather
than widening it: video_gen's collector looks for "videos"/"images"/"gifs" in a
ComfyUI history dict and would misfile a JSON manifest as an MP4.

Contract v1 (laptop-owned, docs/footage-worker-contract-v1.md). params.footage
is the whole request; operations:

  capabilities  no sources; what this worker can actually do today
  probe         one source; measure it, copy nothing
  assemble      sources in playback order, each with a half-open trim
  extend        one source, continue past its end          (H3, needs obvpm)
  prepend       one source, continue before its start      (H3, needs obvpm)
  bridge        two sources, departure then arrival        (H3, needs obvpm)
  loop          one source, its end back to its beginning  (H3, needs obvpm)

Artifacts are uploaded BEFORE the manifest that references them, so a manifest
never points at something that is not there yet. CPU-only operations never
touch the GPU and never pause the TTS workers.
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
import uuid

import config
import db
from videogen import segments

CONTRACT_VERSION = 1

# H3's native canvas for this prototype. 768p stays unadvertised until it is
# actually exercised on this card (Phase 0 was 480p only).
MAX_RAW_480P = 107          # largest sampling window proven on the 4080
MAX_RAW_768P = 0            # 0 = not offered
MAX_CONTEXT_BYTES = 8 * 1024 * 1024
GRID_OFFSET, GRID_STRIDE = 5, 17    # H3 pixel-frame grid: 5, 22, 39, ... 17k+5
# The SHARED AV grid is coarser: 39 + 51k (51 = 3 x 17). A pinned run off
# this grid lands the pinned sound a fraction of a frame from the pinned
# picture, which upstream warns is "an audible click over a clean picture".
# Within the 107 ceiling only 39 and 90 qualify.
AV_GRID_OFFSET, AV_GRID_STRIDE = 39, 51
# Largest AV-grid run inside the validated 107 ceiling. 141 is the next
# rung and is NOT validated on this card, so it is not offered.
MAX_AV_RAW = 90
PIN_WINDOW = 39                     # the window Phase 0 measured

# Proven end to end THROUGH THIS ADAPTER on the AV grid, not merely in a
# hand-built Phase 0 graph. Each is added only after its own raw-90 run.
#   extend: raw 90 -> delivered 51, and a SECOND generation from that
#           candidate pinned at raw_start 51 and delivered 51 again, so
#           chaining holds. bridge/loop await their own raw-90 runs at
#           12 new frames; prepend is not exercised at all.
#
# Withdrawn 2026-09-22 after review of a7951ab found that the user's cut was
# ignored, then RESTORED for extend only once each branch had its own run on
# this host. The earlier extend proof used untrimmed sources and passed by
# construction, so it is not counted:
#   latent, cut [0,56) of a 73-frame take -> pinned source frames [17,56),
#           recipe source_start_frame 17. The ignored-cut answer would have
#           been 34, so this number alone separates right from confidently
#           wrong.
#   pixel,  same cut -> staged window src[17,56), pin carries a real take_id
#           rather than None. First footage this path has ever produced.
#   auto,   cut [0,60) -> raw 21 is off the 17-frame grid, so it re-encoded
#           from pixels BEFORE sampling and named [56,73] as the legal ends.
#   latent, cut [0,60) -> refused before the GPU, naming the same ends.
#   chain,  a second extend from the trimmed-parent candidate pinned at
#           raw_start 51 and delivered 51, so sequences still grow.
#   prepend, raw 90 -> delivered 51, one pin placed AFTER the new footage,
#           arrival seam ratio 1.3 "seamless". Held context is reported at
#           the correct end (0 before / 39 after), which the site reserves
#           budget from.
#
# BRIDGE AND LOOP RUN BUT ARE DELIBERATELY NOT OFFERED. Both are mechanically
# correct at raw 90 / 12 new frames — right frame count, both pins present,
# correctly placed, correctly attributed — and both measure a HARD CUT at the
# arrival:
#   loop    departure ratio 1.2 "seamless", arrival ratio 10.6 "hard cut"
#   bridge  departure ratio 1.2 "seamless", arrival ratio 18.3 "hard cut"
# A seamless departure and a hard-cut arrival is the join a user would see
# tear. 12 new frames is half a second to travel from one pinned window to
# another, and that appears to be too few. Offering these would ship a button
# whose whole purpose is an invisible join, which measurably is not one.
# WITHDRAWN AGAIN 2026-09-22 after review of 370d65c/6480fe3. The media path
# is proven — trimmed latent, pixel import, auto fallback and chaining all
# produce correct footage — but the FAILURE paths are not:
#   comfy_client.wait() calls a bare interrupt() on timeout and on cancel,
#   before _abort_prompt is ever reached, so a cancelled footage task can
#   stop an unrelated video_gen prompt on the same ComfyUI. That is a live
#   hazard to other work and is reason enough on its own.
#   An uncertain submit whose queue read also fails resumes the TTS workers,
#   because _prompt_is_live cannot distinguish "not queued" from "cannot
#   tell"; queue_state accepts a non-2xx body as an empty queue.
#   The prompt journal is written and never read, so a same-task retry
#   submits a second generation instead of reconciling the first.
#   Heartbeat is a class instance, not a callable, so _hb drops every
#   phase, fraction and ETA it claimed to forward.
#   resolve_boundary compares SOURCE frames against a 39 OUTPUT-frame
#   window, so a 30fps import passes the check with too little context.
# Correct output on the happy path is not enough when the failure paths can
# disturb other jobs or silently under-pin an import.
PROVEN_GENERATION_OPS = ()

# Advertising an operation this runner would then refuse is worse than not
# offering it: the website enables the button on capabilities alone, so the
# two MUST agree. True only because extend has run end to end here; what is
# actually offered is still governed by PROVEN_GENERATION_OPS.
GENERATION_ADAPTER_READY = True


class FootageError(RuntimeError):
    """A request this worker will not run, with a reason the website shows."""


# ---------------------------------------------------------------- probing

def _ffprobe(path):
    out = subprocess.run(
        [segments._tool("ffprobe"), "-v", "error",
         "-select_streams", "v:0", "-show_streams", "-show_format",
         "-of", "json", path],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise FootageError(f"ffprobe failed on {os.path.basename(path)}: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def _rate(text):
    """'24/1' -> {"num": 24, "den": 1}."""
    num, _, den = (text or "").partition("/")
    try:
        n, d = int(num), int(den or 1)
    except ValueError:
        raise FootageError(f"unreadable frame rate {text!r}")
    if n <= 0 or d <= 0:
        raise FootageError(f"non-positive frame rate {text!r}")
    return {"num": n, "den": d}


def _has_audio(path):
    out = subprocess.run(
        [segments._tool("ffprobe"), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=120)
    return bool(out.stdout.strip())


def probe_media(path):
    """MediaInfo for one file. Frame counts describe the DECODED file, not the
    container's nominal duration, so a trim expressed in frames is exact."""
    info = _ffprobe(path)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise FootageError(f"{os.path.basename(path)} has no video stream")
    s = streams[0]

    # CFR is required for frame addressing. r_frame_rate is the base rate and
    # avg_frame_rate the realised one; they diverge on variable-rate sources.
    r, avg = _rate(s.get("r_frame_rate")), _rate(s.get("avg_frame_rate") or s.get("r_frame_rate"))
    if r["num"] * avg["den"] != avg["num"] * r["den"]:
        raise FootageError(
            "variable frame rate is not supported for frame-accurate editing "
            f"({os.path.basename(path)}: base {r['num']}/{r['den']}, "
            f"average {avg['num']}/{avg['den']}). Normalize it first.")

    frames = s.get("nb_frames")
    if not frames or int(frames) <= 0:
        # Containers often omit nb_frames; counting is slower but exact, and
        # every downstream trim depends on this number being right.
        cnt = subprocess.run(
            [segments._tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=600)
        frames = (cnt.stdout or "").strip().split(",")[0]
    frame_count = int(frames or 0)
    if frame_count <= 0:
        raise FootageError(f"could not determine a frame count for {os.path.basename(path)}")

    return {"fps": r, "frame_count": frame_count,
            "width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
            "duration_seconds": round(frame_count * r["den"] / r["num"], 6),
            "has_audio": _has_audio(path)}


def sha256_size(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)


# ------------------------------------------------------- frame arithmetic

def allocate_frames(in_frame, out_frame, source_fps, output_fps):
    """Output frames one source trim contributes, per contract v1:

        max(1, round((out-in) * source_den * output_num / (source_num * output_den)))

    Exact integer arithmetic — no float drift — and never zero, so a very short
    trim still shows one frame rather than vanishing from the sequence.
    """
    span = int(out_frame) - int(in_frame)
    if span <= 0:
        raise FootageError(f"trim must be half-open and non-empty (got {in_frame}..{out_frame})")
    num = span * source_fps["den"] * output_fps["num"]
    den = source_fps["num"] * output_fps["den"]
    # round-half-up on exact integers
    return max(1, (2 * num + den) // (2 * den))


def solve_sample_window(desired_new, held_prefix, held_suffix, av_grid=False):
    """(raw, delivered) for a continuation: snap the whole sampling window UP
    to a legal run length, then report what is actually delivered.

    The held context is sampled but trimmed off the delivered clip, so asking
    for N new frames costs N + held, and grid rounding can hand back more new
    frames than were asked for (the contract calls new_frames a minimum).

    Two grids are in play and they are NOT the same:
      video: 17k+5 -> 5, 22, 39, 56, 73, 90, 107, 124
      shared AV: 39 + 51k -> 39, 90, 141, 192, 243
    A run that is only video-legal puts the pinned sound a fraction of a frame
    away from the pinned picture. Upstream is explicit about the consequence:
    "the join skips that much -- an audible click over a clean picture". So a
    pinned run whose sources carry audio must land on the AV grid; 107 does
    not, which is why it is unsafe for an audible join even though the picture
    is fine.
    """
    need = int(desired_new) + int(held_prefix) + int(held_suffix)
    if av_grid:
        raw = AV_GRID_OFFSET
        while raw < need:
            raw += AV_GRID_STRIDE
    elif need <= GRID_OFFSET:
        raw = GRID_OFFSET
    else:
        steps = -(-(need - GRID_OFFSET) // GRID_STRIDE)   # ceil
        raw = GRID_OFFSET + steps * GRID_STRIDE
    return raw, raw - int(held_prefix) - int(held_suffix)


def held_frames(op):
    """(held_before, held_after) for one operation.

    Held context is sampled and then trimmed off the delivered clip, so the
    website reserves budget from these. They are NOT symmetric: prepend
    generates INTO its source, so its 39 held frames sit AFTER the new
    footage, not before it. Reporting prepend as held_prefix would have the
    site reserve at the wrong end.
    """
    if op == "prepend":
        return 0, PIN_WINDOW
    if op in ("bridge", "loop"):
        return PIN_WINDOW, PIN_WINDOW
    return PIN_WINDOW, 0


def generation_limits():
    """Per-operation limits on the shared AV grid, for operations proven here.

    The ceiling is the largest AV-grid run within the validated 107-frame
    sampling limit, which is 90. So one-sided work delivers 90-39 = 51 and
    two-sided 90-39-39 = 12 — a short seam repair, not a one-second
    transition, and the UI should say so.
    """
    lim = {}
    for op in PROVEN_GENERATION_OPS:
        prefix, suffix = held_frames(op)
        lim[op] = {"max_new_frames": MAX_AV_RAW - prefix - suffix,
                   "held_prefix_frames": prefix, "held_suffix_frames": suffix,
                   "grid_offset": AV_GRID_OFFSET, "grid_stride": AV_GRID_STRIDE}
    return lim


# ------------------------------------------------------------- operations

def _comfy_identity():
    """ComfyUI version/commit and whether obvpm is loaded, for capabilities.

    Never starts the server: capabilities must be answerable on a CPU-only
    runner while the GPU is busy, so a missing server reports unknown rather
    than blocking or launching one.
    """
    from videogen import comfy_client
    version = commit = None
    obvpm = None
    try:
        if comfy_client.is_up():
            version = (comfy_client.system_stats().get("system") or {}).get("comfyui_version")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        head = subprocess.run(["git", "-C", config.COMFYUI_DIR, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        commit = (head.stdout or "").strip() or None
    except Exception:                                    # noqa: BLE001
        pass
    node = os.path.join(config.COMFYUI_DIR, "custom_nodes", "comfyui-obvpm-timeline")
    if os.path.isdir(node):
        # The pack is installed as a plain directory INSIDE the ComfyUI
        # worktree, so `git rev-parse` here would walk up and report ComfyUI's
        # own HEAD — the wrong commit entirely. The installer records the
        # pinned upstream revision beside the code instead.
        stamp = os.path.join(node, "INSTALLED_REV.txt")
        if os.path.isfile(stamp):
            with open(stamp) as f:
                obvpm = f.read().strip() or "present"
        else:
            obvpm = "present"
    return version, commit, obvpm


def capabilities_block():
    version, commit, obvpm = _comfy_identity()
    ops = ["capabilities", "probe", "assemble"]
    # Generation needs BOTH: the node pack installed in the ComfyUI this worker
    # drives, AND an adapter in run() that can actually serve the request.
    # Phase 0 proved the operations in an isolated instance; that proves the
    # nodes work, not that this runner can drive them.
    # An empty proven list means no generative operation is offered at all, so
    # latent_context must read false too: claiming latent capability while
    # serving nothing generative is the same lie in a different field.
    can_generate = (bool(obvpm) and GENERATION_ADAPTER_READY
                    and bool(PROVEN_GENERATION_OPS))
    if can_generate:
        ops += list(PROVEN_GENERATION_OPS)
    return {"contract_version": CONTRACT_VERSION,
            "worker_revision": _worker_revision(),
            "comfyui_version": version, "comfyui_commit": commit,
            "obvpm_commit": obvpm,
            "operations": ops,
            "latent_context": can_generate,
            # the AV-grid ceiling, because that is the run we will actually
            # sample; quoting the video-grid 107 would under-reserve budget
            "max_generation_frames": {"480p": MAX_AV_RAW if can_generate else 0,
                                      "768p": MAX_RAW_768P},
            "max_context_bytes": MAX_CONTEXT_BYTES if can_generate else 0,
            "generation_limits": generation_limits() if can_generate else {}}


def _worker_revision():
    r = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    return (r.stdout or "").strip() or "unknown"


def _fetch_source(src, work_dir, name, log):
    """Resolve one contract Source's media ref to a local file. Refs come from
    the website's scoped IDs — never from a browser — but are still validated
    here: bucket must be one we serve, and the probe must agree with the trim.
    """
    ref = src.get("media") or {}
    bucket, path = ref.get("bucket"), ref.get("path")
    if bucket not in ("assets", "renders") or not path:
        raise FootageError(f"source {name}: media ref must name the assets or renders bucket")
    local = os.path.join(work_dir, f"{name}{os.path.splitext(path)[1] or '.mp4'}")
    db.download_to(bucket, path, local) if hasattr(db, "download_to") else _download(bucket, path, local)
    log(f"{name}: {bucket}/{path} -> {os.path.getsize(local)} bytes")
    return local


def _download(bucket, path, local):
    data = db.sb.storage.from_(bucket).download(path)
    with open(local, "wb") as f:
        f.write(data)


def op_probe(req, work_dir, log):
    sources = req.get("sources") or []
    if len(sources) != 1:
        raise FootageError("probe takes exactly one source")
    src = sources[0]
    local = _fetch_source(src, work_dir, "probe", log)
    info = probe_media(local)
    digest, size = sha256_size(local)
    out = int(src.get("out_frame") or 0)
    # out_frame 0 is the "inspect the whole original" sentinel, and contract v1
    # permits it ONLY here.
    if out and out > info["frame_count"]:
        raise FootageError(f"trim ends at frame {out} but the media has {info['frame_count']}")
    log(f"probe: {info['width']}x{info['height']} {info['frame_count']}f "
        f"{info['fps']['num']}/{info['fps']['den']} audio={info['has_audio']}")
    return {"media": dict(src.get("media") or {}, sha256=digest, size=size, info=info),
            "lineage": {"source_take_ids": [src.get("take_id")],
                        "source_trims": [{"take_id": src.get("take_id"),
                                          "in_frame": int(src.get("in_frame") or 0),
                                          "out_frame": out or info["frame_count"]}]},
            "warnings": []}


def op_assemble(req, work_dir, hb, log):
    """Trim each source to its half-open window, normalize to the output canvas
    and rate, and concatenate into one MP4 with continuous timestamps.

    The result is explicitly a DERIVATIVE: it may resample rates and sizes, so
    it never carries a latent context of its own.
    """
    sources = req.get("sources") or []
    if not sources:
        raise FootageError("assemble needs at least one source")
    out_spec = req.get("output") or {}
    ofps = out_spec.get("fps") or {"num": 24, "den": 1}
    width, height = int(out_spec.get("width") or 832), int(out_spec.get("height") or 480)

    parts, trims, take_ids, allocated, warnings = [], [], [], 0, []
    any_audio = False
    for i, src in enumerate(sources):
        local = _fetch_source(src, work_dir, f"src{i:02d}", log)
        info = probe_media(local)
        a, b = int(src.get("in_frame") or 0), int(src.get("out_frame") or 0)
        if b <= a or b > info["frame_count"]:
            raise FootageError(
                f"source {i}: trim {a}..{b} is outside the media's {info['frame_count']} frames")
        n = allocate_frames(a, b, info["fps"], ofps)
        allocated += n
        any_audio = any_audio or info["has_audio"]

        part = os.path.join(work_dir, f"part{i:02d}.mp4")
        _cut(local, a, b, info, ofps, width, height, n, part)
        parts.append(part)
        take_ids.append(src.get("take_id"))
        trims.append({"take_id": src.get("take_id"), "in_frame": a, "out_frame": b})
        if (info["width"], info["height"]) != (width, height):
            warnings.append(f"source {i} rescaled from {info['width']}x{info['height']}")
        if info["fps"] != ofps:
            warnings.append(
                f"source {i} retimed from {info['fps']['num']}/{info['fps']['den']} "
                f"to {ofps['num']}/{ofps['den']}")
        if not info["has_audio"]:
            warnings.append(f"source {i} is silent; silence was inserted")
        hb.beat() if hasattr(hb, "beat") else None
        log(f"src{i:02d}: {a}..{b} -> {n} output frames")

    final = os.path.join(work_dir, "assembled.mp4")
    _concat(parts, final, log)
    info = probe_media(final)
    if info["frame_count"] != allocated:
        # The allocation IS the contract; a mismatch means the concat drifted.
        warnings.append(
            f"assembled {info['frame_count']} frames, allocation predicted {allocated}")
    digest, size = sha256_size(final)
    log(f"assembled {len(parts)} parts -> {info['frame_count']} frames, {size} bytes")
    return {"_media_local": final,
            "media": {"bucket": config.BUCKET, "path": None, "sha256": digest,
                      "size": size, "info": info},
            "lineage": {"source_take_ids": take_ids, "source_trims": trims},
            "warnings": warnings}


def _cut(src, a, b, info, ofps, width, height, want, dest):
    """One trimmed, normalized part. Frame-exact: select by frame index rather
    than by timestamp, so a trim never lands a frame early or late."""
    vf = (f"select='between(n\\,{a}\\,{b - 1})',setpts=N/FRAME_RATE/TB,"
          f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:-1:-1:color=black,fps={ofps['num']}/{ofps['den']}")
    cmd = [segments._tool("ffmpeg"), "-v", "error", "-y", "-i", src]
    if not info["has_audio"]:
        # Silence rather than no track at all, so concat never has to reconcile
        # a stream that exists in some parts and not others.
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-vf", vf, "-frames:v", str(want),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if info["has_audio"]:
        start = a * 48000 * info["fps"]["den"] // info["fps"]["num"]
        cmd += ["-map", "0:v:0", "-map", "0:a:0",
                "-af", f"atrim=start_sample={start},asetpts=N/SR/TB"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd += ["-c:a", "aac", "-ar", "48000", "-ac", "2", dest]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise FootageError(f"trim failed: {r.stderr.strip()[:400]}")


def _concat(parts, dest, log):
    lst = os.path.join(os.path.dirname(dest), "concat.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p.replace(chr(92), '/')}'\n")
    r = subprocess.run(
        [segments._tool("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", "-fflags", "+genpts", dest],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise FootageError(f"concat failed: {r.stderr.strip()[:400]}")


def read_mctx_header(path):
    """The __metadata__ dict of a .mctx.safetensors sidecar.

    safetensors layout: 8-byte little-endian header length, then that many
    bytes of JSON. Read directly rather than importing obvpm's mctx module —
    it is a ComfyUI custom node with package-relative imports and is not
    importable from the worker.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise FootageError(f"{os.path.basename(path)} is not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if not 0 < n < (64 << 20):
            raise FootageError(f"{os.path.basename(path)}: implausible header length {n}")
        head = json.loads(f.read(n))
    return head.get("__metadata__") or {}


# ------------------------------------------------------ H3 continuation

# obvpm resolves a take by scanning ComfyUI's output folder, so a source pair
# has to be staged there under a name we choose. Everything for one task lives
# in its own directory, keyed by task id, so two concurrent tasks cannot see
# each other's takes and upstream's filenames never leak into our IDs.
def _stage_pair(src, work_dir, idx, log):
    """Put one source's MP4 (and .mctx sidecar, when it has one) where obvpm
    can find it. Returns (clip_ref, has_context, local_mp4)."""
    local = _fetch_source(src, work_dir, f"stage{idx:02d}", log)
    ctx_ref = src.get("context") or {}
    stage_dir = os.path.join(config.COMFYUI_DIR, "output", _stage_folder(src))
    os.makedirs(stage_dir, exist_ok=True)
    name = _stage_stem(stage_dir, f"src{idx:02d}")
    shutil.copyfile(local, os.path.join(stage_dir, f"{name}.mp4"))
    has_ctx = False
    if ctx_ref.get("path"):
        sidecar = os.path.join(work_dir, f"stage{idx:02d}.mctx.safetensors")
        _download(ctx_ref.get("bucket") or "renders", ctx_ref["path"], sidecar)
        # The pair is identified by CONTENT: obvpm matches the sidecar to the
        # MP4 by hash, so a mismatched pair is caught there, not here.
        shutil.copyfile(sidecar, os.path.join(stage_dir, f"{name}.mctx.safetensors"))
        has_ctx = True
    log(f"staged {name}.mp4{' + context' if has_ctx else ' (video only)'}")
    return f"{_stage_folder(src)}/{name}.mp4", has_ctx, local


_STAGE_KEY = {}


def _stage_folder(src):
    return _STAGE_KEY["folder"]


def _stage_stem(stage_dir, base):
    """A stem whose .mp4 and .mctx.safetensors can both actually be written.

    Staging is keyed by task id, so a RETRY of the same task — which the
    farm's reclaim path can produce — restages under the same names. ComfyUI
    keeps the previous attempt's sidecar memory-mapped, and overwriting an
    mmapped file fails with EINVAL on Windows, so the retry dies before it
    starts. Clear the names when possible, and step aside when not.
    """
    for attempt in range(20):
        stem = base if attempt == 0 else f"{base}-r{attempt}"
        blocked = False
        for ext in (".mp4", ".mctx.safetensors"):
            path = os.path.join(stage_dir, stem + ext)
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError:
                blocked = True
                break
        if not blocked:
            return stem
    raise FootageError(
        f"could not clear a staging name under {stage_dir}; 20 attempts are "
        "all still held open by ComfyUI")


def _stage_window(local, info, spec, req, idx, log):
    """Trim ONE pixel boundary's window and normalize it exactly as assembly
    does: aspect-fit with black pad, output frame rate, 48k stereo audio.

    This is what makes the declared fps truthful and makes "cover" a no-op
    instead of a second, differently-framed crop of the car.
    """
    stage_dir = os.path.join(config.COMFYUI_DIR, "output", _stage_folder(None))
    os.makedirs(stage_dir, exist_ok=True)
    name = _stage_stem(stage_dir, f"b{idx:02d}")
    dest = os.path.join(stage_dir, f"{name}.mp4")
    ofps = (req.get("output") or {}).get("fps") or {"num": 24, "den": 1}
    width, height = _canvas(req)
    a, b = spec["in_frame"], spec["out_frame"]
    # The window is PIN_WINDOW frames at the OUTPUT rate, so convert it back
    # into source frames rather than assuming the source runs at 24.
    span = allocate_frames(0, spec["window"], ofps, info["fps"])
    if spec["role"] == "departure":
        a2, b2 = max(a, b - span), b
    else:
        a2, b2 = a, min(b, a + span)
    _cut(local, a2, b2, info, ofps, width, height, spec["window"], dest)
    got = probe_media(dest)
    log(f"{name}: pixel window src[{a2},{b2}) -> {got['frame_count']}f "
        f"{width}x{height} @{ofps['num']}/{ofps['den']}"
        f"{'' if info['has_audio'] else ' (silent source)'}")
    return (f"{_stage_folder(None)}/{name}.mp4", got, (a2, b2))


LATENT_GROUP = 17           # obvpm slices only at 17-frame group boundaries


def boundary_plan(op):
    """Which end of which source supplies each pinned window.

    A departure is pinned BEFORE the new footage and an arrival AFTER it.
    loop takes both boundaries from ONE source; bridge takes one from each.
    """
    return {"extend":  [(0, "departure")],
            "prepend": [(0, "arrival")],
            "loop":    [(0, "departure"), (0, "arrival")],
            "bridge":  [(0, "departure"), (1, "arrival")]}[op]


def resolve_boundary(role, src, header, window, frame_count,
                     source_fps=None, output_fps=None):
    """Where one pinned window sits INSIDE the user's cut.

    The cut is [in_frame, out_frame) in the source's delivered frames. A
    departure holds the last `window` frames of that cut, so the window ENDS
    at out_frame; an arrival holds the first `window`, so it ends at
    in_frame + window. obvpm's take_from_frame is exactly that end, in
    delivered coordinates, which is why nothing here converts to raw except
    the legality test.

    Returns a dict describing the pin, including whether a LATENT slice of it
    is legal. Upstream slices only at 17-frame group boundaries
    (nodes_pins.py: raw_start % FRAMES_PER_GROUP), so an interior cut is
    frequently illegal and must be resolved BEFORE the GPU is booked.
    """
    delivered = int(header.get("delivered_frames") or 0) or int(frame_count)
    pinned_head = int(header.get("pinned_head_frames") or 0)
    a = int(src.get("in_frame") or 0)
    b = int(src.get("out_frame") or 0) or delivered
    if not 0 <= a < b <= delivered:
        raise FootageError(
            f"cut [{a},{b}) is outside the source's {delivered} delivered "
            "frames")
    # The window is `window` frames at the OUTPUT rate; the cut is in SOURCE
    # frames. Comparing them directly passed a 40-frame cut of 30fps footage
    # that yields only 32 output frames, and rejected a 20-frame cut of 12fps
    # footage that yields 40 — both wrong, and the short one only surfaced as
    # a warning after the GPU had already been booked.
    have_out = allocate_frames(a, b, source_fps, output_fps) \
        if source_fps and output_fps else b - a
    if have_out < window:
        need_src = allocate_frames(0, window, output_fps, source_fps) \
            if source_fps and output_fps else window
        out_rate = (output_fps["num"] / output_fps["den"]) if output_fps else 24.0
        raise FootageError(
            f"a continuation holds {window} frames of context at the output "
            f"rate, but the cut [{a},{b}) supplies only {have_out}. Select at "
            f"least {need_src} frames of this clip "
            f"({window / out_rate:.2f}s of finished footage).")

    if role == "departure":
        cut, take_from = b, ("tail" if b == delivered else "at_frame")
    else:
        cut, take_from = a + window, ("head" if a == 0 else "at_frame")

    # Mirror upstream's own arithmetic rather than approximating it.
    d_start = 0 if take_from == "head" else cut - window
    raw_start = pinned_head + d_start
    legal = raw_start % LATENT_GROUP == 0
    suggest = []
    if not legal:
        lo = (raw_start // LATENT_GROUP) * LATENT_GROUP
        for boundary in (lo, lo + LATENT_GROUP):
            end = boundary - pinned_head + window
            if 0 <= boundary - pinned_head and end <= delivered:
                suggest.append(end)
    return {"role": role, "place": "before" if role == "departure" else "after",
            "mode": "masked" if role == "departure" else "both",
            "take_from": take_from, "take_from_frame": int(cut),
            "in_frame": a, "out_frame": b, "window": window,
            "raw_start": raw_start, "latent_legal": legal,
            "legal_ends": sorted(set(suggest)), "take_id": src.get("take_id")}


def _pin_spec_node(mctx_ref, window, take_from, place, mode, chain=None,
                   take_from_frame=0):
    inputs = {"mctx": mctx_ref, "window": str(window),
              "take_from": take_from, "take_from_frame": int(take_from_frame),
              "place": place, "place_at_frame": 0,
              "audio_window": 0, "mode": mode,
              "mask_ramp_frames": 0, "mask_ramp_edge": 0.0, "mask_hold": 0.0}
    if chain is not None:
        inputs["pin_specs"] = chain
    return {"class_type": "H3MCtxPinSpec", "inputs": inputs}


def build_continuation(op, req, staged, length, prefix, latent_only, plan):
    """The API graph for extend/prepend/bridge/loop.

    Departure is pinned BEFORE the new footage and arrival AFTER it. A source
    with a verified context is loaded through H3LoadMCtx (raw latents); one
    without goes through H3MCtxFromFrames, which re-encodes pixels and is
    exact only to the VAE round trip.
    """
    gen = req.get("generation") or {}
    g = _h3_base(gen.get("prompt") or "", int(gen.get("seed") or 0), length,
                 *_canvas(req))
    mctx_nodes = {}
    # A LATENT source is loaded once and shared by both of its boundaries:
    # take_from_frame distinguishes them inside the same stored latent.
    for i, (clip_ref, has_ctx, _local) in enumerate(staged):
        if has_ctx:
            g[f"loadsrc{i}"] = {"class_type": "H3LoadMCtx",
                                "inputs": {"clip": clip_ref, "create_pins": "none",
                                           "pin_window": str(PIN_WINDOW)}}
            mctx_nodes[("src", i)] = [f"loadsrc{i}", 0]
    # A PIXEL boundary gets its OWN encoder over its OWN pre-trimmed clip.
    # Sharing one encoder per source was wrong for loop: both pins then came
    # from the same kept tail, so the arrival was the clip's end rather than
    # its beginning.
    for n, spec in enumerate(plan):
        i = spec["src_index"]
        if staged[i][1]:
            continue
        if latent_only:
            raise FootageError(
                f"source {i} has no original generation context, so a "
                "latent continuation is impossible. Use 'auto' to fall "
                "back to the pixel path, or pick a take that has context.")
        key = f"b{n}"
        # The staged clip is already trimmed to this boundary's window and
        # normalized to the output rate and canvas, so fps is a measured
        # fact rather than a guess and "cover" cannot crop anything.
        # _stage_pair writes into ComfyUI/output, but LoadVideo validates
        # through folder_paths.exists_annotated_filepath, and a BARE
        # relative path is resolved against input. Without the annotation
        # the prompt is rejected at submission with "Invalid video file".
        # H3LoadMCtx takes a clip identifier, not an annotated path, so
        # this must not be applied there.
        g[f"lv{key}"] = {"class_type": "LoadVideo",
                         "inputs": {"file": f"{spec['clip_ref']} [output]"}}
        g[f"comp{key}"] = {"class_type": "GetVideoComponents",
                           "inputs": {"video": [f"lv{key}", 0]}}
        enc = {"images": [f"comp{key}", 0],
               "video_vae": ["vae", 0], "audio_vae": ["avae", 0],
               "latent": ["cond", 1], "fps": float(spec["staged_fps"]),
               "keep": "tail" if spec["role"] == "departure" else "head",
               "max_frames": PIN_WINDOW, "fit": "cover"}
        # Absent audio means the pin carries encoded silence, which is right
        # for a genuinely silent source and wrong for one we simply failed to
        # connect — so this follows what the staged file actually contains.
        if spec.get("staged_has_audio"):
            enc["audio"] = [f"comp{key}", 1]
        g[f"enc{key}"] = {"class_type": "H3MCtxFromFrames", "inputs": enc}
        mctx_nodes[("bnd", n)] = [f"enc{key}", 0]

    # Each pinned window comes from the plan, which has already placed it
    # inside the user's cut. A pixel source was staged pre-trimmed, so its
    # window is the whole staged clip and take_from_frame does not apply.
    if not plan:
        raise FootageError(f"{op!r} is not a continuation")
    chain = None
    for n, spec in enumerate(plan):
        name = f"spec{chr(ord('A') + n)}"
        src_i = spec["src_index"]
        pixel = not staged[src_i][1]
        take_from = ("tail" if spec["role"] == "departure" else "head") \
            if pixel else spec["take_from"]
        ref = mctx_nodes[("bnd", n)] if pixel else mctx_nodes[("src", src_i)]
        g[name] = _pin_spec_node(
            ref, PIN_WINDOW, take_from, spec["place"],
            spec["mode"], chain=None if chain is None else [chain, 0],
            take_from_frame=0 if take_from != "at_frame"
            else spec["take_from_frame"])
        chain = name

    g["apply"] = {"class_type": "H3MCtxApplyPins",
                  "inputs": {"conditioning": ["cond", 0], "latent": ["cond", 1],
                             # let obvpm snap a window down rather than fail; the
                             # runner reports any snap it actually performed.
                             "snap_window_down_to_available": True,
                             "freeze_audio": False,
                             "video_vae": ["vae", 0], "audio_vae": ["avae", 0],
                             "pin_specs": [chain, 0]}}
    g["guider"] = {"class_type": "BasicGuider",
                   "inputs": {"model": ["lora", 0], "conditioning": ["apply", 0]}}
    # the sampler MUST consume ApplyPins output 1; pins is output 2
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                              "sampler": ["sampler", 0], "sigmas": ["sched", 0],
                              "latent_image": ["apply", 1]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio",
                 "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["save"] = {"class_type": "H3TrimAndSaveVideoWithMCtx",
                 "inputs": {"samples": ["sample", 0], "images": ["dec", 0],
                            "audio": ["adec", 0], "base_folder": _STAGE_KEY["folder"],
                            "filename_prefix": prefix, "crf": 19,
                            "save_conditioning": True, "pins": ["apply", 2]}}
    # The saver writes the pair but reports NOTHING to /history — its outputs
    # dict comes back empty. H3ResultPreview is the declared result node: fed
    # the saver's path, it emits ui.h3_result carrying clip, parent, parent2,
    # relation, seam and seam2. Without it this runner cannot find its own
    # output, and the two seam measurements would be lost.
    g["result"] = {"class_type": "H3ResultPreview", "inputs": {"path": ["save", 0]}}
    return g


def op_continuation(op, req, jid, work_dir, hb, log, cancel_check, timeout_seconds):
    """Run one extend / prepend / bridge / loop and reconcile what came back."""
    from videogen import comfy_client, tts_guard

    gen = req.get("generation") or {}
    sources = req.get("sources") or []
    need = {"extend": 1, "prepend": 1, "loop": 1, "bridge": 2}[op]
    if len(sources) != need:
        raise FootageError(f"{op} needs exactly {need} source(s), got {len(sources)}")
    mode = gen.get("context_mode") or "auto"
    if mode not in ("auto", "pixel", "latent"):
        raise FootageError(f"context_mode must be auto, pixel or latent (got {mode!r})")
    if (gen.get("resolution") or "480p") != "480p":
        raise FootageError("only 480p is validated on this worker")

    warnings = []
    two_sided = op in ("bridge", "loop")
    # ALWAYS the shared AV grid, for every source audio state. A masked pin
    # window must sit on it regardless of audio (upstream masked_window_ok),
    # and an after-pin with audio additionally constrains the RUN length. An
    # audio stream is not evidence of audible content, and a generated
    # successor can carry audio even when its source did not — so one
    # conservative profile beats switching per source.
    held_pre, held_post = held_frames(op)
    raw, delivered = solve_sample_window(
        int(gen.get("new_frames") or 24), held_pre, held_post, av_grid=True)
    if raw > MAX_AV_RAW:
        most = MAX_AV_RAW - held_pre - held_post
        raise FootageError(
            f"that request needs a {raw} frame sampling window; the largest "
            f"run on the shared audio/video grid within this worker's "
            f"validated limit is {MAX_AV_RAW}. The most new frames {op} can "
            f"add is {most}.")
    if delivered > int(gen.get("new_frames") or 24):
        warnings.append(
            f"grid rounding delivers {delivered} new frames, more than the "
            f"{gen.get('new_frames')} requested")

    # ---- reconcile any PRIOR attempt at this task, BEFORE touching staging --
    # Reclaim can re-enter a task under its own id. Staging would then rewrite
    # the very inputs a still-running prompt is reading, and stepping aside to
    # a fresh name would quietly turn an uncertain retry into a SECOND
    # generation on the same GPU. Settle the previous attempt first.
    prior = _journal_last(jid)
    if prior and prior.get("state") not in _SETTLED and prior.get("prompt_id"):
        ppid = prior["prompt_id"]
        state = _prompt_state(ppid)
        log(f"task {jid} has a prior prompt {ppid} in state "
            f"{prior['state']!r}; ComfyUI says {state}")
        if state == LIVE:
            raise FootageError(
                f"this task already has prompt {ppid} on the ComfyUI queue "
                f"(recorded as {prior['state']!r}). Refusing to submit a "
                "second generation for the same task. Let it finish, or "
                "cancel it, then retry with a new task.")
        if state == UNKNOWN:
            raise FootageError(
                f"this task has an unsettled prompt {ppid} and ComfyUI could "
                "not be read to find out whether it is still running. "
                "Refusing to risk a duplicate generation.")
        _journal(jid, ppid, "reconciled-gone", op=op)

    _STAGE_KEY["folder"] = f"footage/{jid}"
    staged = [_stage_pair(s, work_dir, i, log) for i, s in enumerate(sources)]
    if mode == "latent" and not all(has for _, has, _ in staged):
        raise FootageError(
            "context_mode 'latent' was requested but at least one source has no "
            "original generation context. Legal alternatives: use 'auto' to "
            "re-encode from pixels, or choose a take that has context.")
    if mode == "pixel":
        staged = [(ref, False, loc) for ref, _has, loc in staged]

    # ---- per-source, per-boundary preflight, BEFORE the GPU is booked ----
    # Each pinned window has to sit inside the user's cut, and a latent slice
    # is legal only on the 17-frame group boundary. Resolving that here means
    # an illegal cut costs nothing; discovering it after sampling would waste
    # a run, and ignoring it (as this runner used to) silently generated from
    # footage the user had trimmed away.
    infos = [probe_media(loc) for _, _, loc in staged]
    headers = []
    for i, (_ref, has_ctx, loc) in enumerate(staged):
        side = os.path.join(work_dir, f"stage{i:02d}.mctx.safetensors")
        headers.append(read_mctx_header(side)
                       if has_ctx and os.path.isfile(side) else {})
    plan = []
    for i, role in boundary_plan(op):
        spec = resolve_boundary(role, sources[i], headers[i], PIN_WINDOW,
                                infos[i]["frame_count"],
                                source_fps=infos[i]["fps"],
                                output_fps=(req.get("output") or {}).get("fps")
                                or {"num": 24, "den": 1})
        spec["src_index"] = i
        plan.append(spec)

    # A source goes to pixels if ANY of its boundaries cannot be sliced.
    illegal = {}
    for spec in plan:
        if staged[spec["src_index"]][1] and not spec["latent_legal"]:
            illegal.setdefault(spec["src_index"], []).append(spec)
    for i, specs in illegal.items():
        ends = sorted({e for s in specs for e in s["legal_ends"]})
        detail = (f"source {i}'s cut puts a pinned window at raw frame "
                  f"{specs[0]['raw_start']}, which is not on the "
                  f"{LATENT_GROUP}-frame latent grid")
        if mode == "latent":
            alts = (f" Legal window end frames for this clip: {ends}."
                    if ends else "")
            raise FootageError(
                f"context_mode 'latent' was requested but {detail}, so the "
                f"slice would be unsound.{alts} Or use 'auto' to re-encode "
                "this cut from pixels at the cost of exactness.")
        staged[i] = (staged[i][0], False, staged[i][2])
        warnings.append(
            f"{detail}, so this cut was re-encoded from pixels instead; the "
            "join is exact only to the VAE round trip"
            + (f". Latent-grade cuts here end at {ends}" if ends else ""))

    if mode == "pixel":
        warnings.append("pixel context was requested explicitly; the join is "
                        "exact only to the VAE round trip")
    elif not illegal and not all(has for _, has, _ in staged) and mode == "auto":
        warnings.append(
            "continued from re-encoded pixels because original generation "
            "context was not available for every source; the join is exact "
            "only to the VAE round trip")
    resolved_mode = "latent" if all(has for _, has, _ in staged) else "pixel"

    # Stage each pixel boundary's own normalized window.
    for n, spec in enumerate(plan):
        i = spec["src_index"]
        if staged[i][1]:
            continue
        ref, got, src_span = _stage_window(staged[i][2], infos[i], spec, req, n, log)
        spec["clip_ref"] = ref
        spec["staged_fps"] = got["fps"]["num"] / float(got["fps"]["den"])
        spec["staged_has_audio"] = bool(infos[i]["has_audio"])
        spec["staged_source_frames"] = list(src_span)
        if got["frame_count"] != PIN_WINDOW:
            warnings.append(
                f"a pinned window yielded {got['frame_count']} frames rather "
                f"than {PIN_WINDOW}; your cut was not moved")
        if not infos[i]["has_audio"]:
            warnings.append(
                f"source {i} has no audio, so its pinned context carries "
                "encoded silence")
        if (infos[i]["width"], infos[i]["height"]) != _canvas(req):
            warnings.append(
                f"source {i} was rescaled from {infos[i]['width']}x"
                f"{infos[i]['height']} to {_canvas(req)[0]}x{_canvas(req)[1]} "
                "with aspect-fit padding")
        if infos[i]["fps"] != ((req.get("output") or {}).get("fps")
                               or {"num": 24, "den": 1}):
            warnings.append(
                f"source {i} was retimed from {infos[i]['fps']['num']}/"
                f"{infos[i]['fps']['den']} to the output rate")

    graph = build_continuation(op, req, staged, raw, f"{op}-{jid[:8]}",
                               latent_only=(mode == "latent"), plan=plan)
    with open(os.path.join(work_dir, "graph.json"), "w") as f:
        json.dump(graph, f, indent=1)

    pid = str(uuid.uuid4())
    progress = _status_callback(jid, hb, op)

    # H3 needs the GPU and the full card: the TTS workers hold 6-8 GB.
    with tts_guard.paused(log) as guard:
        comfy_client.ensure_server(log)
        t0 = time.time()
        # Journalled BEFORE submission and fsynced. If it cannot be recorded
        # the task fails here, because submitting work we cannot name is the
        # exact situation the journal exists to prevent.
        try:
            _journal(jid, pid, "submitting", op=op)
        except OSError as exc:
            raise FootageError(
                f"could not record prompt {pid} in the journal ({exc}); "
                "refusing to submit work that could not be reconciled "
                "afterwards") from exc
        try:
            pid = comfy_client.submit(graph, prompt_id=pid)
        except BaseException:
            # The POST may have been received even though the response was
            # not. Reconcile rather than assume, and never resubmit.
            _journal(jid, pid, "submit-uncertain", op=op)
            state = _prompt_state(pid)
            if state == LIVE:
                log(f"prompt {pid} IS on the queue despite the failed "
                    "response; aborting it rather than resubmitting")
                try:
                    _abort_prompt(pid, log)
                except FootageError as exc:
                    guard.hold(str(exc))
                    raise
            elif state == UNKNOWN:
                # "Cannot tell" is not "not running". Releasing 6-8 GB of TTS
                # here could starve a render that is very much alive.
                guard.hold(
                    f"submission of prompt {pid} was uncertain and ComfyUI "
                    "could not be read, so the GPU may be in use")
            raise
        _journal(jid, pid, "queued", op=op)
        log(f"comfyui prompt {pid} ({op})")
        try:
            outputs = comfy_client.wait(
                pid, progress, cancel_check, timeout_seconds, since_iso=None,
                # This runner owns its cancellation. The default path calls a
                # bare /interrupt, which stops whatever is RUNNING — possibly
                # another engine's prompt on the same ComfyUI.
                global_interrupt=False)
        except BaseException:
            # Never resume the TTS workers while our prompt may still be on
            # the GPU: they take 6-8 GB and would starve a run that is still
            # going, turning a clean failure into a pathologically slow one.
            log(f"aborting comfyui prompt {pid} before releasing the GPU")
            _journal(jid, pid, "aborting", op=op)
            try:
                _abort_prompt(pid, log)
            except FootageError as exc:
                # Could not prove the GPU is free: hold the workers down.
                _journal(jid, pid, "abort-unconfirmed", op=op)
                guard.hold(str(exc))
                raise
            _journal(jid, pid, "aborted", op=op)
            raise
    _journal(jid, pid, "done", op=op)
    elapsed = time.time() - t0

    saved, result_item = _saved_path(outputs)
    if not saved:
        raise FootageError(f"no clip pair in the obvpm outputs: {json.dumps(outputs)[:400]}")
    mp4 = saved if os.path.isabs(saved) else os.path.join(
        config.COMFYUI_DIR, "output", saved)
    sidecar = mp4[:-4] + ".mctx.safetensors"
    if not os.path.isfile(mp4):
        raise FootageError(f"obvpm reported {mp4} but it is not on disk")

    info = probe_media(mp4)
    digest, size = sha256_size(mp4)
    header = read_mctx_header(sidecar) if os.path.isfile(sidecar) else {}

    # Context may be returned ONLY when it is bound to these exact bytes.
    ctx_out = None
    if header:
        if header.get("self_id") != digest:
            warnings.append(
                "the sidecar does not identify this delivered file, so no "
                "context is attached to this take")
        elif os.path.getsize(sidecar) > MAX_CONTEXT_BYTES:
            raise FootageError(
                f"context artifact is {os.path.getsize(sidecar)} bytes, over the "
                f"{MAX_CONTEXT_BYTES} admission limit; refusing rather than "
                "discarding it — the local copy is kept in the task directory")
        else:
            ctx_out = {"local": sidecar, "sha256": sha256_size(sidecar)[0],
                       "size": os.path.getsize(sidecar), "media_sha256": digest}

    pins = header.get("pins")
    if isinstance(pins, str):
        pins = json.loads(pins)
    pin_windows = verify_recipe(pins, sources, warnings, plan)
    for pw in pin_windows:
        if pw["source_frames"] and pw["source_frames"] != PIN_WINDOW:
            warnings.append(
                f"context window was reduced from {PIN_WINDOW} to "
                f"{pw['source_frames']} frames because the source had only "
                "that much delivered footage available; your cut was not moved")

    d = int(header.get("delivered_frames") or info["frame_count"])
    if d != info["frame_count"]:
        raise FootageError(
            f"sidecar says {d} delivered frames but the file decodes {info['frame_count']}")
    log(f"{op}: {elapsed:.1f}s, raw {header.get('raw_frames')} "
        f"head {header.get('pinned_head_frames')} tail {header.get('pinned_tail_frames')} "
        f"-> delivered {d}")

    return {"_media_local": mp4, "_context_local": ctx_out,
            "media": {"bucket": config.BUCKET, "path": None, "sha256": digest,
                      "size": size, "info": info},
            "lineage": {"source_take_ids": [s.get("take_id") for s in sources],
                        # the USER's windows, verbatim - never the narrower pins
                        "source_trims": [{"take_id": s.get("take_id"),
                                          "in_frame": int(s.get("in_frame") or 0),
                                          "out_frame": int(s.get("out_frame") or 0)}
                                         for s in sources],
                        "pin_windows": pin_windows},
            "seams": _seams(result_item, plan),
            "sampling": {"context_mode": resolved_mode,
                         "requested_new_frames": int(gen.get("new_frames") or 24),
                         "delivered_new_frames": d,
                         "sampled_frames": int(header.get("raw_frames") or raw),
                         "held_prefix_frames": int(header.get("pinned_head_frames") or 0),
                         "held_suffix_frames": int(header.get("pinned_tail_frames") or 0),
                         "seed": int(gen.get("seed") or 0),
                         "elapsed_seconds": round(elapsed, 2)},
            "warnings": warnings}


def _journal_path():
    return os.path.join(config.CACHE_DIR, "footage-prompts.jsonl")


def _journal(task_id, prompt_id, state, op=None):
    """Append one task -> prompt record, DURABLY.

    Written before submission and at every state change so a crash, cancel
    or lost response can be reconciled against ComfyUI instead of
    resubmitting work that may already be running. A pre-submit record that
    is not on disk is worse than useless — it makes a duplicate generation
    look safe — so this flushes and fsyncs, and RAISES if it cannot. The
    earlier version swallowed every disk error, which meant the one failure
    mode the journal exists to prevent was also the one it hid.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(_journal_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_id": task_id, "prompt_id": prompt_id,
            "operation": op, "state": state}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _journal_last(task_id):
    """The most recent journal record for one task, or None.

    The journal was write-only: nothing read it, so an uncertain attempt
    left a perfect record that no retry ever consulted.
    """
    try:
        with open(_journal_path(), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    mine = [r for r in rows if r.get("task_id") == task_id]
    return mine[-1] if mine else None


# Terminal states never need reconciling; anything else may still be live.
_SETTLED = ("done", "aborted", "failed", "reconciled-complete", "reconciled-gone")

LIVE, GONE, UNKNOWN = "live", "gone", "unknown"


def _queue_snapshot():
    """(running, pending) prompt ids, or None when the answer is UNKNOWN.

    Validates transport AND shape. The earlier version called .json() on
    whatever came back, so a 500 body like {"error": "unavailable"} yielded
    empty lists and read as a drained queue — the worst possible mistake
    here, because "drained" releases the TTS workers onto a busy GPU.
    """
    from videogen import comfy_client
    import httpx
    try:
        r = httpx.get(comfy_client._url("/queue"), timeout=10)
    except Exception:                                    # noqa: BLE001
        return None
    if r.status_code != 200:
        return None
    try:
        q = r.json()
    except ValueError:
        return None
    if not isinstance(q, dict) or not (
            isinstance(q.get("queue_running"), list)
            and isinstance(q.get("queue_pending"), list)):
        return None
    try:
        return ([i[1] for i in q["queue_running"]],
                [i[1] for i in q["queue_pending"]])
    except (IndexError, TypeError):
        return None


def _prompt_state(pid):
    """LIVE, GONE or UNKNOWN — never a bare boolean.

    'Not on the queue' and 'cannot tell' are different answers and the
    caller must be able to act differently: one permits release, the other
    must not.
    """
    snap = _queue_snapshot()
    if snap is None:
        return UNKNOWN
    if pid in snap[0] or pid in snap[1]:
        return LIVE
    # Off the queue can mean finished. History is the confirmation, and if
    # history cannot be read either then the honest answer is still UNKNOWN.
    from videogen import comfy_client
    import httpx
    try:
        r = httpx.get(comfy_client._url(f"/history/{pid}"), timeout=10)
        if r.status_code != 200:
            return UNKNOWN
        h = r.json()
    except Exception:                                    # noqa: BLE001
        return UNKNOWN
    if isinstance(h, dict) and pid in h:
        return GONE
    return GONE


def _status_callback(jid, hb, label, lo=10, hi=90):
    """A wait() status callback that actually reports progress.

    render_worker passes a heartbeat.Heartbeat INSTANCE, which is NOT
    callable: it carries a .progress int that its own thread publishes. The
    previous version called hb(...) and swallowed the TypeError twice, so
    every phase, fraction and ETA was dropped — exactly as before it was
    written. This mirrors video_gen._run_prompt, which is the working
    pattern on this worker.
    """
    from videogen import estimate

    def on_status(phase, frac, eta):
        pct = int(lo + (hi - lo) * max(0.0, min(1.0, frac or 0.0)))
        if hb is not None:
            hb.progress = pct
        try:
            db.set_phase(
                jid, f"{label}: {phase} - ~{estimate.fmt_eta(eta)} left"[:120], pct)
        except Exception:                                # noqa: BLE001
            pass                    # telemetry must never fail a render
    return on_status


def _abort_prompt(pid, log, wait_seconds=120):
    """Interrupt OUR prompt and wait for the GPU to actually be released.

    Deliberately narrow: it interrupts the running prompt and deletes only
    this pid from the pending queue. It never clears the queue wholesale —
    other engines' work may be waiting there.
    """
    from videogen import comfy_client
    import httpx

    def queue_state():
        """(running_ids, pending_ids) or None when the queue cannot be read."""
        try:
            q = httpx.get(comfy_client._url("/queue"), timeout=10).json()
        except Exception:                                # noqa: BLE001
            return None
        return ([i[1] for i in (q.get("queue_running") or [])],
                [i[1] for i in (q.get("queue_pending") or [])])

    state = queue_state()
    if state is None:
        # Unknown state: do NOT interrupt, because an unconditional interrupt
        # would hit whatever is running, which may be another engine's prompt.
        raise FootageError(
            f"could not read the ComfyUI queue while aborting prompt {pid}, so "
            "ownership of the running job is unknown and nothing was "
            "interrupted. TTS stays paused. Recover by checking "
            f"{comfy_client._url('/queue')} and, if {pid} is still there, "
            "cancelling it before restarting the TTS workers.")
    running, pending = state
    if pid in running:
        # Targeted: v0.37 honours a prompt_id on /interrupt, so this can never
        # stop a prompt we do not own.
        try:
            httpx.post(comfy_client._url("/interrupt"),
                       json={"prompt_id": pid}, timeout=10)
        except Exception:                                # noqa: BLE001
            pass
    if pid in pending:
        try:
            httpx.post(comfy_client._url("/queue"),
                       json={"delete": [pid]}, timeout=10)
        except Exception:                                # noqa: BLE001
            pass
    if pid not in running and pid not in pending:
        log(f"prompt {pid} was already off the queue; nothing to interrupt")
        return

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        state = queue_state()
        if state is None:
            raise FootageError(
                f"lost contact with ComfyUI while draining prompt {pid}; TTS "
                "stays paused because the GPU may still be in use.")
        if pid not in state[0] and pid not in state[1]:
            log(f"prompt {pid} is off the queue")
            return
        time.sleep(2)
    # Fail CLOSED. Returning here would let the caller's tts_guard exit and
    # resume 6-8 GB of TTS workers into a render that is demonstrably still
    # on the GPU.
    raise FootageError(
        f"prompt {pid} was still on the ComfyUI queue {wait_seconds}s after "
        "being interrupted, so the GPU cannot be assumed free and the TTS "
        "workers have not been restarted. Recover by cancelling it at "
        f"{comfy_client._url('/queue')} and restarting the TTS workers.")


def _saved_path(outputs):
    """obvpm's saver returns the written path as a STRING output, and its
    preview node emits ui.h3_result — neither is a 'videos' entry, which is
    exactly why video_gen's collector cannot be reused here."""
    path, item = None, {}
    for node in (outputs or {}).values():
        for key in ("path", "text", "string"):
            v = node.get(key)
            if isinstance(v, list) and v and isinstance(v[0], str) \
                    and v[0].endswith(".mp4") and path is None:
                path = v[0]
        for entry in (node.get("h3_result") or []):
            clip = entry.get("clip")
            if isinstance(clip, str) and clip.endswith(".mp4"):
                path, item = clip, entry
    return path, item


def _seams(item, plan=None):
    """Departure and arrival seam measurements, kept as SEPARATE objects.

    obvpm reports seam for the departure join and seam2 for the arrival, and
    they are not interchangeable: a two-sided operation that reported one
    number would hide whichever join was worse. Presence of a number is not
    approval of the join — it is a measurement for a human to read.
    """
    # obvpm emits seam then seam2 in the order the pins were applied, so the
    # role comes from the PLAN, not from the key name. prepend has exactly one
    # join and it is an ARRIVAL — labelling it "departure" because it arrived
    # in the "seam" field would misreport which end of the clip was measured.
    roles = [s["role"] for s in (plan or [])] or ["departure", "arrival"]
    out = {}
    for key, role in zip(("seam", "seam2"), roles):
        v = item.get(key)
        if v in (None, "", {}):
            continue
        # The contract is {score?, warning?, details?} and anything else is
        # dropped on parse, so the raw upstream object has to ride inside
        # details or the whole measurement arrives as {}. Upstream's shape
        # also varies — the pixel scan adds fields the latent one does not —
        # so details carries it verbatim rather than being enumerated here.
        entry = {"details": v}
        if isinstance(v, dict):
            score = v.get("ratio")
            if isinstance(score, (int, float)):
                entry["score"] = float(score)
            verdict = v.get("verdict")
            if verdict and verdict != "seamless":
                entry["warning"] = (
                    f"the {role} join measures {verdict!r}"
                    + (f" ({score}x the clip's own motion)"
                       if isinstance(score, (int, float)) else ""))
        else:
            entry["score"] = float(v) if isinstance(v, (int, float)) else None
            if entry["score"] is None:
                entry.pop("score")
        out[role] = entry
    return out


def _canvas(req):
    out = req.get("output") or {}
    return int(out.get("width") or 832), int(out.get("height") or 480)


def _h3_base(prompt, seed, length, width, height):
    from videogen import graphs
    if not (prompt or "").strip():
        raise FootageError("a continuation needs a non-empty prompt")
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": graphs.CHECKPOINTS["fl2va"], "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0],
                            "lora_name": graphs.TURBO_LORAS["fl2va"][8], "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": graphs.TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": graphs.VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": graphs.AUDIO_VAE}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                            "width": width, "height": height, "length": length}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler",
                  "inputs": {"model": ["lora", 0], "scheduler": "simple",
                             "steps": 8, "denoise": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }


def verify_recipe(pins, sources, warnings, plan=None):
    """Map each pin's content hash back to a requested source.

    A NONEMPTY hash that matches no requested source is a hard failure, not a
    warning: it means the take was continued from something the caller did not
    ask for, and no amount of downstream validation can recover that. A BLANK
    hash is the pixel path, which has no content-addressed origin at all, so
    its provenance comes from the request and is flagged.
    """
    # obvpm identifies a source by its clip's self_id, which is the sha256 of
    # the MEDIA file — not of the sidecar. context.sha256 is the sidecar's own
    # bytes and will never appear in a recipe; context.media_sha256 is the
    # media hash restated, so both of those are accepted and the sidecar's own
    # digest deliberately is not.
    by_hash = {}
    for s in sources:
        for digest in ((s.get("media") or {}).get("sha256"),
                       (s.get("context") or {}).get("media_sha256")):
            if digest:
                by_hash[digest] = s.get("take_id")
    # A pixel pin has no content-addressed origin, so its identity comes from
    # the boundary that produced it: specs are emitted in plan order and each
    # names the placement it was built with. That is request provenance, not a
    # hash guess — deliberately so, because the same media can appear in two
    # clips and a hash could not tell them apart.
    resolved = []
    for idx, p in enumerate(pins or []):
        sid = (p.get("source_id") or "").strip()
        if not sid:
            spec = plan[idx] if plan and idx < len(plan) else None
            if spec is None:
                raise FootageError(
                    f"pin {idx} has no content hash and no planned boundary to "
                    "identify it; refusing to record unattributed lineage.")
            if p.get("place") and p["place"] != spec["place"]:
                raise FootageError(
                    f"pin {idx} was placed {p['place']!r} but the planned "
                    f"boundary was {spec['place']!r}; refusing to record "
                    "lineage that does not match the graph.")
            take_id = spec["take_id"]
            warnings.append(
                f"the {spec['place']} window came from re-encoded pixels, so "
                "its origin is recorded from the request rather than proven "
                "by content hash")
        else:
            take_id = by_hash.get(sid)
            if take_id is None:
                raise FootageError(
                    "the generated take was pinned to content hash "
                    f"{sid[:12]}... which is not any requested source. Refusing "
                    "to record false lineage.")
        resolved.append({
            "take_id": take_id, "placement": p.get("place"),
            "source_start_frame": int(p.get("source_start") or 0),
            "source_frames": int(p.get("source_frames") or 0),
            "source_space": "raw",
            "context_mode": "pixel" if not sid else "latent"})
    return resolved


# ------------------------------------------------------------ entry point

def run(job, repo, work_dir, hb, log, cancel_check, timeout_seconds):
    params = job.get("params") or {}
    req = params.get("footage") or {}
    if int(req.get("schema_version") or 0) != CONTRACT_VERSION:
        raise FootageError(f"footage contract v{CONTRACT_VERSION} required, "
                           f"got schema_version {req.get('schema_version')!r}")
    op = req.get("operation")
    jid = job["id"]
    org, pjob = req.get("org_id"), req.get("job_id")
    log(f"footage {op} org={org} job={pjob} task={jid}")

    if op == "capabilities":
        # lineage is REQUIRED on every operation by the result envelope, even
        # where there are no sources at all — the website validates the whole
        # shape and rejects a manifest that omits it.
        body = {"capabilities": capabilities_block(),
                "lineage": {"source_take_ids": [], "source_trims": []},
                "warnings": []}
    elif op == "probe":
        body = op_probe(req, work_dir, log)
    elif op == "assemble":
        body = op_assemble(req, work_dir, hb, log)
    elif op in ("extend", "prepend", "bridge", "loop"):
        # Gate on the SAME list capabilities advertises, before any staging,
        # download or GPU access. Checking GENERATION_ADAPTER_READY alone let
        # run() serve operations the manifest never offered.
        if not GENERATION_ADAPTER_READY or op not in PROVEN_GENERATION_OPS:
            raise FootageError(
                f"operation {op!r} is not available on this worker yet; "
                f"capabilities offers {list(PROVEN_GENERATION_OPS)!r}")
        body = op_continuation(op, req, jid, work_dir, hb, log,
                               cancel_check, timeout_seconds)
    else:
        raise FootageError(f"unknown footage operation {op!r}")

    # Artifacts FIRST, manifest last: a manifest must never reference an
    # object that has not landed.
    media_local = body.pop("_media_local", None)
    ctx_local = body.pop("_context_local", None)
    if media_local:
        remote = f"footage/{org}/{pjob}/{jid}/{os.path.basename(media_local)}"
        db.upload_file(remote, media_local, "video/mp4")
        body["media"]["path"] = remote
        log(f"uploaded media -> {config.BUCKET}/{remote}")
    if ctx_local:
        # Context rides with the take and is never swept by a preview TTL.
        remote = f"footage/{org}/{pjob}/{jid}/context.mctx.safetensors"
        db.upload_file(remote, ctx_local["local"], "application/octet-stream")
        body["context"] = {"bucket": config.BUCKET, "path": remote,
                           "sha256": ctx_local["sha256"], "size": ctx_local["size"],
                           "media_sha256": ctx_local["media_sha256"]}
        log(f"uploaded context -> {config.BUCKET}/{remote}")

    manifest = {"schema_version": CONTRACT_VERSION, "task_id": jid,
                "org_id": org, "job_id": pjob,
                "request_key": req.get("request_key"), "operation": op}
    manifest.update(body)
    # Belt and braces on the envelope's required members, so a new operation
    # can never ship a manifest the website will reject for a missing field.
    manifest.setdefault("capabilities", capabilities_block())
    manifest.setdefault("lineage", {"source_take_ids": [], "source_trims": []})
    manifest.setdefault("warnings", [])

    local = os.path.join(work_dir, "manifest.json")
    with open(local, "w") as f:
        json.dump(manifest, f, indent=1)
    # Also place it at the contract's canonical path. run_job() will upload the
    # same bytes to outputs/<task>.json as it does for every engine; the
    # platform reads the canonical path.
    db.upload_file(f"footage/{org}/{pjob}/{jid}/manifest.json", local, "application/json")
    log(f"manifest -> {config.BUCKET}/footage/{org}/{pjob}/{jid}/manifest.json")
    return local, "json", "application/json"
