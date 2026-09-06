"""ffmpeg / RIFE helpers for the studio and the turntable flow. Stdlib only:
the render-farm venv has no numpy or Pillow, and the arrays here are a few
thousand floats, so plain lists are fine.

RIFE is the portable rife-ncnn-vulkan build at config.RIFE_DIR (user-space,
no admin). Its v4.6 model accepts an arbitrary target frame count, so 24 -> 60
fps is one pass at 2.5x rather than a chain of doublings.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

import config
from videogen import segments

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg():
    return segments._tool("ffmpeg")


def ffprobe():
    return segments._tool("ffprobe")


def run(cmd, cwd=None, binary=False):
    r = subprocess.run(cmd, capture_output=True, text=not binary, creationflags=NO_WINDOW, cwd=cwd,
                       **({} if binary else {"encoding": "utf-8", "errors": "replace"}))
    if r.returncode != 0:
        err = r.stderr if not binary else r.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed ({r.returncode}): {err[-800:]}")
    return r


def info(path):
    """{width, height, fps, frames, duration, has_audio} of a video file."""
    r = run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
             "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path])
    s = json.loads(r.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    frames = int(s.get("nb_read_frames") or 0)
    a = run([ffprobe(), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path])
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": round(fps, 3), "frames": frames,
            "duration": round(frames / fps, 2) if fps else 0, "has_audio": "audio" in a.stdout}


def image_size(path):
    r = run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", path])
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def thumbnail(path, dest, t=1.0, width=640):
    run([ffmpeg(), "-v", "error", "-y", "-ss", str(t), "-i", path, "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "4", dest])
    return dest


def extract_frame(path, n, dest):
    run([ffmpeg(), "-v", "error", "-y", "-i", path, "-vf", f"select=eq(n\\,{n})", "-frames:v", "1", dest])
    return dest


def extract_frames(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    run([ffmpeg(), "-v", "error", "-y", "-i", path, os.path.join(out_dir, "%08d.png")])
    return sorted(os.listdir(out_dir))


def motion_profile(path):
    """Mean absolute luma difference to the previous frame, one value per frame
    (frame 0 copies frame 1). A rotating car sits around 5 on this scale at
    832x480-1344x768; a hard cut is 20+."""
    with tempfile.TemporaryDirectory() as td:
        # Relative output path: ffmpeg's filter parser chokes on the drive colon in C:/...
        run([ffmpeg(), "-v", "error", "-i", path, "-vf",
             "tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=m.txt",
             "-f", "null", "-"], cwd=td)
        vals = [float(x) for x in re.findall(r"YAVG=([\d.]+)", open(os.path.join(td, "m.txt")).read())]
    if len(vals) > 1:
        vals[0] = vals[1]
    return vals


def smooth(v, k=5):
    return [sum(v[max(0, i - k):i + k + 1]) / len(v[max(0, i - k):i + k + 1]) for i in range(len(v))]


def plateau_of(v):
    """Typical steady-state motion: median of the upper half of the smoothed profile."""
    sm = sorted(smooth(v))
    upper = sm[len(sm) // 2:]
    return upper[len(upper) // 2] if upper else 0.0


def find_cut(motion, min_ratio=4.0, min_abs=12.0):
    """(index, plateau) of the first hard cut in a clip, index None when clean.
    The i2v model sometimes 'teleports' the subject mid-clip to reach the last
    frame; the cut shows as one frame with 4-6x the steady-state motion."""
    plateau = plateau_of(motion)
    thr = max(min_ratio * plateau, min_abs)
    for i, x in enumerate(motion):
        if i > 0 and x >= thr:
            return i, plateau
    return None, plateau


def corner_luma_profile(path):
    """Mean luma of the top-left and bottom-right 64x64 corners, per frame (0-255).
    On a plain-backdrop product shot this should stay at the backdrop's brightness;
    the i2v model sometimes drifts the scene into a grey floor / overhead view."""
    with tempfile.TemporaryDirectory() as td:
        run([ffmpeg(), "-v", "error", "-i", path, "-filter_complex",
             "[0:v]split=2[a][b];[a]crop=64:64:0:0[a1];[b]crop=64:64:iw-64:ih-64[b1];[a1][b1]hstack,"
             "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=c.txt", "-f", "null", "-"], cwd=td)
        return [float(x) for x in re.findall(r"YAVG=([\d.]+)", open(os.path.join(td, "c.txt")).read())]


def background_drift(path, reference_luma=None, tolerance=30.0, max_share=0.10):
    """(drifted, share, worst): share of frames whose corner luma is more than
    `tolerance` away from the backdrop; drifted when > max_share. The reference
    defaults to the clip's own first frames (an anchored i2v clip starts on the
    photo), which sidesteps full- vs limited-range luma differences."""
    prof = corner_luma_profile(path)
    if reference_luma is None:
        head = sorted(prof[:5])
        reference_luma = head[len(head) // 2] if head else 255.0
    bad = [x for x in prof if abs(x - reference_luma) > tolerance]
    share = len(bad) / max(len(prof), 1)
    worst = max((abs(x - reference_luma) for x in prof), default=0.0)
    return share > max_share, share, worst


def image_luma(path):
    r = run([ffmpeg(), "-v", "error", "-i", path, "-filter_complex",
             "[0:v]split=2[a][b];[a]crop=64:64:0:0[a1];[b]crop=64:64:iw-64:ih-64[b1];[a1][b1]hstack,signalstats,"
             "metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"])
    m = re.search(r"YAVG=([\d.]+)", r.stderr + r.stdout)
    return float(m.group(1)) if m else 255.0


def corner_color(src):
    r = run([ffmpeg(), "-v", "error", "-i", src, "-vf", "crop=4:4:0:0,scale=1:1", "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], binary=True)
    b = r.stdout[:3]
    return "#%02x%02x%02x" % (b[0], b[1], b[2]) if len(b) == 3 else "#ffffff"


def pad_photo(src, dest, w, h):
    """Fit a photo into w x h without cropping, letterboxed in the photo's own
    background colour (sampled at the top-left corner) so the anchor frame is
    exactly the generation canvas and nothing gets cut off."""
    color = corner_color(src)
    run([ffmpeg(), "-v", "error", "-y", "-i", src, "-vf",
         f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
         f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={color}", dest])
    return dest


def trim_copy(src, dest, start_frame, end_frame):
    """Frame-exact [start_frame, end_frame) copy, video only, near-lossless."""
    run([ffmpeg(), "-v", "error", "-y", "-i", src, "-vf",
         f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
         "-an", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", dest])
    return dest


def concat_trimmed(pieces, dest, fps=24):
    """pieces: [(path, start_frame, end_frame_exclusive)] -> one video-only mp4."""
    fc = []
    cmd = [ffmpeg(), "-v", "error", "-y"]
    for i, (path, a, b) in enumerate(pieces):
        cmd += ["-i", path]
        fc.append(f"[{i}:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]")
    fc.append("".join(f"[v{i}]" for i in range(len(pieces))) + f"concat=n={len(pieces)}:v=1:a=0[v]")
    run(cmd + ["-filter_complex", ";".join(fc), "-map", "[v]", "-r", str(fps), "-c:v", "libx264", "-crf", "10",
               "-pix_fmt", "yuv420p", dest])
    return dest


def rife(in_dir, out_dir, n_frames, model="rife-v4.6"):
    exe = os.path.join(config.RIFE_DIR, "rife-ncnn-vulkan.exe")
    if not os.path.exists(exe):
        raise RuntimeError(f"rife-ncnn-vulkan not found at {config.RIFE_DIR} (set RIFE_DIR in .env)")
    os.makedirs(out_dir, exist_ok=True)
    run([exe, "-i", in_dir, "-o", out_dir, "-n", str(n_frames), "-m", os.path.join(config.RIFE_DIR, model),
         "-g", "0", "-j", "2:4:4", "-f", "%08d.png"])
    return sorted(os.listdir(out_dir))


def encode_frames(frames_dir, dest, fps, shorter_size=None, audio_src=None, loop_audio=False, crf=14):
    """PNG sequence -> mp4 (optionally lanczos to a short edge), with audio_src's
    track under it (looped when the video is longer than the source)."""
    cmd = [ffmpeg(), "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%08d.png")]
    has_audio = bool(audio_src) and info(audio_src)["has_audio"]
    if has_audio:
        if loop_audio:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0"]
    if shorter_size:
        s = int(shorter_size)
        cmd += ["-vf", f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)':flags=lanczos"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"] if has_audio else ["-an"]
    cmd += ["-movflags", "+faststart", dest]
    run(cmd)
    return dest


def interpolate(src, dest, fps=60, shorter_size=None, tmp_root=None, log=print):
    """src (any fps) -> dest at fps via RIFE at native resolution, then lanczos
    to shorter_size, source audio carried over. ~10 s for 250 frames on the 4080."""
    inf = info(src)
    n = int(round(inf["frames"] * fps / inf["fps"]))
    tmp = tempfile.mkdtemp(prefix="rife_", dir=tmp_root)
    try:
        log(f"interpolate: {inf['frames']} frames @ {inf['fps']} -> {n} @ {fps}")
        extract_frames(src, os.path.join(tmp, "in"))
        got = rife(os.path.join(tmp, "in"), os.path.join(tmp, "out"), n)
        log(f"rife: {len(got)} frames")
        encode_frames(os.path.join(tmp, "out"), dest, fps, shorter_size, audio_src=src)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def _interp(xs, xp, fp):
    """Linear interpolation of xs on the monotonic table (xp, fp)."""
    out, j = [], 0
    for x in xs:
        while j + 1 < len(xp) - 1 and xp[j + 1] <= x:
            j += 1
        x0, x1 = xp[j], xp[j + 1]
        f0, f1 = fp[j], fp[j + 1]
        out.append(f0 if x1 == x0 else f0 + (f1 - f0) * (x - x0) / (x1 - x0))
    return out


def remap_constant_speed(src, frames_out, fps=60, density=10, tmp_root=None, log=print):
    """Rewrite src's timing so the accumulated frame-to-frame motion grows
    linearly: RIFE the clip to `density`x frames, then pick output frames at
    equal steps of cumulative motion. Removes the ease-in/out the i2v model
    puts at its anchor frames without touching the anchors themselves. Writes
    a PNG sequence to frames_out and returns (n_out, profile_before, plateau)."""
    inf = info(src)
    N = inf["frames"]
    m = motion_profile(src)
    ms = smooth(m, 5)
    floor = 0.15 * sorted(ms)[len(ms) // 2]
    ms = [max(x, floor) for x in ms]          # a static stretch must not stall the remap
    plateau = plateau_of(m)
    C = [0.0]
    for x in ms[1:]:
        C.append(C[-1] + x)
    n_out = int(round(C[-1] / plateau * fps / inf["fps"]))
    tmp = tempfile.mkdtemp(prefix="remap_", dir=tmp_root)
    try:
        extract_frames(src, os.path.join(tmp, "in"))
        got = rife(os.path.join(tmp, "in"), os.path.join(tmp, "out"), N * density)
        log(f"remap: {N} frames -> {len(got)} dense -> {n_out} at {fps} fps ({n_out / fps:.1f} s)")
        targets = [C[-1] * k / n_out for k in range(n_out)]      # endpoint excluded: the last frame equals the first
        t_src = _interp(targets, C, list(range(N)))
        if os.path.isdir(frames_out):
            shutil.rmtree(frames_out)
        os.makedirs(frames_out)
        for i, t in enumerate(t_src):
            d = int(round(t * (len(got) - 1) / max(N - 1, 1)))
            shutil.copyfile(os.path.join(tmp, "out", got[d]), os.path.join(frames_out, f"{i:08d}.png"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return n_out, m, plateau
