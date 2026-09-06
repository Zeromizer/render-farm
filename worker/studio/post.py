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
        run([ffmpeg(), "-v", "error", "-i", os.path.abspath(path), "-vf",
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
        run([ffmpeg(), "-v", "error", "-i", os.path.abspath(path), "-filter_complex",
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


def subject_bbox(src, tolerance=40, sample_width=192):
    """Bounding box of the subject on a plain backdrop, as fractions of the
    image (x0, y0, x1, y1), plus the pixel size: {box, width, height}. The
    subject is every pixel whose colour differs from the corner colour by
    more than `tolerance` in any channel, found on a downscaled copy. Good
    enough to compare the framing of studio car photos; not a matte."""
    w, h = image_size(src)
    sw = min(sample_width, w)
    sh = max(1, int(round(h * sw / w)))
    r = run([ffmpeg(), "-v", "error", "-i", src, "-vf", f"scale={sw}:{sh}", "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], binary=True)
    px = r.stdout
    if len(px) < sw * sh * 3:
        raise RuntimeError(f"could not read {os.path.basename(src)} for framing analysis")
    corners = [px[0:3], px[(sw - 1) * 3:(sw - 1) * 3 + 3], px[(sh - 1) * sw * 3:(sh - 1) * sw * 3 + 3],
               px[(sh * sw - 1) * 3:(sh * sw - 1) * 3 + 3]]
    bg = [sorted(c[i] for c in corners)[1] for i in range(3)]   # a robust corner colour
    x0, y0, x1, y1 = sw, sh, -1, -1
    for y in range(sh):
        row = px[y * sw * 3:(y + 1) * sw * 3]
        for x in range(sw):
            o = x * 3
            if (abs(row[o] - bg[0]) > tolerance or abs(row[o + 1] - bg[1]) > tolerance
                    or abs(row[o + 2] - bg[2]) > tolerance):
                if x < x0:
                    x0 = x
                if x > x1:
                    x1 = x
                if y < y0:
                    y0 = y
                if y > y1:
                    y1 = y
    if x1 < 0:
        return {"box": None, "width": w, "height": h}
    return {"box": (x0 / sw, y0 / sh, (x1 + 1) / sw, (y1 + 1) / sh), "width": w, "height": h}


def pad_photos_common(photos, w, h, dest_dir, min_fill=0.35):
    """Pad several same-scale photos of one subject onto the w x h canvas with ONE
    common scale factor, so the subject keeps its relative size between views
    (a side view is wider than a front view at the same distance; scaling each
    to fit would enlarge the narrower ones). photos: {name: src}. Returns
    {name: {"local": dest, "box": (x0,y0,x1,y1) on the canvas, "scale": f}}.
    Raises RuntimeError with the numbers when the photos are not compatible:
    a subject cropped by the photo edge, subject heights that disagree by more
    than 25 % (different distance / focal length), or a subject that would fill
    less than `min_fill` of the canvas height."""
    meas = {}
    for name, src in photos.items():
        m = subject_bbox(src)
        if m["box"] is None:
            raise RuntimeError(f"{name} photo: no subject found on a plain backdrop")
        x0, y0, x1, y1 = m["box"]
        touching = [side for side, v in (("left", x0 <= 0.005), ("top", y0 <= 0.005),
                                          ("right", x1 >= 0.995), ("bottom", y1 >= 0.995)) if v]
        if touching:
            raise RuntimeError(f"{name} photo: the subject touches the {'/'.join(touching)} edge; "
                               f"anchors must show the whole car with empty space around it")
        meas[name] = m
    heights = {n: (m["box"][3] - m["box"][1]) * m["height"] for n, m in meas.items()}
    ref = sorted(heights.values())[len(heights) // 2]
    off = {n: abs(v - ref) / ref for n, v in heights.items()}
    worst = max(off, key=off.get)
    if off[worst] > 0.25:
        raise RuntimeError("anchor photos are not at the same scale: subject height "
                           + ", ".join(f"{n} {int(v)} px" for n, v in heights.items())
                           + f" ({worst} is {off[worst]:.0%} off). Shoot all views from the same "
                           "distance and focal length, or crop them to a common scale first")
    # One factor for all: the largest that fits every photo on the canvas.
    scale = min(min(w / m["width"], h / m["height"]) for m in meas.values())
    fill = ref * scale / h
    if fill < min_fill:
        raise RuntimeError(f"the car would fill only {fill:.0%} of the canvas height at a common scale "
                           f"(smallest photo drives the fit); crop the photos closer to the car")
    out = {}
    for name, src in photos.items():
        m = meas[name]
        sw, sh = max(2, int(round(m["width"] * scale)) // 2 * 2), max(2, int(round(m["height"] * scale)) // 2 * 2)
        dest = os.path.join(dest_dir, f"{name}_{w}x{h}.png")
        color = corner_color(src)
        run([ffmpeg(), "-v", "error", "-y", "-i", src, "-vf",
             f"scale={sw}:{sh}:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={color}", dest])
        ox, oy = (w - sw) / 2, (h - sh) / 2
        x0, y0, x1, y1 = m["box"]
        out[name] = {"local": dest, "scale": scale,
                     "box": ((ox + x0 * sw) / w, (oy + y0 * sh) / h, (ox + x1 * sw) / w, (oy + y1 * sh) / h)}
    return out


def frame_diff(a_path, b_path):
    """Mean absolute luma difference (0-255) between two images/first frames
    of two videos, both scaled to 256 px wide. Identical anchors -> ~0."""
    with tempfile.TemporaryDirectory() as td:
        a_path, b_path = os.path.abspath(a_path), os.path.abspath(b_path)   # cwd is the temp dir below
        run([ffmpeg(), "-v", "error", "-i", a_path, "-i", b_path, "-filter_complex",
             "[0:v]trim=end_frame=1,scale=256:-2,format=gray[a];[1:v]trim=end_frame=1,scale=256:-2,format=gray[b];"
             "[a][b]blend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=d.txt",
             "-frames:v", "1", "-f", "null", "-"], cwd=td)
        vals = re.findall(r"YAVG=([\d.]+)", open(os.path.join(td, "d.txt")).read())
    return float(vals[0]) if vals else 255.0


def last_frame_png(path, dest):
    """The final frame of a video as a PNG."""
    n = info(path)["frames"]
    return extract_frame(path, max(0, n - 1), dest)


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
