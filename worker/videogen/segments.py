"""Split a clip into fixed-frame segments and glue upscaled segments back.

WHY: the SeedVR2 graph decodes and colour-corrects the whole clip in RAM at
output resolution. 243 frames at 1664x960 took the ComfyUI process to 82 GB of
virtual memory on this 31 GB box and Windows killed it. 73 frames (3 s) is
proven fine, so long clips are upscaled as a series of short ones and joined
with the concat demuxer; the original audio track is muxed back over the join
so the soundtrack is never re-generated or re-encoded per segment.
"""
import json
import os
import subprocess

DEFAULT_SEGMENT_FRAMES = 73  # 3 s at 24 fps; proven at 1664x960 on 31 GB RAM

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_BIN = {}


def _tool(name):
    """Full path of ffmpeg/ffprobe. The worker runs from a Startup shortcut whose
    PATH predates the winget ffmpeg install (bare "ffmpeg" -> WinError 2), so
    this reuses the resolver the hyperframes runner already has for that."""
    if name not in _BIN:
        from runners.hyperframes import _ffmpeg_dir
        d = _ffmpeg_dir()
        _BIN[name] = os.path.join(d, name + ".exe") if d else name
    return _BIN[name]


def _run(cmd):
    cmd = [_tool(cmd[0])] + list(cmd[1:])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed ({r.returncode}): {r.stderr[-800:]}")
    return r


def probe(path):
    """(frame_count, fps_float, has_audio)"""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
              "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json", path])
    v = json.loads(r.stdout)["streams"][0]
    num, den = v["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    frames = int(v.get("nb_read_frames") or 0)
    a = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
              "-of", "csv=p=0", path])
    return frames, fps, "audio" in a.stdout


def split(path, frames_per_segment, work_dir, log):
    """Write <work_dir>/seg_000.mp4 ... each with exactly frames_per_segment
    frames (last one shorter), video only, near-lossless. Returns the paths
    (a single-element list when the clip already fits)."""
    total, fps, _ = probe(path)
    if total <= frames_per_segment:
        return [path], fps, total
    out = []
    start = 0
    i = 0
    while start < total:
        n = min(frames_per_segment, total - start)
        seg = os.path.join(work_dir, f"seg_{i:03d}.mp4")
        # Frame-exact: decode everything and keep [start, start+n). Re-encode
        # near-lossless so the model sees what H3 produced, not a keyframe cut.
        _run(["ffmpeg", "-v", "error", "-y", "-i", path,
              "-vf", f"select='between(n\,{start}\,{start + n - 1})',setpts=N/FRAME_RATE/TB",
              "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "8", "-pix_fmt", "yuv420p",
              "-r", f"{fps:.6f}", seg])
        out.append(seg)
        start += n
        i += 1
    log(f"segments: {total} frames @ {fps:g} fps -> {len(out)} x {frames_per_segment}")
    return out, fps, total


def lanczos(src, dest, factor=None, shorter_size=None, log=print):
    """Plain lanczos resize with ffmpeg, audio copied through. This is the
    default upscale: on clean H3 footage SeedVR2 etches fur and invents
    speckle, and a 768p -> 1080p step gains little from a restoration model.
    Instant compared with SeedVR2's ~55 s per second of video."""
    if shorter_size:
        s = int(shorter_size)
        # Short edge -> s, long edge follows the aspect ratio, both even.
        vf = f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)':flags=lanczos"
        how = f"short edge -> {s}px"
    else:
        f = float(factor or 2.0)
        vf = f"scale='trunc(iw*{f}/2)*2':'trunc(ih*{f}/2)*2':flags=lanczos"
        how = f"x{f:g}"
    _, _, has_audio = probe(src)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "medium",
           "-crf", "14", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "copy"] if has_audio else ["-an"]
    cmd += ["-movflags", "+faststart", dest]
    _run(cmd)
    log(f"lanczos upscale {how} -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def concat(segment_paths, audio_source, dest, log):
    """Concatenate upscaled segments (stream copy) and mux audio_source's
    audio track over the result (-shortest guards against a frame or two of
    drift at the tail)."""
    lst = os.path.join(os.path.dirname(dest), "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write("file '" + p.replace("\\", "/").replace("'", "'\''") + "'\n")
    _, _, has_audio = probe(audio_source)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst]
    if has_audio:
        cmd += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-map", "0:v:0", "-c:v", "copy"]
    cmd += ["-movflags", "+faststart", dest]
    _run(cmd)
    log(f"concat: {len(segment_paths)} segments -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def trim_frames(src, dest, frames, fps=24.0, log=print):
    """Keep the first `frames` video frames (near-lossless re-encode) with the audio cut a
    little past them: the decoded H3 import slices audio itself and fails when it is short."""
    _, _, has_audio = probe(src)
    frames = int(frames)
    end = frames / float(fps)
    # per-stream trims (not -frames:v / -t): the audio must run past the last video frame, and
    # ffmpeg stops every stream as soon as a global frame or time limit is reached
    fc = f"[0:v]trim=end_frame={frames},setpts=PTS-STARTPTS[v]"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src]
    if has_audio:
        fc += f";[0:a]apad=pad_dur=1,atrim=end={end + 0.2:.4f},asetpts=PTS-STARTPTS[a]"
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-filter_complex", fc, "-map", "[v]", "-an"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "10", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dest]
    _run(cmd)
    log(f"trim -> first {frames} frames ({end:.3f} s) -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def crop_exact(src, dest, width, height, log=print):
    """Centre-crop to exactly width x height (near-lossless), audio copied.
    The H3 latent refine runs on a 32-aligned canvas (1088 for a 1080 request);
    this trims the few border pixels afterwards instead of resampling."""
    _, _, has_audio = probe(src)
    vf = f"crop={int(width)}:{int(height)}"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "medium",
           "-crf", "10", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "copy"] if has_audio else ["-an"]
    cmd += ["-movflags", "+faststart", dest]
    _run(cmd)
    log(f"crop -> {int(width)}x{int(height)} -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest
