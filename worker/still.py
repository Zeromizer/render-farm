"""A still for a finished video, made from the file the worker already has.

The render platform draws a ~40 KB JPEG on every tile and player that has one
(rp_renders.poster_path, platform migration 0059). Without one, the browser
opens the MP4 master to show a frame, and browsers doing that were 9-12 GB/day
of Supabase egress (review 2026-10-02). Here the frame costs nothing: the
master is on this disk, about to be deleted with work_dir.

One frame, 480 px on the long edge, never a flat one (an intro fading from
black): a second in, else earlier, else later. Best effort throughout. A
render must never fail or wait over its still, so every error returns None and
the platform falls back to capturing it in the browser.
"""
import os
import subprocess

from extract.common import find_ffmpeg_tool

MAX_EDGE = 480
FLAT_STDEV = 6.0
VIDEO_EXTS = {"mp4", "mov", "webm", "m4v"}
# The worker runs windowless; without this every ffmpeg call flashes a console.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _gray_stdev(ffmpeg, path, t):
    """Luminance spread of the frame at t, from a 32x32 grey decode."""
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", path,
         "-frames:v", "1", "-vf", "scale=32:32", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True, timeout=60, creationflags=NO_WINDOW,
    )
    px = r.stdout
    if r.returncode != 0 or len(px) < 1024:
        return None
    mean = sum(px) / len(px)
    return (sum((p - mean) ** 2 for p in px) / len(px)) ** 0.5


def make_still(video_path, ext, out_path, log=print):
    """Write a JPEG still of video_path to out_path; return out_path or None."""
    if (ext or "").lower() not in VIDEO_EXTS:
        return None
    try:
        ffmpeg = find_ffmpeg_tool("ffmpeg")
        for t in (1, 0.4, 2.5):
            spread = _gray_stdev(ffmpeg, video_path, t)
            if spread is None or spread < FLAT_STDEV:
                continue
            scale = (f"scale='if(gt(iw,ih),min({MAX_EDGE},iw),-2)'"
                     f":'if(gt(iw,ih),-2,min({MAX_EDGE},ih))'")
            r = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", video_path,
                 "-frames:v", "1", "-vf", scale, "-q:v", "5", "-y", out_path],
                capture_output=True, timeout=60, creationflags=NO_WINDOW,
            )
            if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
        log("still: no usable frame (flat or undecodable); the browser will capture one")
    except Exception as e:  # never fail a render over its still
        log(f"still: skipped ({str(e)[:160]})")
    return None
