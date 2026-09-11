"""ffmpeg / ffprobe steps for the aftereffects engine: ProRes 4444 master,
VP9 alpha review copy, frame/alpha verification, contact sheet."""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

import proc  # noqa: E402
from aftereffects.errors import AEError  # noqa: E402

ALPHA_PIX_FMTS = {"yuva444p10le", "yuva444p12le", "yuva444p", "yuva420p", "yuva422p", "rgba", "argb",
                  "bgra", "abgr", "rgba64le", "rgba64be", "gbrap", "gbrap10le", "gbrap12le", "gbrap16le",
                  "ya8", "ya16le", "ya16be", "yuva420p10le", "yuva422p10le", "pal8"}


def ffmpeg_dir():
    """Same resolution order as the hyperframes runner: FFMPEG_DIR, winget Gyan, PATH."""
    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "ffmpeg.exe")):
        return env_dir
    local = os.environ.get("LOCALAPPDATA", "")
    for hit in sorted(glob.glob(os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*",
                                             "ffmpeg-*", "bin", "ffmpeg.exe")), reverse=True):
        return os.path.dirname(hit)
    found = shutil.which("ffmpeg")
    return os.path.dirname(found) if found else None


def tool(name):
    d = ffmpeg_dir()
    if not d:
        raise AEError("RENDER_FAILED", "ffmpeg not found (FFMPEG_DIR, winget Gyan.FFmpeg, or PATH)")
    return os.path.join(d, name + ".exe")


def version():
    try:
        out = subprocess.run([tool("ffmpeg"), "-version"], capture_output=True, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
        return out.splitlines()[0].split(" Copyright")[0].replace("ffmpeg version ", "")
    except Exception:  # noqa: BLE001
        return None


def _run(cmd, timeout_s, cancel_check, log, on_line=None):
    from aftereffects.clocked import run_clocked
    log(f"  $ {' '.join(os.path.basename(cmd[0]) if i == 0 else c for i, c in enumerate(cmd))}"[:600])
    rc, tail = run_clocked(cmd, None, on_line or (lambda l: None), cancel_check, max(1, timeout_s))
    if rc != 0:
        raise AEError("RENDER_FAILED", f"{os.path.basename(cmd[0])} exit code {rc}: " + " | ".join(tail[-6:]),
                      {"exit_code": rc, "tail": tail[-15:]})


def probe(path):
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if out.returncode != 0:
        raise AEError("VERIFY_FAILED", f"ffprobe failed on {os.path.basename(path)}: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout or "{}")
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise AEError("VERIFY_FAILED", f"no video stream in {os.path.basename(path)}")
    return data, v


def count_frames(path):
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        return int(out.stdout.strip().split(",")[0])
    except ValueError:
        raise AEError("VERIFY_FAILED", f"could not count frames of {os.path.basename(path)}: {out.stderr[:200]}")


def _fps_of(stream):
    num, _, den = (stream.get("r_frame_rate") or "0/1").partition("/")
    try:
        return int(num) / int(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def sequence_inputs(pattern, fps):
    """ffmpeg input args for a glob like .../frame_*.png (also validates the files)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise AEError("RENDER_FAILED", f"no frames matched {pattern}")
    first = os.path.basename(files[0])
    m = re.match(r"^(.*?)(\d+)(\.\w+)$", first)
    if not m:
        raise AEError("RENDER_FAILED", f"unexpected frame name {first}")
    width = len(m.group(2))
    start = int(m.group(2))
    printf = os.path.join(os.path.dirname(pattern), f"{m.group(1)}%0{width}d{m.group(3)}")
    return ["-framerate", f"{fps:g}", "-start_number", str(start), "-i", printf], len(files)


# Straight alpha out of AE's premultiplied "Lossless with Alpha": divide each
# colour plane by the alpha plane. ffmpeg's own unpremultiply filter is a
# no-op on this input (measured on 9.0.1: 128/128 -> 127), blend=divide is
# exact (128/128 -> 255, text edges 253 at alpha 183) and runs in well under
# a second for 150 frames of 1080x1920.
UNPREMULTIPLY_VF = ("format=gbrap,split[c][a2];[c]format=gbrp[cc];[a2]alphaextract,format=gray,split[a3][a4];"
                    "[a3]format=gbrp[ab];[cc][ab]blend=all_mode=divide[d];[d][a4]alphamerge,format=gbrap")


def encode_prores4444(src_args, dst, fps, timeout_s, cancel_check, log, vf=None):
    cmd = [tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-stats", *src_args,
           *(["-vf", vf] if vf else []),
           "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
           "-r", f"{fps:g}", "-an", "-movflags", "+faststart", dst]
    _run(cmd, timeout_s, cancel_check, log)
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise AEError("RENDER_FAILED", "ProRes 4444 master was not written")


def encode_vp9_alpha(master, dst, timeout_s, cancel_check, log, crf=30):
    cmd = [tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", master,
           "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-crf", str(crf), "-b:v", "0",
           "-row-mt", "1", "-deadline", "good", "-cpu-used", "2", "-an", dst]
    _run(cmd, timeout_s, cancel_check, log)
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise AEError("RENDER_FAILED", "VP9 alpha review copy was not written")


def _bits(pix_fmt):
    m = re.search(r"(\d+)(?:le|be)$", pix_fmt or "")
    return int(m.group(1)) if m else 8


def alpha_stats(path, work_dir, timeout_s, cancel_check, log, pix_fmt=None):
    """Per-frame mean/min/max of the alpha plane via alphaextract + signalstats,
    normalized to 0..255 full range.

    signalstats sees the extracted gray plane as limited-range luma at the
    source bit depth: a 12-bit ProRes 4444 master reports transparent as 256
    and opaque as 3760 (16 and 235 << 4). The raw alpha round-trips to 0/255
    (checked with -pix_fmt gray), so the numbers are undone here rather than
    trusted as they come."""
    stats_file = os.path.join(work_dir, "alpha-stats.txt")
    if os.path.exists(stats_file):
        os.remove(stats_file)
    esc = stats_file.replace("\\", "/").replace(":", "\:")
    cmd = [tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", f"alphaextract,signalstats,metadata=print:file='{esc}'", "-f", "null", "-"]
    _run(cmd, timeout_s, cancel_check, log)
    if pix_fmt is None:
        _, v = probe(path)
        pix_fmt = v.get("pix_fmt")
    scale = float(1 << (_bits(pix_fmt) - 8))

    def norm(v):
        return max(0.0, min(255.0, (v / scale - 16.0) * 255.0 / 219.0))

    frames = []
    cur = {}
    if os.path.exists(stats_file):
        with open(stats_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("frame:"):
                    if cur:
                        frames.append(cur)
                    cur = {}
                elif "lavfi.signalstats." in line:
                    k, _, v = line.strip().partition("=")
                    key = k.rsplit(".", 1)[1]
                    try:
                        cur[key] = float(v)
                    except ValueError:
                        pass
        if cur:
            frames.append(cur)
    if not frames:
        raise AEError("ALPHA_MISSING", "could not measure the alpha plane (alphaextract produced no statistics)")
    avgs = [norm(f.get("YAVG", 0.0)) for f in frames]
    mins = [norm(f.get("YMIN", 0.0)) for f in frames]
    maxs = [norm(f.get("YMAX", 0.0)) for f in frames]
    return {"frames": len(frames), "scale": "0..255", "mean_alpha_min": round(min(avgs), 2),
            "mean_alpha_max": round(max(avgs), 2), "mean_alpha_avg": round(sum(avgs) / len(avgs), 2),
            "frames_with_transparent_pixels": sum(1 for m in mins if m < 8),
            "frames_with_opaque_pixels": sum(1 for m in maxs if m > 247),
            "frames_fully_transparent": sum(1 for m in maxs if m < 8),
            "frames_fully_opaque": sum(1 for m in mins if m > 247)}


def alpha_range_raw(path, frame, decoder=None):
    """(min, max) of the 8-bit alpha of one frame, decoded to raw gray: the
    ground truth the statistics above are checked against."""
    cmd = [tool("ffmpeg"), "-v", "error"] + (["-c:v", decoder] if decoder else []) +           ["-i", path, "-vf", f"select=eq(n\,{int(frame)}),alphaextract,format=gray", "-frames:v", "1",
           "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if out.returncode != 0 or not out.stdout:
        raise AEError("ALPHA_MISSING", f"could not decode the alpha of frame {frame} of {os.path.basename(path)}: "
                      f"{out.stderr.decode('utf-8', 'replace')[-200:]}")
    return min(out.stdout), max(out.stdout)


def contact_sheet(master, dst, frames, timeout_s, cancel_check, log, columns=6, thumb_h=480):
    """Sampled frames over a light row and a dark row, so alpha edges can be judged by eye."""
    step = max(1, frames // columns)
    _, v = probe(master)
    w, h = int(v["width"]), int(v["height"])
    tw = max(2, int(round(w * thumb_h / h / 2)) * 2)
    fc = (f"[0:v]select='not(mod(n\\,{step}))',scale={tw}:{thumb_h},setpts=N/TB,split=2[l][d];"
          f"color=c=#EDEDED:s={tw}x{thumb_h}:r=1,format=rgba[lb];color=c=#141414:s={tw}x{thumb_h}:r=1,format=rgba[db];"
          f"[lb][l]overlay=shortest=1:format=auto[lo];[db][d]overlay=shortest=1:format=auto[do];"
          f"[lo]tile={columns}x1[lt];[do]tile={columns}x1[dt];[lt][dt]vstack,format=rgb24")
    cmd = [tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", master, "-filter_complex", fc,
           "-frames:v", "1", "-update", "1", dst]
    _run(cmd, timeout_s, cancel_check, log)
    if not os.path.exists(dst):
        raise AEError("VERIFY_FAILED", "contact sheet was not written")


def verify_master(path, comp, work_dir, timeout_s, cancel_check, log):
    """Dimensions, fps, frame count/duration and real alpha content of the master."""
    data, v = probe(path)
    checks = {"codec": v.get("codec_name"), "profile": v.get("profile"), "pix_fmt": v.get("pix_fmt"),
              "width": int(v.get("width", 0)), "height": int(v.get("height", 0)), "fps": round(_fps_of(v), 3),
              "duration_s": float(data.get("format", {}).get("duration") or v.get("duration") or 0)}
    if checks["pix_fmt"] not in ALPHA_PIX_FMTS:
        raise AEError("ALPHA_MISSING", f"master pix_fmt {checks['pix_fmt']} carries no alpha channel")
    if (checks["width"], checks["height"]) != (comp["width"], comp["height"]):
        raise AEError("VERIFY_FAILED", f"master is {checks['width']}x{checks['height']}, request was {comp['width']}x{comp['height']}")
    if abs(checks["fps"] - comp["fps"]) > 0.01:
        raise AEError("VERIFY_FAILED", f"master fps {checks['fps']} != requested {comp['fps']}")
    n = count_frames(path)
    checks["frames"] = n
    if n != comp["frames"]:
        raise AEError("RENDER_INCOMPLETE", f"master has {n} frames, expected {comp['frames']}")
    if abs(checks["duration_s"] - comp["duration_s"]) > 1.5 / comp["fps"]:
        raise AEError("RENDER_INCOMPLETE", f"master duration {checks['duration_s']:.3f}s, expected {comp['duration_s']}s")
    t0 = time.monotonic()
    a = alpha_stats(path, work_dir, timeout_s, cancel_check, log, pix_fmt=checks["pix_fmt"])
    checks["alpha"] = a
    mid = n // 2
    a["raw_mid_frame"] = {"frame": mid, "min": None, "max": None}
    a["raw_mid_frame"]["min"], a["raw_mid_frame"]["max"] = alpha_range_raw(path, mid)
    checks["alpha_measure_s"] = round(time.monotonic() - t0, 2)
    if a["frames"] != n:
        raise AEError("VERIFY_FAILED", f"alpha statistics cover {a['frames']} frames of {n}")
    if a["frames_with_transparent_pixels"] == 0:
        raise AEError("ALPHA_MISSING", "no frame has any transparent pixel: the overlay is fully opaque")
    if a["frames_with_opaque_pixels"] == 0:
        raise AEError("ALPHA_MISSING", "no frame has any opaque pixel: nothing was drawn")
    return checks


def decoded_pix_fmt(path, decoder=None):
    """The pixel format an actual decode yields (showinfo), which for VP9 alpha
    differs from what ffprobe reports off the container."""
    cmd = [tool("ffmpeg"), "-hide_banner"] + (["-c:v", decoder] if decoder else []) +           ["-i", path, "-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    m = re.search(r"fmt:(\S+)", out.stderr or "")
    return m.group(1) if m else None


def verify_review(path, frame):
    """The VP9 review copy: ffprobe reports the container's yuv420p; only the
    libvpx decoder reconstructs the alpha plane, so the check decodes."""
    _, v = probe(path)
    info = {"codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"),
            "size": os.path.getsize(path), "container_pix_fmt": v.get("pix_fmt")}
    info["pix_fmt"] = decoded_pix_fmt(path, decoder="libvpx-vp9")
    if info["pix_fmt"] not in ALPHA_PIX_FMTS:
        raise AEError("ALPHA_MISSING", f"review copy decodes to {info['pix_fmt']}: no alpha plane")
    lo, hi = alpha_range_raw(path, frame, decoder="libvpx-vp9")
    info["raw_alpha_frame"] = {"frame": frame, "min": lo, "max": hi}
    if lo > 8 or hi < 247:
        raise AEError("ALPHA_MISSING", f"review copy frame {frame} alpha spans {lo}..{hi}: not a usable alpha")
    return info


def source_has_alpha(path):
    _, v = probe(path)
    return v.get("pix_fmt") in ALPHA_PIX_FMTS, v.get("pix_fmt"), v.get("codec_name")
