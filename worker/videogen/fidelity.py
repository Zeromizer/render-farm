"""Numeric fidelity check for an upscaled clip against its source.

WHY: a latent refine re-synthesises detail, and on car footage the things
that must not drift are small (badges, grille pattern, wheel spokes, plate
lettering, dealer logo). Nothing here judges those semantically; this
measures how far the upscaled frames moved from a plain Lanczos resize of the
source, globally and per cell of a 4x4 grid, so a drifting corner or a
re-imagined badge shows up as one cell well below the frame's own mean. The
numbers are review information for the operator (with the side-by-side
compare video), never a gate: the job does not fail on them.

ffmpeg only (the worker venv has no numpy/PIL): the ssim and psnr filters
write per-frame stats files. Stats paths are relative and ffmpeg runs with
cwd=work_dir because its filter parser trips over "C:" in option values
(same trap as studio/post.py).

Thresholds were anchored on the render PC 2026-09-10 (see FIDELITY_DEFAULTS in
graphs_h3.py): a healthy latent refine scores ~0.92 SSIM / 25 dB against the
lanczos reference, well below raw SeedVR2 (0.956 / 35.7), so the floors only
catch a broken refine; the per-cell drift flag and the compare video are the
badge/plate review.
"""
import json
import os
import re
import subprocess

from videogen import segments

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
GRID = 4
_SSIM_LINE = re.compile(r"n:(\d+)\s+Y:([\d.infa]+)\s+U:([\d.infa]+)\s+V:([\d.infa]+)\s+All:([\d.infa]+)")
_PSNR_LINE = re.compile(r"n:(\d+)\s+mse_avg:([\d.infa]+)\s+mse_y:[\d.infa]+\s+mse_u:[\d.infa]+\s+mse_v:[\d.infa]+"
                        r"\s+psnr_avg:([\d.infa]+)\s+psnr_y:([\d.infa]+)")
INF_PSNR = 99.0


def _num(s):
    if s in ("inf", "nan"):
        return INF_PSNR if s == "inf" else 0.0
    return float(s)


def parse_ssim_log(text):
    """[{n, Y, U, V, All}] in frame order."""
    out = []
    for m in _SSIM_LINE.finditer(text or ""):
        out.append({"n": int(m.group(1)), "Y": _num(m.group(2)), "U": _num(m.group(3)), "V": _num(m.group(4)),
                    "All": _num(m.group(5))})
    return out


def parse_psnr_log(text):
    """[{n, mse_avg, psnr_avg, psnr_y}] in frame order (inf -> 99)."""
    out = []
    for m in _PSNR_LINE.finditer(text or ""):
        out.append({"n": int(m.group(1)), "mse_avg": _num(m.group(2)), "psnr_avg": _num(m.group(3)),
                    "psnr_y": _num(m.group(4))})
    return out


def _run(cmd, cwd):
    cmd = [segments._tool(cmd[0])] + list(cmd[1:])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=_NO_WINDOW, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed ({r.returncode}): {r.stderr[-800:]}")
    return r


def _info(path):
    """{width, height, fps, frames, has_audio} via ffprobe (kept local so this
    module never imports config, which the laptop tests cannot load)."""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
              "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path], cwd=None)
    st = json.loads(r.stdout)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    a = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
              "-of", "csv=p=0", path], cwd=None)
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": round(fps, 3),
            "frames": int(st.get("nb_read_frames") or 0), "has_audio": "audio" in a.stdout}


def _percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))
    return s[k]


def measure(src, up, work_dir, grid=GRID, thresholds=None, log=print):
    """Compare `up` (the upscaled clip) with `src` resized to up's size.

    One ffmpeg pass: global ssim + psnr, plus grid x grid cell ssims. Returns
    the fidelity dict (see summarise). Frame counts are compared on the
    shorter of the two (shortest=1), and reported."""
    up_info = _info(up)
    src_info = _info(src)
    W, H = up_info["width"], up_info["height"]
    n = grid * grid
    cw, ch = W // grid, H // grid
    parts = [f"[0:v]split={n + 1}" + "".join(f"[u{i}]" for i in range(n + 1)),
             f"[1:v]scale={W}:{H}:flags=lanczos,split={n + 1}" + "".join(f"[r{i}]" for i in range(n + 1)),
             "[u0]split=2[ug][up]", "[r0]split=2[rg][rp]",
             "[ug][rg]ssim=stats_file=ssim.log:shortest=1[sg]",
             "[up][rp]psnr=stats_file=psnr.log:shortest=1[sp]"]
    maps = ["-map", "[sg]", "-map", "[sp]"]
    for i in range(n):
        r, c = divmod(i, grid)
        x, y = c * cw, r * ch
        parts.append(f"[u{i + 1}]crop={cw}:{ch}:{x}:{y}[uc{i}]")
        parts.append(f"[r{i + 1}]crop={cw}:{ch}:{x}:{y}[rc{i}]")
        parts.append(f"[uc{i}][rc{i}]ssim=stats_file=cell_{r}_{c}.log:shortest=1[sc{i}]")
        maps += ["-map", f"[sc{i}]"]
    os.makedirs(work_dir, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", os.path.abspath(up), "-i", os.path.abspath(src),
           "-filter_complex", ";".join(parts)] + maps + ["-f", "null", "-"]
    _run(cmd, cwd=work_dir)
    with open(os.path.join(work_dir, "ssim.log"), encoding="utf-8") as f:
        ssim = parse_ssim_log(f.read())
    with open(os.path.join(work_dir, "psnr.log"), encoding="utf-8") as f:
        psnr = parse_psnr_log(f.read())
    cells = {}
    for r in range(grid):
        for c in range(grid):
            with open(os.path.join(work_dir, f"cell_{r}_{c}.log"), encoding="utf-8") as f:
                cells[(r, c)] = [e["All"] for e in parse_ssim_log(f.read())]
    res = summarise(ssim, psnr, cells, grid, thresholds)
    res["size"] = [W, H]
    res["source"] = {"width": src_info["width"], "height": src_info["height"], "frames": src_info["frames"],
                     "fps": src_info["fps"]}
    res["upscaled"] = {"width": W, "height": H, "frames": up_info["frames"], "fps": up_info["fps"]}
    if up_info["frames"] != src_info["frames"]:
        res["warnings"].append(f"frame count differs: source {src_info['frames']}, upscaled {up_info['frames']}; "
                               f"compared the first {res['frames']}")
    log(f"fidelity: ssim mean {res['ssim']['mean']:.3f} min {res['ssim']['min']:.3f}, psnr mean "
        f"{res['psnr']['mean']:.1f} dB, verdict {res['verdict']}" +
        (f" ({len(res['flags']['cells_drift'])} drifting cells)" if res["flags"]["cells_drift"] else ""))
    return res


def summarise(ssim_frames, psnr_frames, cells, grid=GRID, thresholds=None):
    """Fold per-frame stats into the fidelity JSON. cells: {(row, col): [All per frame]}."""
    from videogen.graphs_h3 import FIDELITY_DEFAULTS
    t = dict(FIDELITY_DEFAULTS)
    t.update(thresholds or {})
    all_ssim = [e["All"] for e in ssim_frames]
    all_psnr = [e["psnr_avg"] for e in psnr_frames]
    frames = min(len(all_ssim), len(all_psnr)) if all_psnr else len(all_ssim)
    flags = {"frames_below_ssim": [i for i, v in enumerate(all_ssim) if v < t["ssim_min"]],
             "frames_below_psnr": [i for i, v in enumerate(all_psnr) if v < t["psnr_min"]],
             "cells_low": [], "cells_drift": []}
    cell_mean = [[None] * grid for _ in range(grid)]
    cell_min = [[None] * grid for _ in range(grid)]
    per_frame_cells = {}
    for (r, c), vals in cells.items():
        if not vals:
            continue
        cell_mean[r][c] = round(sum(vals) / len(vals), 4)
        cell_min[r][c] = round(min(vals), 4)
        for i, v in enumerate(vals):
            per_frame_cells.setdefault(i, []).append(((r, c), v))
    for i, entries in per_frame_cells.items():
        mean = sum(v for _, v in entries) / len(entries)
        for (r, c), v in entries:
            if v < t["cell_ssim_min"]:
                flags["cells_low"].append({"frame": i, "row": r, "col": c, "ssim": round(v, 4)})
            if mean - v > t["cell_drop_max"]:
                flags["cells_drift"].append({"frame": i, "row": r, "col": c, "ssim": round(v, 4),
                                             "frame_mean": round(mean, 4)})
    verdict = "ok"
    if flags["frames_below_ssim"] or flags["frames_below_psnr"] or flags["cells_drift"] or flags["cells_low"]:
        verdict = "review"
    return {"frames": frames,
            "ssim": {"mean": round(sum(all_ssim) / len(all_ssim), 4) if all_ssim else None,
                     "min": round(min(all_ssim), 4) if all_ssim else None,
                     "p05": round(_percentile(all_ssim, 0.05), 4) if all_ssim else None,
                     "per_frame": [round(v, 4) for v in all_ssim]},
            "psnr": {"mean": round(sum(all_psnr) / len(all_psnr), 2) if all_psnr else None,
                     "min": round(min(all_psnr), 2) if all_psnr else None,
                     "p05": round(_percentile(all_psnr, 0.05), 2) if all_psnr else None,
                     "per_frame": [round(v, 2) for v in all_psnr]},
            "grid": {"rows": grid, "cols": grid, "cell_mean": cell_mean, "cell_min": cell_min},
            "flags": flags, "thresholds": t, "verdict": verdict, "warnings": [],
            "note": ("SSIM/PSNR of the upscaled clip against a lanczos resize of the source; a latent refine "
                     "legitimately adds detail, so absolute values run below a resize-vs-resize baseline. "
                     "cells_drift (a cell far below its frame's mean) is the badge/plate/wheel signal. "
                     "Provisional thresholds; calibrate on the PC. Warns, never gates.")}


def compare_video(src, up, dest, log=print):
    """Side-by-side: lanczos-resized source | upscaled, source audio if any."""
    up_info = _info(up)
    src_info = _info(src)
    W, H = up_info["width"], up_info["height"]
    fc = f"[1:v]scale={W}:{H}:flags=lanczos[ref];[ref][0:v]hstack=inputs=2:shortest=1[out]"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", os.path.abspath(up), "-i", os.path.abspath(src),
           "-filter_complex", fc, "-map", "[out]"]
    if src_info["has_audio"]:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-shortest",
            "-movflags", "+faststart", os.path.abspath(dest)]
    _run(cmd, cwd=os.path.dirname(os.path.abspath(dest)) or None)
    log(f"compare video (source | upscaled) -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def write_json(res, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return path
