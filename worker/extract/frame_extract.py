"""frame_extract: pull stills out of a static-lock pan at given timestamps
(render-asset-gate-spec.md §7.6).

  python frame_extract.py <video> <out_dir> --frames '[{"at_s": 0.8, "slot": "position_lock"}, ...]'

Writes <out_dir>/<slot>.png per frame and <out_dir>/index.json listing them.
Timestamps past the end are clamped to the last half second, so a short clip
still yields every requested slot.
"""
import argparse
import json
import os

from common import find_ffmpeg_tool, probe_duration, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--frames", required=True, help="JSON list of {at_s, slot}")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg_tool("ffmpeg")
    ffprobe = find_ffmpeg_tool("ffprobe")
    os.makedirs(args.out_dir, exist_ok=True)
    frames = json.loads(args.frames)
    duration = probe_duration(ffprobe, args.video)

    index = []
    for i, fr in enumerate(frames):
        at = float(fr["at_s"])
        if duration > 0:
            at = min(at, max(0.0, duration - 0.5))
        slot = str(fr["slot"])
        out = os.path.join(args.out_dir, f"{slot}.png")
        run([ffmpeg, "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", args.video,
             "-frames:v", "1", "-q:v", "2", out])
        if not os.path.exists(out):
            raise RuntimeError(f"no frame at {at}s for {slot}")
        index.append({"slot": slot, "at_s": round(at, 3), "path": out})
        print(f"frame {slot} @ {at:.2f}s")
        print(f"PROGRESS {int(10 + 80 * (i + 1) / max(1, len(frames)))}")

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": duration, "frames": index}, f, indent=1)


if __name__ == "__main__":
    main()
