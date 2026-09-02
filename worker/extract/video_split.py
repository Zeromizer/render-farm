"""video_split: cut a Seedance take into shots (render-asset-gate-spec.md §10).

  python video_split.py <video> <out_dir> [--shot-count N] [--boundaries 1.2,3.4] [--threshold 27]

Declared boundaries win; otherwise PySceneDetect's ContentDetector with
min_scene_len=4 (the reference-extraction lesson: the default 15 merges the
fast cuts these ads live on). Each shot is re-encoded rather than
stream-copied — a cut never lands on a keyframe — and gets a 480px poster a
little way in, not at frame zero. Writes shot-N.mp4, shot-N.jpg, index.json.
"""
import argparse
import json
import os

from common import find_ffmpeg_tool, probe_duration, run

MIN_SHOT_S = 0.4


def detect(video, threshold):
    from scenedetect import ContentDetector, detect as sd_detect

    scenes = sd_detect(video, ContentDetector(threshold=threshold, min_scene_len=4))
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--shot-count", type=int, default=None)
    ap.add_argument("--boundaries", default=None, help="comma-separated cut times in seconds")
    ap.add_argument("--threshold", type=float, default=27.0)
    args = ap.parse_args()

    ffmpeg = find_ffmpeg_tool("ffmpeg")
    ffprobe = find_ffmpeg_tool("ffprobe")
    os.makedirs(args.out_dir, exist_ok=True)
    duration = probe_duration(ffprobe, args.video)
    if duration <= 0:
        raise RuntimeError("could not read the clip's duration")

    if args.boundaries:
        cuts = sorted({float(x) for x in args.boundaries.split(",") if x.strip()})
        cuts = [c for c in cuts if 0 < c < duration]
        edges = [0.0] + cuts + [duration]
        bounds = list(zip(edges[:-1], edges[1:]))
        source = "declared"
    else:
        bounds = detect(args.video, args.threshold) or [(0.0, duration)]
        source = "detected"
    bounds = [(s, e) for s, e in bounds if e - s >= MIN_SHOT_S] or [(0.0, duration)]
    print(f"shots {len(bounds)} ({source}); declared shot_count={args.shot_count}")
    print("PROGRESS 20")

    index = []
    for i, (s, e) in enumerate(bounds, start=1):
        clip = os.path.join(args.out_dir, f"shot-{i}.mp4")
        poster = os.path.join(args.out_dir, f"shot-{i}.jpg")
        run([ffmpeg, "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", args.video,
             "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-movflags", "+faststart", clip])
        at = s + min(1.0, (e - s) / 2)
        run([ffmpeg, "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", args.video, "-frames:v", "1",
             "-vf", "scale=480:-2", "-q:v", "4", poster])
        index.append({"index": i, "start_s": round(s, 3), "end_s": round(e, 3),
                      "clip": clip, "poster": poster})
        print(f"shot {i}: {s:.2f}-{e:.2f}s")
        print(f"PROGRESS {int(20 + 75 * i / len(bounds))}")

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": duration, "source": source, "shots": index}, f, indent=1)


if __name__ == "__main__":
    main()
