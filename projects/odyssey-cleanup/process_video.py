"""Remove the burned-in opening title and subtitle band with ProPainter.

The source asset is supplied by render-farm at runtime. ProPainter is cloned
from its official v0.1.0 release into an ignored cache directory, so model
weights and source are reused between the draft and final jobs.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
PROPAINTER_DIR = PROJECT_DIR / ".propainter-runtime-e870e793"
PROPAINTER_REPO = "https://github.com/sczhou/ProPainter.git"
PROPAINTER_REVISION = "e870e79321c31b733e2031af5aa2fb1fe3ac7eec"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("RUN " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_propainter() -> None:
    inference = PROPAINTER_DIR / "inference_propainter.py"
    if inference.exists():
        print("Using cached ProPainter checkout", flush=True)
        return
    if PROPAINTER_DIR.exists():
        shutil.rmtree(PROPAINTER_DIR)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            PROPAINTER_REPO,
            str(PROPAINTER_DIR),
        ]
    )
    head = subprocess.check_output(
        ["git", "-C", str(PROPAINTER_DIR), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != PROPAINTER_REVISION:
        run(["git", "-C", str(PROPAINTER_DIR), "fetch", "--depth", "1", "origin", PROPAINTER_REVISION])
        run(["git", "-C", str(PROPAINTER_DIR), "checkout", "--detach", PROPAINTER_REVISION])
    if not inference.exists():
        raise RuntimeError(f"ProPainter inference entry point missing after clone: {inference}")


def video_info(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise RuntimeError("Input video metadata is incomplete")
    return width, height, fps, frames


def make_masks(
    mask_dir: Path,
    video_path: Path,
    width: int,
    height: int,
    fps: float,
    frames: int,
    include_title: bool,
) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Coordinates were measured on the 576x1024 source. Text pixels are
    # selected inside these regions instead of masking the entire rectangle;
    # this avoids asking the model to reconstruct intact car/body details.
    sx = width / 576.0
    sy = height / 1024.0
    subtitle = (round(25 * sx), round(700 * sy), round(551 * sx), round(865 * sy))
    title = (round(35 * sx), round(315 * sy), round(575 * sx), round(450 * sy))
    title_frames = round(1.05 * fps)
    dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def detect_text(frame: np.ndarray, region: tuple[int, int, int, int], cyan: bool) -> np.ndarray:
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Cyan subtitle fill.
        cyan_seed = np.zeros(gray.shape, dtype=np.uint8)
        if cyan:
            cyan_seed[
                (hsv[:, :, 0] >= 72)
                & (hsv[:, :, 0] <= 102)
                & (hsv[:, :, 1] >= 75)
                & (hsv[:, :, 2] >= 125)
            ] = 255

        # White title/subtitle fill is distinguished from the bright showroom
        # by requiring a nearby dark outline/shadow.
        local_min = cv2.erode(gray, np.ones((9, 9), dtype=np.uint8))
        white_seed = np.zeros(gray.shape, dtype=np.uint8)
        white_seed[
            (hsv[:, :, 1] <= 80)
            & (hsv[:, :, 2] >= 175)
            & ((gray.astype(np.int16) - local_min.astype(np.int16)) >= 48)
        ] = 255

        letters = cv2.bitwise_or(cyan_seed, white_seed)
        letters = cv2.morphologyEx(
            letters,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )
        letters = cv2.dilate(letters, dilation, iterations=1)
        detected = np.zeros((height, width), dtype=np.uint8)
        detected[y1:y2, x1:x2] = letters
        return detected

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for mask generation: {video_path}")

    for index in range(frames):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read video frame {index} for masking")
        mask = detect_text(frame, subtitle, cyan=True)
        if include_title and index < title_frames:
            mask = cv2.bitwise_or(mask, detect_text(frame, title, cyan=False))
        if not cv2.imwrite(str(mask_dir / f"{index:05d}.png"), mask):
            cap.release()
            raise RuntimeError(f"Could not write mask frame {index}")
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draft-seconds", type=float, default=0.0)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    args = parser.parse_args()

    source = (PROJECT_DIR.parent.parent / args.input).resolve()
    output = (PROJECT_DIR.parent.parent / args.output).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    work_root = Path(os.environ.get("RENDER_WORK_DIR", PROJECT_DIR / ".work"))
    work_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_executable()
    working_input = source

    if args.draft_seconds > 0:
        working_input = work_root / "odyssey_draft_source.mp4"
        run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-i",
                str(source),
                "-t",
                str(args.draft_seconds),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "copy",
                str(working_input),
            ]
        )

    width, height, fps, frames = video_info(working_input)
    duration = frames / fps
    segment_count = max(1, math.ceil(duration / args.chunk_seconds))
    print(
        f"Input: {width}x{height}, {fps:.3f} fps, {frames} frames; "
        f"processing {segment_count} temporal segments",
        flush=True,
    )
    ensure_propainter()
    print("PROGRESS 8", flush=True)

    inpainted_segments: list[Path] = []
    for segment_index in range(segment_count):
        start = duration * segment_index / segment_count
        end = duration * (segment_index + 1) / segment_count
        segment_input = work_root / f"segment_{segment_index:02d}.mp4"
        run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(working_input),
                "-t",
                f"{end - start:.6f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "15",
                "-pix_fmt",
                "yuv420p",
                str(segment_input),
            ]
        )
        seg_width, seg_height, seg_fps, seg_frames = video_info(segment_input)
        mask_dir = work_root / f"masks_{segment_index:02d}"
        make_masks(
            mask_dir,
            segment_input,
            seg_width,
            seg_height,
            seg_fps,
            seg_frames,
            include_title=segment_index == 0,
        )
        result_root = work_root / f"propainter-results-{segment_index:02d}"
        run(
            [
                sys.executable,
                "-u",
                str(PROPAINTER_DIR / "inference_propainter.py"),
                "--video",
                str(segment_input),
                "--mask",
                str(mask_dir),
                "--output",
                str(result_root),
                "--height",
                str(seg_height),
                "--width",
                str(seg_width),
                "--save_fps",
                str(round(seg_fps)),
                "--mask_dilation",
                "2",
                "--subvideo_length",
                "50",
                "--neighbor_length",
                "10",
                "--ref_stride",
                "10",
                "--raft_iter",
                "20",
                "--fp16",
            ],
            cwd=PROPAINTER_DIR,
        )
        inpainted = result_root / segment_input.stem / "inpaint_out.mp4"
        if not inpainted.exists():
            raise FileNotFoundError(f"ProPainter output missing: {inpainted}")
        inpainted_segments.append(inpainted)
        print(
            f"PROGRESS {8 + round((segment_index + 1) / segment_count * 80)}",
            flush=True,
        )

    concat_list = work_root / "inpainted-segments.ffconcat"
    concat_lines = ["ffconcat version 1.0"]
    for path in inpainted_segments:
        escaped = path.as_posix().replace("'", "'\\''")
        concat_lines.append(f"file '{escaped}'")
    concat_list.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    inpainted = work_root / "inpainted-full.mp4"
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(inpainted),
        ]
    )
    print("PROGRESS 90", flush=True)

    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(inpainted),
            "-i",
            str(working_input),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-vf",
            "scale=1080:1920:flags=lanczos,unsharp=5:5:0.35:5:5:0.0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    print("PROGRESS 100", flush=True)


if __name__ == "__main__":
    main()
