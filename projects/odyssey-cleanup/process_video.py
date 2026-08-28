"""Remove the burned-in opening title and subtitle band with ProPainter.

The source asset is supplied by render-farm at runtime. ProPainter is cloned
from its official v0.1.0 release into an ignored cache directory, so model
weights and source are reused between the draft and final jobs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
PROPAINTER_DIR = PROJECT_DIR / ".propainter-runtime"
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


def make_masks(mask_dir: Path, width: int, height: int, fps: float, frames: int) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Coordinates were measured on the 576x1024 source. Scale them if the
    # input dimensions ever differ. The small padding covers text outlines.
    sx = width / 576.0
    sy = height / 1024.0
    subtitle = (
        round(48 * sx),
        round(738 * sy),
        round(528 * sx),
        round(816 * sy),
    )
    title = (
        round(58 * sx),
        round(338 * sy),
        round(566 * sx),
        round(425 * sy),
    )
    title_frames = round(1.05 * fps)

    for index in range(frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = subtitle
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        if index < title_frames:
            x1, y1, x2, y2 = title
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        if not cv2.imwrite(str(mask_dir / f"{index:05d}.png"), mask):
            raise RuntimeError(f"Could not write mask frame {index}")
        if index and index % max(1, frames // 10) == 0:
            print(f"PROGRESS {min(10, round(index / frames * 10))}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draft-seconds", type=float, default=0.0)
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
    print(f"Input: {width}x{height}, {fps:.3f} fps, {frames} frames", flush=True)
    mask_dir = work_root / "masks"
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    make_masks(mask_dir, width, height, fps, frames)
    ensure_propainter()
    print("PROGRESS 12", flush=True)

    result_root = work_root / "propainter-results"
    run(
        [
            sys.executable,
            "-u",
            str(PROPAINTER_DIR / "inference_propainter.py"),
            "--video",
            str(working_input),
            "--mask",
            str(mask_dir),
            "--output",
            str(result_root),
            "--height",
            str(height),
            "--width",
            str(width),
            "--save_fps",
            str(round(fps)),
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
    print("PROGRESS 88", flush=True)

    inpainted = result_root / working_input.stem / "inpaint_out.mp4"
    if not inpainted.exists():
        raise FileNotFoundError(f"ProPainter output missing: {inpainted}")

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
