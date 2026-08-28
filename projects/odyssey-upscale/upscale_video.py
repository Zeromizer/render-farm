"""Upscale a portrait video to UHD 4K with Real-ESRGAN NCNN/Vulkan."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
REALESRGAN_RELEASE = "20220424"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)


def run(command: list[str]) -> None:
    print("RUN " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def video_info(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Input video metadata is incomplete")
    return fps, frames, width, height


def ensure_realesrgan(cache_root: Path) -> Path:
    runtime = cache_root / f"realesrgan-ncnn-vulkan-{REALESRGAN_RELEASE}"
    executable = runtime / "realesrgan-ncnn-vulkan.exe"
    if executable.exists():
        print(f"Using cached Real-ESRGAN runtime: {runtime}", flush=True)
        return executable

    archive = cache_root / f"realesrgan-ncnn-vulkan-{REALESRGAN_RELEASE}.zip"
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official Real-ESRGAN release from {REALESRGAN_URL}", flush=True)
    urllib.request.urlretrieve(REALESRGAN_URL, archive)
    staging = cache_root / "realesrgan-extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(staging)
    candidates = list(staging.rglob("realesrgan-ncnn-vulkan.exe"))
    if len(candidates) != 1:
        raise RuntimeError("Real-ESRGAN executable was not found in the release archive")
    extracted_root = candidates[0].parent
    if runtime.exists():
        shutil.rmtree(runtime)
    shutil.move(str(extracted_root), str(runtime))
    if staging.exists():
        shutil.rmtree(staging)
    if not executable.exists():
        raise FileNotFoundError(executable)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=2160)
    parser.add_argument("--height", type=int, default=3840)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--draft-seconds", type=float, default=0.0)
    args = parser.parse_args()

    source = (REPO_ROOT / args.input).resolve()
    output = (REPO_ROOT / args.output).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    work_root = Path(__import__("os").environ.get("RENDER_WORK_DIR", PROJECT_DIR / ".work"))
    runtime_cache = work_root.parent / "realesrgan-runtime-cache"
    frames_dir = work_root / "frames"
    upscaled_dir = work_root / "upscaled"
    for directory in (frames_dir, upscaled_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    working_input = source
    if args.draft_seconds > 0:
        working_input = work_root / "draft-source.mp4"
        run([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source),
            "-t", str(args.draft_seconds), "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(working_input),
        ])

    fps, frame_count, width, height = video_info(working_input)
    print(
        f"Input: {width}x{height}, {fps:.6f} fps, {frame_count} frames; "
        f"target: {args.width}x{args.height}",
        flush=True,
    )
    print("PROGRESS 3", flush=True)

    run([
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(working_input),
        "-map", "0:v:0", "-vsync", "0", str(frames_dir / "%08d.png"),
    ])
    extracted_count = len(list(frames_dir.glob("*.png")))
    if extracted_count == 0:
        raise RuntimeError("No frames were extracted")
    print(f"Extracted {extracted_count} frames", flush=True)
    print("PROGRESS 10", flush=True)

    realesrgan = ensure_realesrgan(runtime_cache)
    run([
        str(realesrgan), "-i", str(frames_dir), "-o", str(upscaled_dir),
        "-n", "realesrgan-x4plus", "-s", "4", "-t", str(args.tile),
        "-f", "png", "-j", "2:2:2",
    ])
    upscaled_count = len(list(upscaled_dir.glob("*.png")))
    if upscaled_count != extracted_count:
        raise RuntimeError(
            f"Upscale produced {upscaled_count} frames from {extracted_count} inputs"
        )
    print("PROGRESS 92", flush=True)

    # Preserve the entire image. For non-9:16 sources, letterbox instead of cropping.
    scale_filter = (
        f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease:"
        "flags=lanczos,"
        f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    run([
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-framerate", f"{fps:.12g}", "-i", str(upscaled_dir / "%08d.png"),
        "-i", str(working_input), "-map", "0:v:0", "-map", "1:a:0?",
        "-vf", scale_filter, "-c:v", "libx265", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "256k",
        "-shortest", "-movflags", "+faststart", str(output),
    ])
    print("PROGRESS 100", flush=True)


if __name__ == "__main__":
    main()
