"""Pure compute for the `matte` engine: a clip in, an alpha matte out.

Same split as extract/extract.py - this file knows nothing about Supabase,
rows or buckets. It takes a local video and a set of parameters, writes one
local output file, and prints PROGRESS/PHASE lines that runners/matte.py turns
into farm_render_jobs.progress and .phase.

usage:
  python matte.py <video> <out_path> --model <name> --output <kind> [...]

Three things here are load-bearing and easy to get wrong:

1. THE PROVIDER ASSERTION. onnxruntime silently falls back to CPU when the
   CUDA libraries are missing. On CPU this model costs ~9.5 s/frame, and the
   worker runs one job at a time, so a silent fallback does not merely run
   slow - it holds the whole render queue. We check before frame one.

2. THE DLL DIRECTORIES. On Windows the CUDA/cuDNN DLLs live inside the
   nvidia-* wheels in site-packages, which is NOT on the DLL search path.
   os.add_dll_directory has to be called before onnxruntime is imported or the
   CUDA provider fails to load and we take the silent-CPU path we just spent
   effort ruling out.

3. TEMPORAL MEDIAN IS OFF BY DEFAULT HERE. A per-pixel median across N frames
   assumes the edge barely moves between them. On the footage that motivated
   this engine the subject's hand travels ~8-9 px/frame, so a 3-frame window
   medians "subject, subject, background" along the whole leading edge and
   erodes it - the same class of artefact that ruled out sparse keying. It is
   right for a static subject and wrong for a moving one, so the caller must
   ask for it explicitly.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

# One resolver for ffmpeg, not a fourth copy of it: extract/common.py already
# owns the FFMPEG_DIR -> PATH -> winget search that this machine needs (the
# winget install is not on the worker's inherited PATH).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract"))
from common import find_ffmpeg_tool  # noqa: E402

MODELS = {
    "birefnet-general", "birefnet-general-lite", "birefnet-portrait",
    "bria-rmbg", "u2net_human_seg", "isnet-general-use",
}
OUTPUTS = {"webm_alpha", "mask_mp4", "png_sequence"}


def emit(kind, text):
    """PROGRESS <0-100> and PHASE <text> are parsed by runners/matte.py."""
    print(f"{kind} {text}", flush=True)


def require_cuda():
    """Make CUDA loadable and confirm it, or fail loudly. Returns providers.

    preload_dlls() is the load-bearing call and it is NOT optional on Windows.
    The CUDA and cuDNN DLLs live inside the nvidia-* wheels in site-packages,
    which is not on the DLL search path, and onnxruntime 1.29 does not find
    them by itself. Putting those directories on the path with
    os.add_dll_directory is NOT enough either - measured: with both
    nvidia/cu13/bin/x86_64 and nvidia/cudnn/bin added, InferenceSession still
    came back CPU-only and onnxruntime logged nothing about why, even at
    log_severity_level=0. preload_dlls() loads them the way the runtime
    expects and the same session then takes CUDAExecutionProvider.

    Note what get_available_providers() is worth here: it lists what the build
    was COMPILED with, not what can load. It reported CUDAExecutionProvider on
    a process where CUDA could not initialise at all. That is why the real
    assertion is on the session in segment(), not on this list.
    """
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "matte requires the GPU: CUDAExecutionProvider is not available. "
            f"onnxruntime reports {providers}, version {ort.__version__}. "
            "Refusing to run on CPU - at ~9.5 s/frame this would block every "
            "render behind it on a single-job worker."
        )
    return providers


def probe(video, ffprobe):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", video],
        capture_output=True, text=True, check=True,
    ).stdout
    s = (json.loads(out).get("streams") or [{}])[0]
    num, _, den = (s.get("r_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1)
    return int(s.get("width") or 0), int(s.get("height") or 0), fps


def extract_frames(video, out_dir, ffmpeg, start_s, end_s, fps, scale, src_w, src_h):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start_s is not None:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", video]
    if end_s is not None:
        cmd += ["-to", str(end_s if start_s is None else end_s - start_s)]
    vf = []
    if fps is not None:
        vf.append(f"fps={fps}")
    if scale:
        # Long edge to `scale`, keeping aspect; -2 keeps dimensions even, which
        # yuva420p requires downstream.
        vf.append(f"scale={scale}:-2" if src_w >= src_h else f"scale=-2:{scale}")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-start_number", "0", os.path.join(out_dir, "src-%06d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(glob.glob(os.path.join(out_dir, "src-*.png")))


def segment(frames, alpha_dir, model, providers):
    """Per-frame alpha from rembg. Returns the list of alpha png paths."""
    import numpy as np
    from PIL import Image
    from rembg import new_session, remove

    os.makedirs(alpha_dir, exist_ok=True)
    session = new_session(model, providers=["CUDAExecutionProvider"])
    # rembg does not surface the session's providers, so confirm the real one
    # rather than trusting the request we just made.
    actual = getattr(getattr(session, "inner_session", None), "get_providers", list)()
    if actual and "CUDAExecutionProvider" not in actual:
        raise RuntimeError(f"rembg session did not take the GPU: providers={actual}")
    emit("PHASE", f"segmenting 0/{len(frames)}")

    out = []
    for i, src in enumerate(frames):
        with Image.open(src) as im:
            cut = remove(im.convert("RGB"), session=session)
        alpha = np.array(cut.convert("RGBA"))[:, :, 3]
        dst = os.path.join(alpha_dir, f"a-{i:06d}.png")
        Image.fromarray(alpha, mode="L").save(dst)
        out.append(dst)
        if i % 5 == 0 or i == len(frames) - 1:
            emit("PROGRESS", str(int((i + 1) / len(frames) * 100)))
            emit("PHASE", f"segmenting {i + 1}/{len(frames)}")
    return out


def refine(alpha_paths, frames, rgba_dir, mask_dir, close_px, feather_px, temporal_median):
    """Morphology, feather and (optional) temporal median; writes RGBA + mask."""
    import cv2
    import numpy as np
    from PIL import Image

    os.makedirs(rgba_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    half = temporal_median // 2 if temporal_median and temporal_median > 1 else 0

    def load(i):
        return cv2.imread(alpha_paths[i], cv2.IMREAD_GRAYSCALE)

    for i in range(len(alpha_paths)):
        a = load(i)
        if half:
            lo, hi = max(0, i - half), min(len(alpha_paths) - 1, i + half)
            a = np.median(np.stack([load(j) for j in range(lo, hi + 1)]), axis=0).astype(np.uint8)
        if close_px and close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1,) * 2)
            a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k)
        if feather_px and feather_px > 0:
            k = feather_px * 2 + 1
            a = cv2.GaussianBlur(a, (k, k), 0)
        cv2.imwrite(os.path.join(mask_dir, f"m-{i:06d}.png"), a)
        rgb = cv2.cvtColor(cv2.imread(frames[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        Image.fromarray(np.dstack([rgb, a])).save(os.path.join(rgba_dir, f"r-{i:06d}.png"))
        if i % 10 == 0:
            emit("PHASE", f"refining {i + 1}/{len(alpha_paths)}")


def encode(kind, rgba_dir, mask_dir, out_path, fps, ffmpeg):
    if kind == "png_sequence":
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(glob.glob(os.path.join(rgba_dir, "r-*.png"))):
                z.write(p, os.path.basename(p))
        return
    if kind == "webm_alpha":
        # yuva420p is the alpha-carrying pixel format; -auto-alt-ref 0 is
        # mandatory, VP9's alt-ref frames drop the alpha plane otherwise.
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-framerate", str(fps), "-i", os.path.join(rgba_dir, "r-%06d.png"),
               "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0",
               "-crf", "30", "-auto-alt-ref", "0", out_path]
    else:  # mask_mp4 - alpha in luma, greyscale
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-framerate", str(fps), "-i", os.path.join(mask_dir, "m-%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_path")
    ap.add_argument("--model", default="birefnet-general-lite")
    ap.add_argument("--output", default="webm_alpha")
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--end-s", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--feather-px", type=int, default=1)
    ap.add_argument("--close-px", type=int, default=3)
    ap.add_argument("--temporal-median", type=int, default=0)
    a = ap.parse_args()

    if a.model not in MODELS:
        raise SystemExit(f"unknown model {a.model!r}; wired models are {sorted(MODELS)}")
    if a.output not in OUTPUTS:
        raise SystemExit(f"unknown output {a.output!r}; supported are {sorted(OUTPUTS)}")

    providers = require_cuda()
    emit("PHASE", "gpu ready")
    print(f"[matte] providers={providers}", flush=True)

    ffmpeg = find_ffmpeg_tool("ffmpeg")
    ffprobe = find_ffmpeg_tool("ffprobe")
    src_w, src_h, src_fps = probe(a.video, ffprobe)
    out_fps = a.fps or src_fps
    print(f"[matte] source {src_w}x{src_h} @ {src_fps:.3f} fps -> out {out_fps:.3f} fps", flush=True)
    if a.temporal_median and a.temporal_median > 1:
        print(f"[matte] WARNING temporal_median={a.temporal_median}: a per-pixel median "
              "erodes edges that move more than a pixel or two between frames", flush=True)

    work = os.path.dirname(os.path.abspath(a.out_path))
    frames_dir = os.path.join(work, "_frames")
    emit("PHASE", "extracting frames")
    frames = extract_frames(a.video, frames_dir, ffmpeg, a.start_s, a.end_s,
                            a.fps, a.scale, src_w, src_h)
    if not frames:
        raise SystemExit("no frames extracted - check start_s/end_s against the clip length")
    print(f"[matte] {len(frames)} frames", flush=True)

    import time
    t0 = time.time()
    alphas = segment(frames, os.path.join(work, "_alpha"), a.model, providers)
    per_frame = (time.time() - t0) / len(frames)
    # The number the platform asked for, on its own line so it is greppable.
    print(f"[matte] MEASURED {per_frame:.3f} s/frame for {a.model} at "
          f"{src_w}x{src_h} over {len(frames)} frames", flush=True)

    emit("PHASE", "refining")
    refine(alphas, frames, os.path.join(work, "_rgba"), os.path.join(work, "_mask"),
           a.close_px, a.feather_px, a.temporal_median)

    emit("PHASE", "encoding")
    encode(a.output, os.path.join(work, "_rgba"), os.path.join(work, "_mask"),
           a.out_path, out_fps, ffmpeg)
    for d in ("_frames", "_alpha", "_rgba", "_mask"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    emit("PROGRESS", "100")
    print(f"[matte] WROTE {a.out_path}", flush=True)


if __name__ == "__main__":
    main()
