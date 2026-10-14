"""Pure compute for the `matte` engine: a clip in, an alpha matte out.

Same split as extract/extract.py - this file knows nothing about Supabase,
rows or buckets. It takes a local video and a set of parameters, writes one
local output file, and prints PROGRESS/PHASE lines that runners/matte.py turns
into farm_render_jobs.progress and .phase.

usage:
  python matte.py <video|image> <out_path> --model <name> --output <kind> [...]

  --output png takes ONE still image and writes one RGBA png (sticker cut-outs).

Four things here are load-bearing and easy to get wrong:

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

4. THE SESSION. CUDA with a capped arena, never the default arena; TensorRT
   fp32 instead only when opted in (MATTE_TRT=1) AND a pre-built engine for
   the model is in the cache. Measured on the 4080 (birefnet-portrait,
   1080x1920, 2026-09-27): the default kNextPowerOfTwo arena grows to
   ~15.5 GB, past what the 16 GB card has free, and session.run falls from
   0.27 s to 1.0-1.8 s/frame. The 10 GB cap keeps it at 0.27 s (8 GB OOMs in
   the aspp_deforms decoder) with masks bit-identical to the old session.
   TensorRT fp32 runs it in 0.12 s at ~4.4 GB but is not bit-identical (see
   TRT_ENABLED), which is why it is opt-in. An engine takes ~5 min to build,
   so a job NEVER builds one: warmup_trt.py does, at deploy, and a job without
   a ready engine uses capped CUDA. Nothing about TensorRT may fail a job.
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
OUTPUTS = {"webm_alpha", "mask_mp4", "png_sequence", "png"}

# Session config (see 4. above). TensorRT is OFF unless MATTE_TRT=1: capped CUDA
# gives masks bit-identical to the old default-arena session, TensorRT fp32 does
# not quite (mean |d| 0.004/255 raw, but after refine 9 of 150 frames of a person
# clip moved a ~1,600 px blob in an ambiguous dark gap; disabling TF32 made it
# worse, not better - it is kernel-level float noise, not a setting). Shawn chose
# identical masks (2026-09-27). The engines and warmup_trt.py stay so it can be
# switched on with MATTE_TRT=1 in the worker .env.
GPU_MEM_LIMIT_GB = float(os.environ.get("MATTE_GPU_MEM_LIMIT_GB", "10"))
TRT_ENABLED = os.environ.get("MATTE_TRT", "0") == "1"
TRT_WORKSPACE_GB = 6
# Under the render-farm cache, NOT the job work dir (deleted after every job)
# and NOT the venv (rebuilt whenever requirements.txt changes). No cache sweep
# touches it: those are repos/, work/, assets/ and venvs/ only.
TRT_ROOT = os.path.join(
    os.environ.get("RENDER_CACHE_DIR")
    or os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "render-farm"),
    "trt")
_trt_libs_error = "not checked"


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

    The TensorRT libraries (the tensorrt_cu13_libs wheel) go on the search path
    first, the same way and for the same reason; if that fails, the job carries
    on and make_session() uses capped CUDA.
    """
    global _trt_libs_error
    _trt_libs_error = _add_trt_dlls() if TRT_ENABLED else "off (MATTE_TRT=1 enables)"
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


def _add_trt_dlls():
    """nvinfer_10.dll etc. onto the DLL path, before onnxruntime is imported.
    Returns None when done, else why not (the job then runs capped CUDA)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("tensorrt_libs")
        if spec is None or not spec.origin:
            return "tensorrt_libs is not installed"
        libs = os.path.dirname(spec.origin)
        os.add_dll_directory(libs)
        os.environ["PATH"] = libs + os.pathsep + os.environ.get("PATH", "")
        return None
    except Exception as e:  # noqa: BLE001 - TensorRT must never fail a job
        return f"{type(e).__name__}: {e}"[:200]


def cuda_options():
    return {"cudnn_conv_algo_search": "HEURISTIC",
            "gpu_mem_limit": int(GPU_MEM_LIMIT_GB * 1024 ** 3),
            "arena_extend_strategy": "kSameAsRequested"}


def trt_options(model):
    # fp32 on purpose: fp16 is 2.4x faster again but moves 0.17% of mask pixels
    # (edges, dark gaps); fp32 matches the CUDA masks.
    return {"trt_fp16_enable": False,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": os.path.join(TRT_ROOT, model),
            "trt_max_workspace_size": TRT_WORKSPACE_GB * 1024 ** 3}


def trt_fingerprint(model):
    """What a cached engine was built against. An engine that does not match is
    rebuilt by onnxruntime on session creation - minutes inside a user's job -
    so a job only offers TensorRT when warmup_trt.py recorded this exact set."""
    from importlib import metadata

    import onnxruntime as ort

    # rembg's own model location (BaseSession.u2net_home)
    home = os.path.expanduser(
        os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")))
    st = os.stat(os.path.join(home, f"{model}.onnx"))
    return {"model": model, "model_bytes": st.st_size, "model_mtime": int(st.st_mtime),
            "onnxruntime": ort.__version__, "tensorrt": metadata.version("tensorrt_cu13_libs"),
            "precision": "fp32", "workspace_gb": TRT_WORKSPACE_GB}


def trt_ready(model):
    """(True, None) when a warmed-up engine for this model is in the cache."""
    d = os.path.join(TRT_ROOT, model)
    try:
        with open(os.path.join(d, "ready.json"), encoding="utf-8") as f:
            ready = json.load(f)
        if not glob.glob(os.path.join(d, "*.engine")):
            return False, "no engine file"
        if ready.get("fingerprint") != trt_fingerprint(model):
            return False, "engine built for a different model/runtime; rerun warmup_trt.py"
        return True, None
    except FileNotFoundError:
        return False, "no warmed-up engine (warmup_trt.py)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:200]


def make_session(model, build=False):
    """(session, ep, why_not_tensorrt). ep is "tensorrt" or "cuda".

    TensorRT only when an engine is ready (or build=True, which only
    warmup_trt.py passes); anything else about it failing falls through to the
    capped CUDA session. CPU is still refused, as before."""
    import gc

    from rembg import new_session

    why = _trt_libs_error
    if why is None and not build:
        _, why = trt_ready(model)
    if why is None:
        try:
            s = new_session(model, providers=[("TensorrtExecutionProvider", trt_options(model)),
                                              ("CUDAExecutionProvider", cuda_options())])
            actual = s.inner_session.get_providers()
            if actual and actual[0] == "TensorrtExecutionProvider":
                return s, "tensorrt", None
            # onnxruntime drops a provider it cannot create and falls back on its
            # own, WITHOUT our CUDA options: rebuild the capped one below.
            why = f"tensorrt did not take: providers={actual}"
            del s
            gc.collect()
        except Exception as e:  # noqa: BLE001 - TensorRT must never fail a job
            why = f"tensorrt session failed: {type(e).__name__}: {e}"[:300]
    s = new_session(model, providers=[("CUDAExecutionProvider", cuda_options())])
    # rembg does not surface the session's providers, so confirm the real one
    # rather than trusting the request we just made.
    actual = getattr(getattr(s, "inner_session", None), "get_providers", list)()
    if actual and "CUDAExecutionProvider" not in actual:
        raise RuntimeError(f"rembg session did not take the GPU: providers={actual}")
    return s, "cuda", why


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
    """Per-frame alpha from rembg. Returns (alpha png paths, ep that ran)."""
    import numpy as np
    from PIL import Image
    from rembg import remove

    os.makedirs(alpha_dir, exist_ok=True)
    session, ep, why = make_session(model)
    print(f"[matte] ep={ep}" + (f" (tensorrt skipped: {why})" if why else ""), flush=True)
    emit("PHASE", f"segmenting 0/{len(frames)} ({ep})")

    out = []
    for i, src in enumerate(frames):
        with Image.open(src) as im:
            cut = remove(im.convert("RGB"), session=session)
        alpha = np.array(cut.convert("RGBA"))[:, :, 3]
        dst = os.path.join(alpha_dir, f"a-{i:06d}.png")
        # No mode= here: a 2-D uint8 array already infers "L", and the mode
        # parameter is removed in Pillow 13 (2026-10-15). Passing it would make
        # every matte job die at this line the day the pin in requirements.txt
        # moves, with a traceback naming Pillow rather than the pin.
        Image.fromarray(alpha).save(dst)
        out.append(dst)
        if i % 5 == 0 or i == len(frames) - 1:
            emit("PROGRESS", str(int((i + 1) / len(frames) * 100)))
            emit("PHASE", f"segmenting {i + 1}/{len(frames)} ({ep})")
    return out, ep


def anchor_edge(alpha_paths, sample=12):
    """Which frame edge does the subject enter from? Most alpha on it wins.

    Decided ONCE over a sample of the clip, not per frame. A per-frame answer
    flips on any frame where the subject is briefly clear of its edge, which is
    precisely the frame drop_detached's fallback exists to survive - so letting
    it also move the anchor would compound the problem instead of containing it.
    """
    import cv2
    import numpy as np

    n = min(sample, len(alpha_paths))
    idx = np.unique(np.linspace(0, len(alpha_paths) - 1, n).astype(int))
    totals = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    for i in idx:
        a = cv2.imread(alpha_paths[int(i)], cv2.IMREAD_GRAYSCALE)
        totals["top"] += float(a[0, :].mean())
        totals["bottom"] += float(a[-1, :].mean())
        totals["left"] += float(a[:, 0].mean())
        totals["right"] += float(a[:, -1].mean())
    return max(totals, key=totals.get), totals


def drop_detached(a, anchor, min_area_frac=0.0004):
    """Keep only the alpha blobs anchored to the edge the subject enters from.

    birefnet-portrait keeps the person and drops the cards, but anything else
    SALIENT and detached survives - on the reference clip the food photos
    sitting on the cards did, in half the sampled frames. No model setting
    removes those, because the property that separates them from the subject is
    geometric rather than semantic: an arm runs off frame at the edge it came
    in from, and a photo of fries touches nothing.

    THE FALLBACK IS LOAD-BEARING. If the subject is entirely inside the frame -
    a hand lifted clear of the bottom - then no component touches the anchor and
    the plain rule blanks that frame outright. That is a worse failure than the
    one being fixed and a silent one, so we keep the largest component instead
    and let the proof sheet show what happened.
    """
    import cv2
    import numpy as np

    n, labels, stats, _ = cv2.connectedComponentsWithStats((a > 127).astype(np.uint8), 8)
    if n <= 1:  # nothing but background; leave it alone
        return a
    h, w = a.shape
    min_area = max(1, int(min_area_frac * h * w))
    keep = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue  # specks along the anchor edge are not the subject either
        if {"top": y == 0, "bottom": y + ch >= h,
            "left": x == 0, "right": x + cw >= w}[anchor]:
            keep.append(i)
    if not keep:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    return np.where(np.isin(labels, keep), a, 0).astype(np.uint8)


def refine(alpha_paths, frames, rgba_dir, mask_dir, close_px, feather_px,
           temporal_median, anchor=None):
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
        # After CLOSE and before feather, deliberately: close first so a
        # fingertip bridged by a one-pixel gap is one component and does not get
        # dropped, feather after so the blur is not asked to soften an edge that
        # is about to be deleted.
        if anchor:
            a = drop_detached(a, anchor)
        if feather_px and feather_px > 0:
            k = feather_px * 2 + 1
            a = cv2.GaussianBlur(a, (k, k), 0)
        cv2.imwrite(os.path.join(mask_dir, f"m-{i:06d}.png"), a)
        rgb = cv2.cvtColor(cv2.imread(frames[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        Image.fromarray(np.dstack([rgb, a])).save(os.path.join(rgba_dir, f"r-{i:06d}.png"))
        if i % 10 == 0:
            emit("PHASE", f"refining {i + 1}/{len(alpha_paths)}")


def proof_sheet(rgba_dir, out_png, start_s, fps, model, anchor, tiles=8):
    """One picture that answers every question anyone asks about a matte.

    WHY THIS IS HERE AND NOT DOWNSTREAM. All three ways to check alpha after
    the fact fail SILENTLY, returning a plausible answer instead of an error:

      ffprobe                 reports pix_fmt=yuv420p on a correct alpha webm
      ffmpeg default decoder  returns a fully opaque frame unless you put
                              -c:v libvpx-vp9 BEFORE -i (measured on a real
                              output: 0.0% transparent vs 85.8% with the flag)
      seek + canvas drawImage hands back frame 1 every time, so a moving matte
                              reads as frozen

    An agent that trips any one of them reports a bug that does not exist and
    then iterates on it. Here, before encoding, none of them exist - the alpha
    is an array in memory. So the proof is composed at the only point where it
    cannot be got wrong, and everyone downstream just looks at the picture.

    Magenta ground, not black or a checkerboard: #ff00ff is the one colour a
    real subject never is, so a hole reads unambiguously as a hole. Over black,
    a hole and a correctly matted dark subject look the same.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    paths = sorted(glob.glob(os.path.join(rgba_dir, "r-*.png")))
    if not paths:
        return None
    idx = np.unique(np.linspace(0, len(paths) - 1, min(tiles, len(paths))).astype(int))

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1 takes no size here
        font = ImageFont.load_default()

    cells = []
    for i in idx:
        with Image.open(paths[int(i)]) as im:
            im = im.convert("RGBA")
            a = np.asarray(im)[:, :, 3]
            ground = Image.new("RGBA", im.size, (255, 0, 255, 255))
            flat = Image.alpha_composite(ground, im).convert("RGB")
        total = a.size
        clear = 100.0 * np.count_nonzero(a < 8) / total
        solid = 100.0 * np.count_nonzero(a > 247) / total
        cells.append((flat, float(start_s or 0) + int(i) / float(fps), clear, solid,
                      max(0.0, 100.0 - clear - solid)))

    tw = 240
    th = max(1, round(cells[0][0].height * tw / cells[0][0].width))
    band, head, cols = 30, 26, 4
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, head + rows * (th + band)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((6, 7), f"matte proof - {model} - {len(paths)} frames @ {fps:g} fps"
                   f" - anchor edge {anchor or 'none'} - magenta = transparent",
           fill=(235, 235, 235), font=font)

    for n, (flat, t, clear, solid, feather) in enumerate(cells):
        x, y = (n % cols) * tw, head + (n // cols) * (th + band)
        sheet.paste(flat.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 5, y + th + 3), f"t={t:6.2f}s", fill=(235, 235, 235), font=font)
        d.text((x + 5, y + th + 15),
               f"clear {clear:5.1f}  solid {solid:5.1f}  feather {feather:4.1f}",
               fill=(170, 170, 170), font=font)
    sheet.save(out_png)
    return out_png


def load_still(src, out_dir):
    """One image in, one src-000000.png out - the still-image path of "png".

    Deliberately PIL rather than ffmpeg: platform assets arrive extensionless
    (sha256/<hex>), which PIL sniffs by content, and ffmpeg's image2 demuxer
    picks a codec by extension. Any existing alpha is flattened onto WHITE,
    because generated stickers are asked for on a plain white ground and a
    black flatten would give the segmenter a different picture than the one
    the prompt described.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    with Image.open(src) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            ground = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(ground, rgba).convert("RGB")
        else:
            rgb = im.convert("RGB")
    dst = os.path.join(out_dir, "src-000000.png")
    rgb.save(dst)
    return [dst], rgb.size


def encode(kind, rgba_dir, mask_dir, out_path, fps, ffmpeg):
    if kind == "png":
        # A still: the single refined RGBA frame IS the deliverable.
        shutil.copyfile(os.path.join(rgba_dir, "r-000000.png"), out_path)
        return
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
    # Default ON: the agent that needs it off is matting two separate subjects
    # and knows it, whereas the agent that needs it on does not know the food
    # photos are coming.
    ap.add_argument("--drop-detached", dest="drop_detached",
                    action="store_true", default=True)
    ap.add_argument("--no-drop-detached", dest="drop_detached", action="store_false")
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
    still = a.output == "png"
    if still:
        src_w = src_h = 0
        src_fps = out_fps = 1.0
    else:
        src_w, src_h, src_fps = probe(a.video, ffprobe)
        out_fps = a.fps or src_fps
    print(f"[matte] source {src_w}x{src_h} @ {src_fps:.3f} fps -> out {out_fps:.3f} fps", flush=True)
    if a.temporal_median and a.temporal_median > 1:
        print(f"[matte] WARNING temporal_median={a.temporal_median}: a per-pixel median "
              "erodes edges that move more than a pixel or two between frames", flush=True)

    work = os.path.dirname(os.path.abspath(a.out_path))
    frames_dir = os.path.join(work, "_frames")
    emit("PHASE", "extracting frames")
    if still:
        frames, (src_w, src_h) = load_still(a.video, frames_dir)
    else:
        frames = extract_frames(a.video, frames_dir, ffmpeg, a.start_s, a.end_s,
                                a.fps, a.scale, src_w, src_h)
    if not frames:
        raise SystemExit("no frames extracted - check start_s/end_s against the clip length")
    print(f"[matte] {len(frames)} frames", flush=True)

    import time
    t0 = time.time()
    alphas, ep = segment(frames, os.path.join(work, "_alpha"), a.model, providers)
    per_frame = (time.time() - t0) / len(frames)
    # The number the platform asked for, on its own line so it is greppable.
    # Includes session creation (~5 s for a cached engine), so short clips read
    # slower per frame than long ones.
    print(f"[matte] MEASURED {per_frame:.3f} s/frame for {a.model} at "
          f"{src_w}x{src_h} over {len(frames)} frames ep={ep}", flush=True)

    anchor = None
    if a.drop_detached:
        anchor, totals = anchor_edge(alphas)
        print(f"[matte] anchor edge {anchor} (mean alpha per edge: "
              + ", ".join(f"{k}={v / max(1, len(totals)):.1f}" for k, v in totals.items())
              + ")", flush=True)

    emit("PHASE", "refining")
    rgba_dir = os.path.join(work, "_rgba")
    refine(alphas, frames, rgba_dir, os.path.join(work, "_mask"),
           a.close_px, a.feather_px, a.temporal_median, anchor)

    # Before encode() and before the cleanup below: the proof is built from the
    # refined RGBA frames, so it shows the alpha the consumer actually gets -
    # after drop_detached, after feather - and those directories do not survive
    # this function.
    # A still needs no proof sheet: the cut-out PNG is itself the picture a
    # reviewer would look at, and the sheet's tiles assume a clip.
    if not still:
        emit("PHASE", "proof sheet")
        proof = proof_sheet(rgba_dir, os.path.splitext(a.out_path)[0] + "-proof.png",
                            a.start_s, out_fps, a.model, anchor)
        if proof:
            print(f"[matte] PROOF {proof}", flush=True)

    emit("PHASE", "encoding")
    encode(a.output, rgba_dir, os.path.join(work, "_mask"),
           a.out_path, out_fps, ffmpeg)
    for d in ("_frames", "_alpha", "_rgba", "_mask"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    emit("PROGRESS", "100")
    print(f"[matte] WROTE {a.out_path}", flush=True)


if __name__ == "__main__":
    main()
