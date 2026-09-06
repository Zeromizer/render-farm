"""PC-side smoke test for the H3 ComfyUI setup. No Supabase involved.

  ..\\..\\.venv\\Scripts\\python smoke.py                       # t2v 832x480 x 73f turbo
  ..\\..\\.venv\\Scripts\\python smoke.py --mode i2v --first-frame C:\\path\\still.png
  ..\\..\\.venv\\Scripts\\python smoke.py --mode r2v --ref-image a.png --ref-image b.png
  ..\\..\\.venv\\Scripts\\python smoke.py --resolution 768p --duration 5 --no-turbo

Starts ComfyUI if it is down, logs VRAM before/after, times the run, writes
the mp4 next to this file (or --out).
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # worker/ for config
sys.path.insert(0, _HERE)

import comfy_client  # noqa: E402
import graphs  # noqa: E402
import segments  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vram(tag):
    try:
        d = (comfy_client.system_stats().get("devices") or [{}])[0]
        log(f"{tag}: vram_free={d.get('vram_free', 0) // (1 << 20)} MiB torch_free={d.get('torch_vram_free', 0) // (1 << 20)} MiB")
    except Exception as e:  # noqa: BLE001
        log(f"{tag}: system_stats failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=graphs.MODES, default="t2v")
    ap.add_argument("--prompt", default="A lone lighthouse on a rocky shore at dusk, waves crashing, "
                                        "gulls calling, wind and surf audible, slow cinematic push-in.")
    ap.add_argument("--duration", type=float, default=3)
    ap.add_argument("--resolution", choices=sorted(graphs.SHORT_EDGE), default="480p")
    ap.add_argument("--ratio", choices=sorted(graphs.RATIOS), default="16:9")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--no-turbo", action="store_true")
    ap.add_argument("--first-frame")
    ap.add_argument("--last-frame")
    ap.add_argument("--ref-image", action="append", default=[])
    ap.add_argument("--ref-video", action="append", default=[])
    ap.add_argument("--ref-audio", action="append", default=[])
    ap.add_argument("--source", help="upscale mode: local clip to upscale")
    ap.add_argument("--upscale", type=float, help="generation modes: SeedVR2 factor to apply afterwards")
    ap.add_argument("--factor", type=float, default=2.0, help="upscale mode: multiplier")
    ap.add_argument("--shorter-size", type=int, help="upscale: target short edge px (overrides factor)")
    ap.add_argument("--color", default="wavelet", choices=sorted(graphs.UPSCALE_COLOR_METHODS))
    ap.add_argument("--frames-per-chunk", type=int, help="4n+1; default auto from free VRAM")
    ap.add_argument("--temporal-overlap", type=int, default=1)
    ap.add_argument("--blend", type=float, default=0.5, help="seedvr2: share of SeedVR2 in the output; rest is lanczos")
    ap.add_argument("--method", choices=("lanczos", "seedvr2"), default="lanczos", help="upscale method (default lanczos)")
    ap.add_argument("--segment-frames", type=int, default=segments.DEFAULT_SEGMENT_FRAMES,
                    help="upscale long clips in segments of this many frames (RAM ceiling)")
    ap.add_argument("--timeout-min", type=int, default=60)
    ap.add_argument("--out")
    ap.add_argument("--no-free", action="store_true", help="keep models loaded after the run")
    a = ap.parse_args()

    comfy_client.ensure_server(log)
    vram("before")

    names = {}
    if a.first_frame:
        names["first_frame"] = comfy_client.upload_input(a.first_frame)
    if a.last_frame:
        names["last_frame"] = comfy_client.upload_input(a.last_frame)
    if a.ref_image:
        names["ref_images"] = [comfy_client.upload_input(x) for x in a.ref_image]
    if a.ref_video:
        names["ref_videos"] = [comfy_client.upload_input(x) for x in a.ref_video]
    if a.ref_audio:
        names["ref_audios"] = [comfy_client.upload_input(x) for x in a.ref_audio]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    u = {"method": a.method, "color_correction": a.color, "temporal_overlap": a.temporal_overlap, "seed": a.seed,
         "blend": a.blend}
    if a.shorter_size:
        u["shorter_size"] = a.shorter_size
    else:
        u["factor"] = a.upscale or a.factor
    if a.frames_per_chunk:
        u["frames_per_chunk"] = a.frames_per_chunk

    def run_graph(graph, label, out):
        t0 = time.monotonic()
        pid = comfy_client.submit(graph)
        log(f"submitted {pid} ({label})")
        outputs = comfy_client.wait(pid, lambda ph, pr: log(f"  {ph}"), lambda: False, a.timeout_min * 60)
        comfy_client.fetch_output(outputs, out)
        log(f"{label} DONE in {time.monotonic() - t0:.0f}s -> {out} ({os.path.getsize(out)} bytes)")
        vram(f"after {label}")

    def upscale_clip(src, out):
        if a.method == "lanczos":
            t0 = time.monotonic()
            segments.lanczos(src, out, factor=u.get("factor"), shorter_size=u.get("shorter_size"), log=log)
            log(f"upscale TOTAL {time.monotonic() - t0:.0f}s -> {out}")
            return
        seg_dir = os.path.join(os.path.dirname(out) or _HERE, f"seg_{stamp}")
        os.makedirs(seg_dir, exist_ok=True)
        parts, fps, total = segments.split(src, a.segment_frames, seg_dir, log)
        t0 = time.monotonic()
        done = []
        for i, part in enumerate(parts):
            name = comfy_client.upload_input(part)
            graph, meta = graphs.build_upscale(name, u, f"video_gen/smoke_{stamp}-seg{i:03d}")
            if i == 0:
                log(f"upscale graph: {meta} ({total} frames, {len(parts)} segment(s))")
            seg_out = out if len(parts) == 1 else os.path.join(seg_dir, f"up_{i:03d}.mp4")
            run_graph(graph, f"upscale {i + 1}/{len(parts)}", seg_out)
            done.append(seg_out)
            if len(parts) > 1:
                comfy_client.free()
        if len(parts) > 1:
            segments.concat(done, src, out, log)
        log(f"upscale TOTAL {time.monotonic() - t0:.0f}s -> {out}")

    if a.mode == "upscale":
        if not a.source:
            ap.error("--mode upscale needs --source")
        out = a.out or os.path.join(_HERE, f"smoke_upscale_{stamp}.mp4")
        upscale_clip(a.source, out)
    else:
        p = {"prompt": a.prompt, "duration_s": a.duration, "resolution": a.resolution,
             "ratio": a.ratio, "seed": a.seed, "turbo": not a.no_turbo}
        if a.steps:
            p["steps"] = a.steps
        graph, meta = graphs.build(a.mode, p, names, f"video_gen/smoke_{stamp}")
        log(f"graph: {meta}")
        out = a.out or os.path.join(_HERE, f"smoke_{a.mode}_{stamp}.mp4")
        run_graph(graph, "generate", out)
        if a.upscale:
            comfy_client.free()
            upscale_clip(out, os.path.splitext(out)[0] + "-upscaled.mp4")
    if not a.no_free:
        comfy_client.free()
        time.sleep(3)
        vram("after /free")


if __name__ == "__main__":
    main()
