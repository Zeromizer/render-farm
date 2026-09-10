"""PC-side smoke test for the H3 ComfyUI setup. No Supabase involved.

  ..\\..\\.venv\\Scripts\\python smoke.py                       # t2v 832x480 x 73f turbo
  ..\\..\\.venv\\Scripts\\python smoke.py --mode i2v --first-frame C:\\path\\still.png
  ..\\..\\.venv\\Scripts\\python smoke.py --mode r2v --ref-image a.png --ref-image b.png
  ..\\..\\.venv\\Scripts\\python smoke.py --resolution 768p --duration 5 --no-turbo

Latent path (mmh3_media + MinimaxH3LatentUpscaler3D, see docs/h3-latent-upscale-pc-handoff.md):
  smoke.py --preflight-only --object-info-dump ..\\tests\\fixtures\\object_info_pc.json
  smoke.py --mode i2v --first-frame still.png --seed 42 --save-latent          # writes <out>.mmh3
  smoke.py --mode upscale --source clip.mp4 --latent clip.mmh3 --method h3_latent_upscale --variant tile --shorter-size 1080
  smoke.py --mode upscale --source clip.mp4 --method h3_latent_upscale --variant decoded --shorter-size 1080
  smoke.py --mode upscale --source clip.mp4 --latent clip.mmh3 --method h3_latent_upscale --variant full --shorter-size 1080

Starts ComfyUI if it is down, logs VRAM before/after (min free during the
latent refine), times the run, writes the mp4 next to this file (or --out),
plus <out>-upscale.json / -fidelity.json / -compare.mp4 for the latent path.
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # worker/ for config
sys.path.insert(0, _HERE)

import comfy_client  # noqa: E402
import graphs  # noqa: E402
import segments  # noqa: E402
from videogen import estimate, fidelity, graphs_h3, h3_preflight, provenance  # noqa: E402


def log(msg):
    # Probe texts and ComfyUI messages can carry emoji; a cp1252 console or pipe must not kill the run.
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{time.strftime('%H:%M:%S')}] {str(msg).encode('ascii', 'replace').decode()}", flush=True)


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
    ap.add_argument("--upscale", type=float, help="generation modes: upscale factor to apply afterwards (with --method)")
    ap.add_argument("--factor", type=float, default=2.0, help="upscale mode: multiplier")
    ap.add_argument("--shorter-size", type=int, help="upscale: target short edge px (overrides factor)")
    ap.add_argument("--color", default="wavelet", choices=sorted(graphs.UPSCALE_COLOR_METHODS))
    ap.add_argument("--frames-per-chunk", type=int, help="4n+1; default auto from free VRAM")
    ap.add_argument("--temporal-overlap", type=int, default=1)
    ap.add_argument("--blend", type=float, default=0.5, help="seedvr2: share of SeedVR2 in the output; rest is lanczos")
    ap.add_argument("--method", choices=("lanczos", "seedvr2", graphs_h3.METHOD), default="lanczos",
                    help="upscale method (default lanczos)")
    ap.add_argument("--segment-frames", type=int, default=segments.DEFAULT_SEGMENT_FRAMES,
                    help="seedvr2: upscale long clips in segments of this many frames (RAM ceiling)")
    # --- h3_latent_upscale / save_latent
    ap.add_argument("--save-latent", action="store_true", help="generation modes: also save the .mmh3 packet")
    ap.add_argument("--variant", choices=graphs_h3.VARIANTS, default="tile", help="h3_latent_upscale variant")
    ap.add_argument("--latent", help="h3_latent_upscale tile/full: local .mmh3 packet of --source")
    ap.add_argument("--denoise", type=float, default=0.0, help="h3: 0 = source-aware, else 0.05-0.5")
    ap.add_argument("--steps-override", type=int, default=0, help="h3: 0 = source profile, else 1-20")
    ap.add_argument("--tile-w", type=int, default=graphs_h3.TILE_DEFAULTS["tile_width"])
    ap.add_argument("--tile-h", type=int, default=graphs_h3.TILE_DEFAULTS["tile_height"])
    ap.add_argument("--tile-overlap", type=int, default=graphs_h3.TILE_DEFAULTS["tile_overlap"])
    ap.add_argument("--context-padding", type=int, default=graphs_h3.TILE_DEFAULTS["context_padding"])
    ap.add_argument("--tile-overlap-mode", choices=graphs_h3.TILE_MODE_OPTIONS["overlap_mode"], default=None)
    ap.add_argument("--tile-blend-mode", choices=graphs_h3.TILE_MODE_OPTIONS["blend_mode"], default=None,
                    help="reprocess + linear/half_cosine blends the tile overlaps (hard leaves visible seams)")
    ap.add_argument("--tile-traversal", choices=graphs_h3.TILE_MODE_OPTIONS["traversal"], default=None)
    ap.add_argument("--tile-context-source", choices=graphs_h3.TILE_MODE_OPTIONS["context_source"], default=None)
    ap.add_argument("--no-force-unload", action="store_true")
    ap.add_argument("--fp16-accumulation", choices=graphs_h3.FP16_ACCUMULATION_OPTIONS, default="Default")
    ap.add_argument("--allow-large-full", action="store_true", help="h3 full: bypass the pixel-frame guard")
    ap.add_argument("--no-fidelity", action="store_true", help="h3: skip the ssim/psnr check and compare video")
    ap.add_argument("--critical-cells", default="", help="h3 fidelity: 'r,c;r,c' cells (4x4 grid) that fail the run "
                    "when they drop > 0.10 below the frame mean (badge/grille/wheel/plate)")
    ap.add_argument("--preflight-only", action="store_true",
                    help="h3: fetch /object_info, check the graphs, print the verdict and exit")
    ap.add_argument("--object-info-dump", help="write /object_info (trimmed to the classes the graphs use) to this json")
    ap.add_argument("--timeout-min", type=int, default=60)
    ap.add_argument("--out")
    ap.add_argument("--no-free", action="store_true", help="keep models loaded after the run")
    a = ap.parse_args()

    comfy_client.ensure_server(log)
    vram("before")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    def h3_params():
        u = {"method": graphs_h3.METHOD, "variant": a.variant, "denoise": a.denoise, "steps_override": a.steps_override,
             "seed": a.seed, "force_unload": not a.no_force_unload, "fp16_accumulation": a.fp16_accumulation,
             "tile_width": a.tile_w, "tile_height": a.tile_h, "tile_overlap": a.tile_overlap,
             "context_padding": a.context_padding, "allow_large_full": a.allow_large_full,
             "fidelity": {"enabled": not a.no_fidelity,
                          "critical_cells": [[int(x) for x in cell.split(",")] for cell in a.critical_cells.split(";") if cell.strip()]},
             "overlap_mode": a.tile_overlap_mode, "blend_mode": a.tile_blend_mode, "traversal": a.tile_traversal,
             "context_source": a.tile_context_source}
        if a.shorter_size:
            u["shorter_size"] = a.shorter_size
        else:
            u["factor"] = a.upscale or a.factor
        if a.latent:
            u["latent"] = {"bucket": "local", "path": a.latent}
        return u

    if a.preflight_only or a.object_info_dump:
        info = comfy_client.object_info()
        gen_graph, _ = graphs_h3.build_generation_with_packet("t2v", {"prompt": "x"}, {}, "video_gen/smoke", "mmh3/smoke")
        u = graphs_h3.validate_upscale_params(dict(h3_params(), variant="tile", latent={"bucket": "b", "path": "p"}), "upscale",
                                              dev=True)
        tile_graph, _ = graphs_h3.build_latent_upscale(u, os.path.abspath("x.mmh3"), (480, 832), "video_gen/smoke")
        full_graph, _ = graphs_h3.build_latent_upscale(dict(u, variant="full"), os.path.abspath("x.mmh3"), (480, 832), "video_gen/smoke")
        dec_graph, _ = graphs_h3.build_decoded_upscale(graphs_h3.validate_upscale_params(dict(h3_params(), variant="decoded"), "upscale", dev=True),
                                                       "x.mp4", (480, 832), "video_gen/smoke")
        classes = set()
        ok = True
        for name, g in (("generation+packet", gen_graph), ("tile", tile_graph), ("full", full_graph), ("decoded", dec_graph)):
            classes |= {n["class_type"] for n in g.values()}
            res = h3_preflight.check(info, g)
            msg = h3_preflight.error_message(res)
            log(f"preflight {name}: {'OK' if not msg else 'FAIL'}"
                + (f" (dropped optional {res['dropped_optional']})" if res["dropped_optional"] else "")
                + (f" unknown inputs {res['unknown_inputs']}" if res["unknown_inputs"] else ""))
            if msg:
                ok = False
                log("  " + msg)
        if a.object_info_dump:
            trimmed = {c: info[c] for c in sorted(classes) if c in info}
            trimmed["__meta__"] = {"written": stamp, "classes_missing": sorted(c for c in classes if c not in info),
                                   "comfyui": (comfy_client.system_stats().get("system") or {}).get("comfyui_version")}
            with open(a.object_info_dump, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, indent=1)
            log(f"object_info ({len(trimmed) - 1} classes) -> {a.object_info_dump}")
        if a.preflight_only:
            sys.exit(0 if ok else 2)

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

    u = {"method": a.method, "color_correction": a.color, "temporal_overlap": a.temporal_overlap, "seed": a.seed,
         "blend": a.blend}
    if a.shorter_size:
        u["shorter_size"] = a.shorter_size
    else:
        u["factor"] = a.upscale or a.factor
    if a.frames_per_chunk:
        u["frames_per_chunk"] = a.frames_per_chunk

    def run_graph(graph, label, out, extra_mmh3=None, prefix_glob=None, hint=None, texts=False):
        t0 = time.monotonic()
        pid = comfy_client.submit(graph)
        log(f"submitted {pid} ({label})")
        outputs = comfy_client.wait(pid, lambda ph, fr, eta: log(f"  {ph} ({fr:.0%}, ~{eta:.0f}s left)"),
                                    lambda: False, a.timeout_min * 60, hint=hint)
        comfy_client.fetch_output(outputs, out)
        log(f"{label} DONE in {time.monotonic() - t0:.0f}s -> {out} ({os.path.getsize(out)} bytes)")
        got = {}
        if extra_mmh3:
            comfy_client.fetch_file_output(outputs, extra_mmh3, prefix_glob=prefix_glob)
            log(f"packet -> {extra_mmh3} ({os.path.getsize(extra_mmh3)} bytes, packet={provenance.is_packet(extra_mmh3)})")
            got["mmh3"] = extra_mmh3
        if texts:
            got["texts"] = comfy_client.fetch_texts(outputs)
        vram(f"after {label}")
        return got

    def h3_upscale(src, out, latent_local):
        from studio import post
        src_info = post.info(src)
        # smoke is the dev tool: tile/decoded allowed here (the production worker needs H3_TILE_DEV=1)
        uu = graphs_h3.validate_upscale_params(h3_params(), "upscale", (src_info["width"], src_info["height"]), src_info["frames"],
                                               dev=True)
        dims = uu["_dims"]
        packet_prefix = f"mmh3/smoke_{stamp}-upscaled" if a.save_latent else None
        if uu["variant"] == "decoded":
            kept = graphs_h3.av_boundary_frames(src_info["frames"])
            if not kept:
                raise SystemExit(f"decoded needs at least {graphs_h3.AV_BOUNDARY_MIN} frames, got {src_info['frames']}")
            if kept != src_info["frames"]:
                trimmed = os.path.splitext(out)[0] + "-source.mp4"
                segments.trim_frames(src, trimmed, kept, log=log)
                log(f"decoded: source has {src_info['frames']} frames; keeping the first {kept} (AV-exact H3 length)")
                src = trimmed
                src_info = post.info(src)
            name = comfy_client.upload_input(src)
            graph, meta = graphs_h3.build_decoded_upscale(uu, name, (src_info["width"], src_info["height"]),
                                                          f"video_gen/smoke_{stamp}-h3up", prompt=a.prompt if a.prompt else None,
                                                          packet_prefix=packet_prefix)
        else:
            if not latent_local:
                ap.error("--variant tile/full needs --latent <packet.mmh3> (or --variant decoded)")
            graph, meta = graphs_h3.build_latent_upscale(uu, os.path.abspath(latent_local), (src_info["width"], src_info["height"]),
                                                         f"video_gen/smoke_{stamp}-h3up", packet_prefix=packet_prefix)
        log(f"h3 upscale graph: {json.dumps({k: v for k, v in meta.items() if k != 'tiles'})}"
            + (f" tiles={meta['tiles']['count']}" if meta.get("tiles") else ""))
        info = comfy_client.object_info()
        graph, res = h3_preflight.run(info, graph)
        log(f"preflight OK" + (f" (dropped {res['dropped_optional']})" if res["dropped_optional"] else ""))
        plan = estimate.latent_upscale_plan(uu, src_info["width"], src_info["height"], src_info["frames"])
        log(f"estimate: {plan['total'] / 60:.1f} min ({plan['tiles']} tiles x {plan['steps']} steps x {plan['step_seconds']:.0f} s)")
        comfy_client.free()
        sampler = provenance.VramSampler(comfy_client.system_stats).start()
        raw = os.path.splitext(out)[0] + "-raw.mp4"
        t0 = time.monotonic()
        got = run_graph(graph, f"h3 upscale {uu['variant']}", raw,
                        extra_mmh3=(os.path.splitext(out)[0] + "-upscaled.mmh3") if packet_prefix else None,
                        prefix_glob=packet_prefix, hint=estimate.latent_upscale_hint(uu, src_info["width"], src_info["height"], src_info["frames"]),
                        texts=True)
        vr = sampler.stop()
        t_prompt = time.monotonic() - t0
        log(f"VRAM during refine: {vr}")
        for node, texts in (got.get("texts") or {}).items():
            for t in texts:
                log(f"probe {node}: {t[:400]}")
        crop = graphs_h3.crop_args(dims["refine"], dims["req"])
        if crop:
            segments.crop_exact(raw, out, dims["req"][0], dims["req"][1], log=log)
        else:
            os.replace(raw, out)
        record = {"meta": meta, "params": {k: v for k, v in uu.items() if k != "_dims"}, "timing_s": {"prompt": round(t_prompt, 1)},
                  "vram_mb": vr, "source": src_info, "probes": got.get("texts")}
        if not a.no_fidelity:
            t1 = time.monotonic()
            fid_dir = os.path.splitext(out)[0] + "-fidelity"
            fid = fidelity.measure(src, out, fid_dir, thresholds=uu["fidelity"], log=log)
            fidelity.write_json(fid, os.path.splitext(out)[0] + "-fidelity.json")
            fidelity.compare_video(src, out, os.path.splitext(out)[0] + "-compare.mp4", log=log)
            record["timing_s"]["fidelity"] = round(time.monotonic() - t1, 1)
            record["fidelity"] = {"verdict": fid["verdict"], "frames": fid["frames"], "ssim": fid["ssim"]["mean"],
                                  "psnr": fid["psnr"]["mean"], "cells_drift": len(fid["flags"]["cells_drift"]),
                                  "cells_critical": len(fid["flags"].get("cells_critical") or [])}
            if fid["verdict"] == "fail":
                log(f"fidelity FAIL: {len(fid['flags']['cells_critical'])} critical cell-frames drifted > "
                    f"{uu['fidelity']['cell_drop_critical']} (a production job would fail here)")
        with open(os.path.splitext(out)[0] + "-upscale.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
        log(f"h3 upscale TOTAL {time.monotonic() - t0:.0f}s -> {out}; record {os.path.splitext(out)[0]}-upscale.json")

    def upscale_clip(src, out, latent_local=None):
        if a.method == graphs_h3.METHOD:
            return h3_upscale(src, out, latent_local)
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
        upscale_clip(a.source, out, a.latent)
    else:
        p = {"prompt": a.prompt, "duration_s": a.duration, "resolution": a.resolution,
             "ratio": a.ratio, "seed": a.seed, "turbo": not a.no_turbo}
        if a.steps:
            p["steps"] = a.steps
        out = a.out or os.path.join(_HERE, f"smoke_{a.mode}_{stamp}.mp4")
        packet_local = None
        if a.save_latent:
            packet_prefix = f"mmh3/smoke_{stamp}"
            graph, meta = graphs_h3.build_generation_with_packet(a.mode, p, names, f"video_gen/smoke_{stamp}", packet_prefix)
            graph, res = h3_preflight.run(comfy_client.object_info(), graph)
            log(f"graph (+packet): {meta}")
            packet_local = os.path.splitext(out)[0] + ".mmh3"
            run_graph(graph, "generate", out, extra_mmh3=packet_local, prefix_glob=packet_prefix)
            try:
                man = provenance.packet_manifest(packet_local)
                log(f"packet: {json.dumps(provenance.packet_summary(man))[:600]}")
            except Exception as e:  # noqa: BLE001
                log(f"packet manifest unreadable: {e}")
        else:
            graph, meta = graphs.build(a.mode, p, names, f"video_gen/smoke_{stamp}")
            log(f"graph: {meta}")
            run_graph(graph, "generate", out)
        if a.upscale or (a.method == graphs_h3.METHOD and a.shorter_size):
            comfy_client.free()
            upscale_clip(out, os.path.splitext(out)[0] + "-upscaled.mp4", packet_local)
    if not a.no_free:
        comfy_client.free()
        time.sleep(3)
        vram("after /free")


if __name__ == "__main__":
    main()
