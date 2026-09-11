"""Isolated After Effects smoke test, no Supabase involved.

  cd worker
  ..\\.venv\\Scripts\\python.exe -m aftereffects.smoke --workspace C:\\Coding\\ae-smoke            # real AE
  ..\\.venv\\Scripts\\python.exe -m aftereffects.smoke --workspace C:\\Coding\\ae-smoke --fake     # no AE: fake host
  ... --revise          also re-open the delivered project, change one text + its timing, re-render
  ... --settings f.json use another text_overlay_v1 settings object
  ... --fake-mov        fake host emits a QuickTime Animation .mov (the "Lossless with Alpha" path)

Default recipe: 5 s, 1080x1920, 30 fps, transparent, two editable text layers
and one rounded rectangle, with position/opacity/scale keyframes. Every run
gets its own attempt-<timestamp> directory under --workspace with
project/, render/, out/ (master, review, contact sheet, bundle, manifest).
"""
import argparse
import json
import os
import shutil
import sys
import time

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from aftereffects import ae_host, media, pipeline, schema  # noqa: E402

DEFAULT_REQUEST = {
    "schema_version": 1,
    "recipe": "text_overlay_v1",
    "composition": "Main",
    "output_profile": "prores4444_alpha",
    "org_id": "smoke-org", "job_id": "smoke-job",
    "assets": [],
    "settings": {
        "composition": {"width": 1080, "height": 1920, "fps": 30, "duration_s": 5},
        "shapes": [
            {"id": "pill", "kind": "rect", "size": [720, 150], "radius": 75, "color": "#FF6A00",
             "position": [540, 1500], "in_s": 0.6, "out_s": 5.0, "fade_in_s": 0.3, "fade_out_s": 0.4,
             "scale_from": 60, "scale_s": 0.45}
        ],
        "texts": [
            {"id": "headline", "text": "Motorised Blinds\nFrom $299", "font": "Arial-BoldMT", "size": 110,
             "color": "#FFFFFF", "position": [540, 760], "justify": "center", "in_s": 0.2, "out_s": 5.0,
             "fade_in_s": 0.4, "fade_out_s": 0.4, "slide_from": [0, 80], "shadow": True},
            {"id": "cta", "text": "Book a free measure", "font": "ArialMT", "size": 56, "color": "#FFFFFF",
             "position": [540, 1500], "justify": "center", "in_s": 0.8, "out_s": 5.0,
             "fade_in_s": 0.3, "fade_out_s": 0.4}
        ],
        "images": []
    }
}

DEFAULT_CHANGES = [{"id": "headline", "text": "Motorised Blinds\nNow $249", "in_s": 0.5, "out_s": 4.6}]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _summary(result):
    c = result["checks"]
    t = result["timings"]
    return {
        "proofs": result.get("proofs"), "easing_resolved": result.get("easing_resolved"), "anchors_resolved": result.get("anchors_resolved"),
        "master": result["master"], "review": result["review"], "bundle": result["bundle"],
        "contact_sheet": result["contact_sheet"], "manifest": result["manifest"],
        "frames": c["frames"], "size": f"{c['width']}x{c['height']}", "fps": c["fps"], "pix_fmt": c["pix_fmt"],
        "alpha": c["alpha"], "timings": t, "editable_layers": [
            {k: L.get(k) for k in ("name", "kind", "text", "font", "in_s", "out_s", "stretch", "stroke", "effects", "matte", "masks", "keys", "motion_blur", "inner")} for L in result["inspect"]["editable_layers"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.path.join("C:\\", "Coding", "ae-smoke"))
    ap.add_argument("--fake", action="store_true", help="no After Effects: the fake host renders the model")
    ap.add_argument("--fake-mov", action="store_true", help="fake host writes a QuickTime Animation mov")
    ap.add_argument("--revise", action="store_true")
    ap.add_argument("--settings", help="JSON file with a settings object for --recipe")
    ap.add_argument("--recipe", default="text_overlay_v1", help="text_overlay_v1 | text_overlay_v2")
    ap.add_argument("--proof", action="append", type=int, default=[], help="v2: proof frame (repeatable)")
    ap.add_argument("--timeout-minutes", type=float, default=20)
    ap.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                    help="stage a local image as request asset NAME (copied WITHOUT a suffix and sniffed, "
                         "exactly like a content-addressed storage object) and add an image layer for it")
    ap.add_argument("--storage-asset", action="append", default=[], metavar="NAME=BUCKET/PATH[@SHA256]",
                    help="fetch request asset NAME from Supabase storage through the worker's own downloader "
                         "(needs the worker .env); the real farm staging path end to end")
    ap.add_argument("--cancel-after-s", type=float, help="report cancel_requested after this many seconds")
    args = ap.parse_args()

    req_raw = json.loads(json.dumps(DEFAULT_REQUEST))
    req_raw["recipe"] = args.recipe
    if args.settings:
        with open(args.settings, encoding="utf-8") as f:
            req_raw["settings"] = json.load(f)
    if args.proof:
        req_raw["settings"].setdefault("output_extras", {})["proof_frames_f"] = args.proof
    staged_src = {}
    for i, spec in enumerate(args.asset):
        name, _, path = spec.partition("=")
        req_raw["assets"].append({"name": name, "bucket": "local", "path": os.path.basename(path), "kind": "image",
                                  "sha256": pipeline.sha256_file(path)})
        if "layers" in req_raw["settings"]:
            pass  # v2 settings files reference their assets themselves
        else:
            req_raw["settings"]["images"].append({"id": f"img_{name}", "asset": name, "position": [540, 300 + 260 * i],
                                                  "scale": 40, "in_s": 0.0, "out_s": 5.0, "fade_in_s": 0.3, "fade_out_s": 0.3})
        staged_src[name] = path
    storage_src = {}
    for spec in args.storage_asset:
        name, _, rest = spec.partition("=")
        loc, _, digest = rest.partition("@")
        bucket, _, path = loc.partition("/")
        entry = {"name": name, "bucket": bucket, "path": path, "kind": "image"}
        if digest:
            entry["sha256"] = digest
        req_raw["assets"].append(entry)
        storage_src[name] = entry
    request = schema.validate_request(req_raw)
    t_launch = time.monotonic()

    def cancel_check():
        return args.cancel_after_s is not None and time.monotonic() - t_launch > args.cancel_after_s

    if args.fake or args.fake_mov:
        from aftereffects.fake_host import FakeHost
        host = FakeHost(log=_log, om_kind="mov" if args.fake_mov else "sequence", ffmpeg=media.tool("ffmpeg"))
        _log("FAKE HOST: outputs are synthetic boxes, not After Effects renders")
    else:
        host = ae_host.RealHost(log=_log)
    _log(f"host: {host.info()}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ws = os.path.join(args.workspace, f"attempt-{stamp}")
    os.makedirs(os.path.join(ws, "inputs"), exist_ok=True)
    # Stage through aftereffects/staging.py, the runner's path: local files are
    # copied under their bare asset name (no suffix) so the content sniffer,
    # not the filename, decides what they are.
    from aftereffects import staging

    def local_download(bucket, path, inputs_dir, name, log):
        if bucket == "local":
            dst = os.path.join(inputs_dir, name)
            shutil.copy2(staged_src[name], dst)
            log(f"copied {staged_src[name]} -> {dst} (no suffix)")
            return dst
        from runners import gate_common   # imports db: needs the worker .env
        return gate_common.download(bucket, path, inputs_dir, name, log)

    assets_local = staging.stage_assets(request, os.path.join(ws, "inputs"), _log, local_download)
    for name, p in assets_local.items():
        _log(f"staged asset {name}: {os.path.basename(p)}")
    timeout = args.timeout_minutes * 60
    t0 = time.monotonic()
    try:
        result = pipeline.run(request, ws, host, assets_local, _log, cancel_check, timeout,
                              progress=lambda f: _log(f"progress {f:.2f}") if int(f * 100) % 10 == 0 else None,
                              phase=lambda p: None, name="smoke")
    except BaseException as e:  # noqa: BLE001 - report exactly what the farm row would see
        _log(f"FAILED after {time.monotonic() - t0:.1f}s: {type(e).__name__}: {e}")
        with open(os.path.join(ws, "out", "manifest.json"), encoding="utf-8") as f:
            _log(f"manifest error: {json.load(f).get('error')}")
        _log(f"pids recorded: {open(os.path.join(ws, 'pids.json')).read() if os.path.exists(os.path.join(ws, 'pids.json')) else 'none'}")
        raise SystemExit(2)
    _log(f"AUTHOR+RENDER OK in {time.monotonic() - t0:.1f}s")
    print(json.dumps(_summary(result), indent=1, default=str))

    if args.revise:
        ws2 = os.path.join(args.workspace, f"attempt-{stamp}-rev")
        os.makedirs(os.path.join(ws2, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(ws2, "project"), exist_ok=True)
        src = os.path.join(ws2, "project", "source.aep")
        shutil.copy2(result["project"], src)
        assets2 = {}
        for name, path in assets_local.items():
            dst = os.path.join(ws2, "inputs", os.path.basename(path))
            shutil.copy2(path, dst)
            assets2[name] = dst
        t1 = time.monotonic()
        rev = pipeline.revise(request, DEFAULT_CHANGES, src, assets2, ws2, host, _log, cancel_check, timeout, name="smoke-v2")
        _log(f"REVISE+RENDER OK in {time.monotonic() - t1:.1f}s; applied {rev['revise'].get('applied')}")
        print(json.dumps(_summary(rev), indent=1, default=str))


if __name__ == "__main__":
    main()
