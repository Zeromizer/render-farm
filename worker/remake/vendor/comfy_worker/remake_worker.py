# SPDX-License-Identifier: Apache-2.0
"""Viggle Remake render worker on ComfyUI as a library: one process, no ComfyUI server.

    ComfyUI/venv/bin/python remake_worker.py [--gpu 0] [--home ~/.viggle-remake] [--max-480 345] [--max-736 345] \
        [-- <ComfyUI flags, e.g. --vram-headroom 6>]

Renders the viggle-remake MCP's file queue with the graph of comfyui/workflows/remake_*_api.json: the mode's int8
transformer -> RemakeKeyframeUntil -> BasicGuider, euler on the mode's six-step sigmas, RemakeConditioning +
RemakeNoise, and RemakePrefix for a hop. The queue protocol is the official worker's (viggle_remake_mcp/worker.py).
"""
import os
import sys
from pathlib import Path

if "--gpu" in sys.argv[:-1]:  # before torch
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
COMFY = Path(__file__).resolve().parent / "ComfyUI"
split = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
# --vram-headroom: dynamic VRAM keeps 4 GB free, else a 311-frame 736 window OOMed on a 2.8 GB attention output
# before the weights make room (a later flag after -- wins)
ours, sys.argv = sys.argv[1:split], [str(COMFY / "main.py"), "--vram-headroom", "4", *sys.argv[split + 1:]]
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(COMFY / "custom_nodes"))

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import cuda_malloc  # noqa: E402,F401  as ComfyUI's own main does before torch
import main  # noqa: E402,F401  ComfyUI's setup (logging, dynamic VRAM, model paths); its server starts only as __main__

import argparse  # noqa: E402
import gc  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.ldm.minimax.model as H3  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.samplers  # noqa: E402
import nodes  # noqa: E402
from comfy_api.latest import InputImpl, Types  # noqa: E402
from comfy_extras.nodes_audio import vae_decode_audio  # noqa: E402
from comfy_extras.nodes_custom_sampler import Guider_Basic  # noqa: E402

R = importlib.import_module("ComfyUI-Viggle-Remake.nodes")

# Each block's MLP over the packed sequence FF_ROWS rows at a time, as the reference worker does: its S x 28672 bf16
# fc1 output alone is 11 GiB at 345 frames on the 736 canvas, which a 32 GB card does not have. int8 quantizes the
# activations per row, so the output is the same.
FF_ROWS = 16384
_mlp = H3.MLP.forward


def mlp_rows(self, x):
    if x.shape[-2] <= FF_ROWS:
        return _mlp(self, x)
    out = torch.empty_like(x)
    for a in range(0, x.shape[-2], FF_ROWS):
        out[..., a:a + FF_ROWS, :] = _mlp(self, x[..., a:a + FF_ROWS, :])
    return out


H3.MLP.forward = mlp_rows

MODES = ("swap", "edit")
FRAMES = range(90, 346, 17)  # 17n + 5, 24 fps
AREA = {480: 832 * 480, 736: 1280 * 736}
SIGMAS = {"swap": [1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0.0],
          "edit": [1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0.0]}
UNET = {m: f"viggle_remake_{m}_pruned_int8_convrot.safetensors" for m in MODES}
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE, AUDIO_VAE = "minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"


def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, help="card index; default: CUDA_VISIBLE_DEVICES as set")
    parser.add_argument("--home", default=os.environ.get("REMAKE_HOME", "~/.viggle-remake"))
    parser.add_argument("--max-480", type=int, default=345, help="longest render at 832x480 (17n+5, 0 = none)")
    parser.add_argument("--max-736", type=int, default=345, help="longest render at 1280x736 (17n+5, 0 = none)")
    args = parser.parse_args(ours)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    host = socket.gethostname()
    home = Path(args.home).expanduser()
    queue, workers = home / "queue", home / "workers"
    for p in (queue, workers):
        p.mkdir(parents=True, exist_ok=True)
    limit = {480: args.max_480, 736: args.max_736}
    state = {"job": None}
    beat_path, lock = workers / f"{host}_{gpu}.json", threading.Lock()

    def beat():
        with lock:
            tmp = beat_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"t": time.time(), "host": host, "gpu": gpu, "pid": os.getpid(),
                                       "job": state["job"], "modes": list(MODES),
                                       "max_frames": {str(k): v for k, v in limit.items()}}))
            tmp.rename(beat_path)

    def beat_loop():
        while True:
            try:
                beat()
            except OSError:  # a transient fs error must not stop the heartbeat
                pass
            time.sleep(10)

    t0 = time.time()
    clip = nodes.CLIPLoader().load_clip(TE, type="minimax")[0]
    vae = nodes.VAELoader().load_vae(VAE)[0]
    audio_vae = nodes.VAELoader().load_vae(AUDIO_VAE)[0]
    loaded = {}

    def model(m):
        # one transformer at a time (21 GB each): a mode switch drops the other one first
        if m not in loaded:
            loaded.clear()
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            loaded[m] = nodes.UNETLoader().load_unet(UNET[m], "default")[0]
        return loaded[m]

    model("swap")
    print(f"loaded in {time.time() - t0:.0f}s; max_frames {limit}", flush=True)
    threading.Thread(target=beat_loop, daemon=True).start()

    @torch.inference_mode()  # as ComfyUI's executor runs every node; without it autograd keeps the VAE's activations
    def render(job):
        m, nf, tier, refs, seed = job["mode"], job["num_frames"], job["tier"], job["refs"], job["seed"]
        assert m in MODES, f"mode must be one of {MODES}, got {m!r}"
        assert nf in FRAMES, f"num_frames must be 17n+5 in 90..345, got {nf}"
        assert tier in limit, f"tier must be one of {list(limit)}, got {tier}"
        if nf > limit[tier]:
            raise ValueError(f"num_frames {nf} exceeds this worker's max {limit[tier]} at tier {tier}")
        assert len(refs) <= 4, f"the node pack takes at most 4 refs, got {len(refs)}"
        pin = job.get("pin") or []
        assert len(pin) in (0, len(refs)) and all(0 <= k < nf for k in pin), (pin, len(refs), nf)

        source = InputImpl.VideoFromFile(job["cond"]).get_components().images
        h, w = source.shape[1], source.shape[2]
        assert h % 32 == 0 and w % 32 == 0 and h * w <= AREA[tier], \
            f"cond must be multiples of 32 with at most {AREA[tier]} pixels at tier {tier}, got {w}x{h}"
        # the soundtrack, as RemakeConditioning takes it (it pads and cuts to the render)
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", job["audio"], "-t", f"{nf / 24:.3f}", "-vn",
                              "-f", "f32le", "-ac", "2", "-ar", str(R.AUDIO_SR), "-"],
                             capture_output=True, check=True).stdout
        audio = {"waveform": torch.frombuffer(bytearray(pcm), dtype=torch.float32).view(-1, 2).T[None],
                 "sample_rate": R.AUDIO_SR}
        ims = [torch.from_numpy(np.array(Image.open(x).convert("RGB"))).float()[None] / 255 for x in refs]
        ims += [None] * (4 - len(ims))
        # short_edge = the cond's own: it is already on its canvas, so nothing is resized
        positive, latent = R.RemakeConditioning().run(clip, vae, audio_vae, source, job["prompt"], ims[0],
                                                      " ".join(map(str, pin)), nf, min(h, w), audio, *ims[1:])
        if job.get("prefix"):
            prev = InputImpl.VideoFromFile(job["prefix"]).get_components().images
            latent = R.RemakePrefix().run(vae, latent, prev, job.get("prefix_frames", 17), job.get("prefix_at", -1))[0]

        guider = Guider_Basic(R.RemakeKeyframeUntil().run(model(m), job.get("keyframe_until", 0.9))[0])
        guider.set_conds(positive)
        noise = R.RemakeNoise().run(positive, seed)[0]
        samples = guider.sample(noise.generate_noise(latent), latent["samples"], comfy.samplers.sampler_object("euler"),
                                torch.tensor(SIGMAS[m]), denoise_mask=latent["noise_mask"], seed=seed)
        out = {"samples": samples.to(mm.intermediate_device())}
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=nodes.VAEDecode().decode(vae, out)[0], audio=vae_decode_audio(audio_vae, out), frame_rate=Fraction(24)))
        dst = Path(job["out"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.mp4")
        video.save_to(str(tmp), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        tmp.rename(dst)

    def live_beats():
        beats = []
        for p in workers.glob("*.json"):
            try:
                hb = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if time.time() - hb["t"] < 120:
                beats.append(hb)
        return beats

    def requeue_orphans(beats):
        live = {hb["job"] for hb in beats}
        for run in queue.glob("*.run"):
            try:
                if time.time() - run.stat().st_mtime > 120 and run.stem not in live:
                    run.rename(run.with_suffix(".json"))
                    print(f"{run.stem}: orphaned, requeued", flush=True)
            except FileNotFoundError:
                pass

    def next_job(beats):
        others = [hb for hb in beats if (hb["host"], hb["gpu"]) != (host, gpu)]
        for p in sorted(queue.glob("*.json")):
            try:
                job = json.loads(p.read_text())
            except (OSError, ValueError):  # claimed meanwhile, or still being written
                continue
            if job.get("prefix") and not Path(job["prefix"]).exists():
                continue
            tier, nf = job.get("tier"), job.get("num_frames", 0)
            if nf > limit.get(tier, 0) and any(
                    job.get("mode") in hb["modes"] and hb["max_frames"].get(str(tier), 0) >= nf for hb in others):
                continue  # too long for this card, not for another live worker: leave it to that one
            run = p.with_suffix(".run")
            try:  # touch first: rename keeps the mtime, and a .run older than 120 s that no heartbeat names is requeued
                os.utime(p)
                p.rename(run)
            except FileNotFoundError:
                continue
            return run
        return None

    while True:
        beats = live_beats()
        requeue_orphans(beats)
        run = next_job(beats)
        if run is None:
            time.sleep(2)
            continue
        state["job"] = run.stem
        beat()
        try:
            job = json.loads(run.read_text())
            t = time.time()
            render(job)
            info = {"out": job["out"], "secs": round(time.time() - t, 1),
                    "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                    "host": host, "gpu": gpu, "mode": job["mode"], "sigmas": SIGMAS[job["mode"]]}
            tmp = run.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(info))
            tmp.rename(run.with_suffix(".done"))
            print(f"{run.stem}: {info}", flush=True)
        except Exception:
            tb = traceback.format_exc()
            run.with_suffix(".err").write_text(tb)
            run.unlink(missing_ok=True)
            print(f"{run.stem}: FAILED\n{tb}", flush=True)
            # the context is gone after a CUDA error, and after an OOM freeing the exception's tensors aborts the
            # process: exit before that happens (run_worker.sh restarts the worker)
            if "CUDA error" in tb or "OutOfMemoryError" in tb:
                os._exit(1)
        run.unlink(missing_ok=True)
        state["job"] = None
        gc.collect()
        mm.soft_empty_cache()


if __name__ == "__main__":
    main_loop()
