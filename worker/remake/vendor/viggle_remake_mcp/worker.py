# SPDX-License-Identifier: Apache-2.0
"""Viggle Remake render worker: one resident process per GPU, rendering jobs from a file queue.

    viggle-remake-worker --gpu 0 [--home ~/.viggle-remake] [--base MiniMaxAI/MiniMax-H3] [--lora-repo Viggle/Viggle-Remake]

The base transformer, the video and audio VAEs and both LoRAs (swap, edit) load once; only the active mode's LoRA sits
on the card. The 64-layer Qwen3-VL conditioner stays in host RAM and its decoder layers are streamed onto the card for
each new prompt; embeddings are cached on disk. Queue protocol and memory table: WORKER_NOTES.md.
"""
import os
import sys

if "--gpu" in sys.argv[:-1]:  # before torch: CUDA reads this once, and some imports below already initialise it
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import cpu_offload  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration  # noqa: E402
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, attention_backend  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.encoders import (MiniMaxH3Ref2VATextEncoderStep,  # noqa: E402
                                                             encode_vae_condition, get_qwen3vl_prompt_embeds)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (audio_latent_num_frames,  # noqa: E402
                                                                     video_latent_num_frames)
from diffusers.utils.export_utils import encode_video  # noqa: E402

from . import worker_patches as P  # noqa: E402

MODES = ("swap", "edit")
FRAMES = range(90, 346, 17)  # 17n + 5, 24 fps
AREA = {480: 832 * 480, 736: 1280 * 736}  # the canvases LAW was measured on; a job's cond may have any aspect within the area
# peak reserved GiB <= A + B * num_frames: the upper envelope of a sweep at 90..345 frames, 6 pinned refs, measured
# on B200; each ref fewer saves ~0.2 (480) / ~0.3 (736) GiB. MARGIN covers fragmentation when the card is full
LAW = {480: (80.8, 0.029), 736: (79.6, 0.088)}
MARGIN = 3.0
BASE_FILES = ["modular_model_index.json", "transformer/*", "text_encoder/*", "tokenizer/*", "processor/*", "vae/*",
              "audio_vae/*", "scheduler/*", "audio_scheduler/*"]


def local(repo, patterns):
    return repo if Path(repo).is_dir() else snapshot_download(repo, allow_patterns=patterns)


def max_frames(cap_gib):
    return {se: max([nf for nf in FRAMES if a + b * nf + MARGIN <= cap_gib], default=0) for se, (a, b) in LAW.items()}


def token_ids(processor, tokenizer, prompt, nf, nref):
    # the presentation is text only: a label per 2 fps block of the source, one per ref, then the prompt verbatim
    _, stamps = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
        np.zeros((nf, 1, 1, 3), dtype=np.uint8), 24.0, 2.0, processor.video_processor.temporal_patch_size)
    segments = ["<Video 1>: ", *[f"<{s:.1f} seconds>" for s in stamps], *[f"<Picture {i + 1}>: " for i in range(nref)],
                prompt]
    return [i for s in segments for i in tokenizer(s, add_special_tokens=False)["input_ids"]], "".join(segments)


class TextEncoder:
    """Qwen3-VL in host RAM; H3 reads hidden_states[50], so decoder layers 51-63 are dropped and the rest are copied
    onto the card one module at a time by accelerate's offload hooks."""

    def __init__(self, base):
        self.processor = AutoProcessor.from_pretrained(base, subfolder="processor")
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(base, subfolder="text_encoder", dtype=torch.bfloat16)
        self.model.model.language_model.layers = self.model.model.language_model.layers[:51]
        cpu_offload(self.model, execution_device=torch.device("cuda"))

    @torch.no_grad()
    def encode(self, ids):
        return get_qwen3vl_prompt_embeds(self.model, self.processor, ids, device=torch.device("cuda"),
                                         dtype=torch.bfloat16).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, help="card index; default: CUDA_VISIBLE_DEVICES as set")
    parser.add_argument("--home", default=os.environ.get("REMAKE_HOME", "~/.viggle-remake"))
    parser.add_argument("--base", default=os.environ.get("REMAKE_BASE", "MiniMaxAI/MiniMax-H3"))
    parser.add_argument("--lora-repo", default=os.environ.get("REMAKE_LORA_REPO", "Viggle/Viggle-Remake"))
    parser.add_argument("--mem-gib", type=float, default=None, help="use at most this much of the card")
    args = parser.parse_args()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    host = socket.gethostname()
    home = Path(args.home).expanduser()
    queue, embeds, workers = home / "queue", home / "embeds", home / "workers"
    for p in (queue, embeds, workers):
        p.mkdir(parents=True, exist_ok=True)

    free, total = torch.cuda.mem_get_info()
    cap = free / 2**30
    if args.mem_gib:
        torch.cuda.set_per_process_memory_fraction(args.mem_gib * 2**30 / total)
        cap = min(cap, args.mem_gib)
    limit = max_frames(cap)
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

    threading.Thread(target=beat_loop, daemon=True).start()
    print(f"{host}_{gpu}: {cap:.1f} GiB for the worker, max_frames {limit}", flush=True)

    t0 = time.time()
    base = local(args.base, BASE_FILES)
    lora = local(args.lora_repo, [f"lora/{m}/*" for m in MODES])
    pipe = ModularPipeline.from_pretrained(base, workflow="ref2va")
    type(pipe).min_duration = property(lambda self: 90 / 24)  # the adapters run from 90 frames; the base floor is 5 s
    pipe.load_components(names=["vae", "audio_vae", "scheduler", "audio_scheduler"], pretrained_model_name_or_path=base,
                         dtype=torch.bfloat16)
    # the LoRAs sit on the base `transformer`, which goes in the ref2va slot; prefix=None: the keys are bare module names
    pipe.transformer_ref = MiniMaxH3Transformer3DModel.from_pretrained(base, subfolder="transformer",
                                                                        torch_dtype=torch.bfloat16)
    for m in MODES:
        pipe.transformer_ref.load_lora_adapter(f"{lora}/lora/{m}", weight_name="pytorch_lora_weights.safetensors",
                                               prefix=None, adapter_name=m)
    pipe.transformer_ref.set_attention_backend("_native_cudnn")  # process-wide: the audio encode opts out
    pipe.to("cuda")
    P.apply(pipe)
    lora_layers = [m for m in pipe.transformer_ref.modules() if hasattr(m, "lora_A")]
    mode = [None]

    def set_mode(m):
        for layer in lora_layers:
            for name in MODES:
                dev = "cuda" if name == m else "cpu"
                layer.lora_A[name].to(dev)
                layer.lora_B[name].to(dev)
        torch.cuda.empty_cache()
        pipe.transformer_ref.set_adapters(m)
        mode[0] = m

    set_mode("swap")
    te = TextEncoder(base)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    def embed(prompt, nf, nref):
        ids, presentation = token_ids(te.processor, te.tokenizer, prompt, nf, nref)
        path = embeds / f"{hashlib.sha1(json.dumps([args.base, ids]).encode()).hexdigest()[:20]}.pt"
        if not path.exists():
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(te.encode(ids), tmp)
            tmp.rename(path)
        return torch.load(path), presentation

    def render(job):
        m, nf, tier, refs, seed = job["mode"], job["num_frames"], job["tier"], job["refs"], job["seed"]
        assert m in MODES, f"mode must be one of {MODES}, got {m!r}"
        assert nf in FRAMES, f"num_frames must be 17n+5 in 90..345, got {nf}"
        assert tier in limit, f"tier must be one of {list(limit)}, got {tier}"
        if nf > limit[tier]:
            raise ValueError(f"num_frames {nf} exceeds this worker's max {limit[tier]} at tier {tier}")
        pin = job.get("pin") or []
        assert len(pin) in (0, len(refs)) and all(0 <= k < nf for k in pin), (pin, len(refs), nf)
        if mode[0] != m:
            set_mode(m)
        assert pipe.transformer_ref.active_adapters() == [m], pipe.transformer_ref.active_adapters()
        P.J.update(mode=m, pin=list(pin), keyframe_until=job.get("keyframe_until", 0.9), np=0)
        P.J["embeds"], presentation = embed(job["prompt"], nf, len(refs))

        video = MiniMaxH3VideoReference.from_file(job["cond"])
        video.audio, video.sample_rate = None, None  # the soundtrack goes in as the target audio pin, not reference audio
        height, width = video.frames.shape[1], video.frames.shape[2]
        assert height % 32 == 0 and width % 32 == 0 and height * width <= AREA[tier], \
            f"cond must be multiples of 32 with at most {AREA[tier]} pixels at tier {tier}, got {width}x{height}"
        se = min(height, width)  # with the cond's own area as the budget the pipeline keeps its exact size
        pipe.register_to_config(canvas_short_edge=se, canvas_max_pixels=height * width, reference_image_short_edge=se)
        r, (_, ph, pw) = pipe.vae_spatial_compression_ratio, pipe.patch_size
        P.J["nkf"] = len(pin) * (height // r // ph) * (width // r // pw)
        # keyframes onto the canvas, stretched: like the cond they are whole frames of the source, at another size
        P.J["kf_images"] = []
        for im in (Image.open(x).convert("RGB") for x in (refs if pin else [])):
            P.J["kf_images"].append(im if im.size == (width, height) else
                                    pipe.image_processor.resize(im, height=height, width=width))

        # prefix: frames [at, at + k) of the previous take are this render's first k frames, a near-clean latent
        latents = None
        if job.get("prefix"):
            k = job.get("prefix_frames", 17)
            assert k % 17 == 0 and k < nf, k
            fr = MiniMaxH3VideoReference.from_file(job["prefix"]).frames
            at = job.get("prefix_at", len(fr) - k)
            px = torch.as_tensor(fr[at:at + k])
            assert tuple(px.shape) == (k, height, width, 3), (px.shape, k, height, width)
            # the VAE encodes 17-frame clips and drops the last 3 latents: pad to 17m + 5, keep the first 5m
            px = torch.cat([px, px[-1:].repeat(5, 1, 1, 1)])
            with torch.no_grad():
                clean = encode_vae_condition(pipe.vae, px.permute(3, 0, 1, 2)[None].to("cuda"), pipe.pixel_mean,
                                             pipe.pixel_std, pipe.keyframe_encode_seed)[:, :, :k // 17 * 5]
            nlat = video_latent_num_frames(nf, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)
            latents = torch.randn((1, clean.shape[1], nlat, *clean.shape[3:]),
                                  generator=torch.Generator().manual_seed(seed + 1000))
            aug = pipe.keyframe_noise_aug
            latents[:, :, :clean.shape[2]] = aug * clean + (1 - aug) * latents[:, :, :clean.shape[2]]
            P.J["prefix_rows"] = patchify_video_latents(clean, pipe.patch_size)
            P.J["np"] = P.J["prefix_rows"].shape[0]

        # the soundtrack as clean target audio latents
        asr = pipe.audio_vae.config.sampling_rate
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", job["audio"], "-t", f"{nf / 24:.3f}", "-vn",
                              "-f", "f32le", "-ac", "2", "-ar", str(asr), "-"], capture_output=True, check=True).stdout
        wav = torch.frombuffer(bytearray(pcm), dtype=torch.float32).view(-1, 2).T
        n_samp = round((nf + 1) / 24 * asr)
        wav = torch.nn.functional.pad(wav, (0, max(0, n_samp - wav.shape[1])))[:, :n_samp]
        with attention_backend(), torch.no_grad():
            posterior = pipe.audio_vae.encode(wav[:, None].to("cuda", torch.bfloat16), return_dict=False)[0]
        mean = torch.tensor(pipe.audio_vae.config.latents_mean).view(1, 1, -1)
        std = torch.tensor(pipe.audio_vae.config.latents_std).view(1, 1, -1)
        rows = (posterior.mode().float().cpu().transpose(1, 2)[:, :audio_latent_num_frames(nf)] - mean) / std

        result = pipe(prompt=presentation, references=[video, *[MiniMaxH3ImageReference.from_file(x) for x in refs]],
                      num_frames=nf, height=height, width=width, num_inference_steps=len(P.SIGMAS[m]) - 1,
                      generator=torch.Generator().manual_seed(seed), audio_latents=rows.permute(0, 2, 1).contiguous(),
                      output=["videos", "audio", "sampling_rate"], **({"latents": latents} if latents is not None else {}))
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.mp4")
        encode_video(result["videos"][0], fps=24, output_path=str(tmp), audio=result["audio"][0],
                     audio_sample_rate=result["sampling_rate"])
        tmp.rename(out)

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
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            render(job)
            info = {"out": job["out"], "secs": round(time.time() - t, 1),
                    "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 1), "host": host, "gpu": gpu,
                    "mode": job["mode"], "sigmas": P.SIGMAS[job["mode"]]}
            tmp = run.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(info))
            tmp.rename(run.with_suffix(".done"))
            print(f"{run.stem}: {info}", flush=True)
        except Exception:
            tb = traceback.format_exc()
            run.with_suffix(".err").write_text(tb)
            print(f"{run.stem}: FAILED\n{tb}", flush=True)
            if "CUDA error" in tb:  # the context is gone; every later job would fail
                run.unlink(missing_ok=True)
                os._exit(1)
        run.unlink(missing_ok=True)
        state["job"] = None
        P.J.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
