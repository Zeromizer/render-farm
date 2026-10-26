"""Build API-format graphs reproducing u/cgpixel23's "4K H3 on 6 GB" workflow (SelfLift + RH DLSS5).
Quality settings are copied from his workflow JSON. His pure low-VRAM workarounds (Q2_K text encoder, int8 VAE,
SLA/Sol sparse attention, chunked feed-forward, highres tiling) are left out on the 16 GB 4080.
Writes spikes/selflift/jobs.json: [{label, graph}]"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
UNET = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors"
LORA = "h3_antiburn_universal_v1.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VVAE, AVAE = "minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"
UPSCALER = os.environ.get("H3_UPSCALER", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")
FRAMES, SEED = 124, 999


def graph(prompt, w, h, variant, prefix):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["shift", 0], "lora_name": LORA, "strength_model": 0.5}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VVAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                                                                 "width": w, "height": h, "length": FRAMES}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["lora", 0], "scheduler": "beta", "steps": 8, "denoise": 1.0}},
    }
    if variant == "baseline":
        g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["cond", 0]}}
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
        g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                       "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["cond", 1]}}
    else:
        g["sample"] = {"class_type": "SelfLiftH3Sampler", "inputs": {
            "model": ["lora", 0], "positive": ["cond", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "latent_image": ["cond", 1], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "seed": SEED, "cfg": 1.0,
            "transition_step": 6, "lowres_scale": 0.5, "rho": 0.0, "w_min": 0.5, "w_max": 1.0,
            "upscaler_model": UPSCALER, "highres_tiling": False, "upscaler_unload": True}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 24.0, "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix + "_raw", "format": "mp4"}}
    if variant == "selflift_dlss":
        g["dlss"] = {"class_type": "RH_DLSS5Enhance", "inputs": {
            "image": ["dec", 0], "upscaling_mode": "2K", "style": "cinematic", "preset": 0, "intensity": 1.0, "tone": 1.0,
            "structure": 1.5, "skin": 1.0, "auto_mask": "off", "motion": "auto", "scene_change_threshold": 0.24,
            "batch_mode": "temporal sequence", "warmup_frames": 1, "keep_audio": "on", "backend": "auto", "channel_order": "auto",
            "hdr": "off", "global_tone": -1.0, "detail": 1.0, "color": 1.0, "runtime_dir": "", "wine_prefix": ""}}
        g["video2"] = {"class_type": "CreateVideo", "inputs": {"images": ["dlss", 0], "fps": 24.0, "audio": ["adec", 0]}}
        g["save2"] = {"class_type": "SaveVideo", "inputs": {"video": ["video2", 0], "filename_prefix": prefix + "_dlss2k", "format": "mp4"}}
    return g


def read(name):
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return f.read()


# His widget is 16:9 at 0.4 MP; his post says 0.6 MP. 0.6 MP on the 32 grid: 1024x576 / 576x1024.
PROMPTS = {"pancake": (read("pancake_prompt.txt"), 1024, 576), "car": (read("car_prompt.txt"), 576, 1024)}
jobs = []
for pk, (pr, w, h) in PROMPTS.items():
    for v in ("selflift_dlss", "baseline"):   # selflift_dlss also saves the SelfLift-only raw
        jobs.append({"label": f"{pk}_{v}", "graph": graph(pr, w, h, v, f"sl_{pk}_{v}")})
with open(os.path.join(D, "jobs.json"), "w", encoding="utf-8") as f:
    json.dump(jobs, f, indent=1, ensure_ascii=False)
print(len(jobs), [j["label"] for j in jobs])
