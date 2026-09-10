"""API-format ComfyUI graphs for the MiniMax H3 *latent* path: packet capture at
generation time and the F07 latent-space upscale (einhorn13/mmh3_media +
LBH-123-AI's MinimaxH3LatentUpscaler3D).

Why a second module: graphs.build() is the tuned, validated generation graph
and must stay byte-for-byte unchanged (test_h3_latent_graphs locks it). This
module reuses its constants and its model chain, swapping only the conditioning
node for MMH3H3AutoCondition and adding MMH3PackH3Result + MMH3Save, so the
`.mmh3` packet records exactly the sampling that produced the clip.

Reference workflows (UI format, vendored under videogen/recipes/reference/):
  F01 mmh3_f01_fl2va.json            -> build_generation_with_packet
  F07 mmh3_f07_latent_upscale.json   -> build_latent_upscale(variant="full")
  F07 mmh3_f07_native_tile_upscale   -> build_latent_upscale(variant="tile")
  (no reference)                     -> build_decoded_upscale: MMH3H3DecodedUpscalePrepare
                                        for clips that only exist as an mp4

Node input names come from the mmh3_media sources (nodes_packet.py,
nodes_conditioning.py, nodes_settings.py, nodes_upscale.py, nodes_h3.py,
nodes_optimization.py) and the upscaler README. DynamicCombo inputs use the
dotted "<input>.<sub>" form exactly like graphs.build_upscale does for
ResizeImageMaskNode. Option strings that the sources do not spell out are
module constants (GEOMETRY_MODE_DIMS, PUT_ROLE, ...) so the /object_info
preflight (h3_preflight.py) can name the right one on the PC in one edit.

Design notes
  * Refine settings come from the packet: MMH3H3UpscaleRefineSampling reuses
    the recorded sampling profile (sampler, scheduler, steps, shifts) and picks
    denoise 0.375 for a "turbo (8 steps)" profile, 0.5 for 4 steps, 0.25
    otherwise, unless denoise is given (clamped 0.05-0.5 by the node).
  * The worker writes that profile itself (sampling_profile_json /
    applied_loras_json) instead of using MMH3H3SamplingPreset, whose presets
    use euler + shift 6/3 + LoRA paths this install does not have.
  * H3 wants 32-aligned canvases; 1080 is not. The refine runs at the
    32-aligned cover size (target_dims()["refine"]) and the runner crops to
    the exact request afterwards (crop_args()).
"""
import json
import math
import os

from videogen import graphs

MMH3_REPO = "https://github.com/einhorn13/mmh3_media"
MMH3_NODE_DIR = "ComfyUI_mmh3_media"
UPSCALER_REPO = "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler"
UPSCALER_NODE_DIR = "Comfyui_Minimax_h3_latent_Upscaler"
UPSCALER_WEIGHTS_URL = "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"
UPSCALER_WEIGHT = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
UPSCALER_FOLDER = "latent_upscale_models"
UPSCALER_CLASS = "MinimaxH3LatentUpscaler3D"

ALIGN = 32
VARIANTS = ("tile", "full", "decoded")
METHOD = "h3_latent_upscale"
# Provisional pixel-frame guard for the full-frame refine on a 16 GB card:
# 1088x1920 x 73 frames (3 s). The PC measurement replaces this.
FULL_MAX_PXF = 1088 * 1920 * 73
# PC 2026-09-10: the F07 reference default (overlap_mode context_only + blend hard) left visible
# hard seams on the 480p -> 1080p car clip (a vertical cut through the hood at the column boundary,
# horizontal bands at the fenders); reprocess + half_cosine re-samples the 64 px overlaps and blends
# them, the seams vanish, and the cost and VRAM are the same. That is the worker default now; the
# F07 pair is still selectable (upscale.overlap_mode / blend_mode).
TILE_DEFAULTS = {"tile_width": 640, "tile_height": 384, "tile_overlap": 64, "context_padding": 64,
                 "traversal": "snake", "overlap_mode": "reprocess", "blend_mode": "half_cosine",
                 "context_source": "composited"}
# MMH3H3NativeTileRefine combos (bca81b8c spatial_tiles.py). Its own rules: context_only overlap
# requires blend hard (exclusive ownership); reprocess overlap requires linear or half_cosine and
# an overlap > 0.
TILE_MODE_OPTIONS = {"traversal": ("row_major", "snake"), "overlap_mode": ("context_only", "reprocess"),
                     "blend_mode": ("hard", "linear", "half_cosine"), "context_source": ("original", "composited")}
# Anchored on the RTX 4080 SUPER 2026-09-10 (lanczos-resized source as the reference): lanczos
# 0.996 / 52.8 dB, lanczos blurred (sigma 1.5) 0.9925 / 47.3, raw SeedVR2 0.956 / 35.7 (bottom
# row of cells 0.83-0.94), tile refine 0.922 / 25.2 (min 0.909 / 23.4), full refine 0.919 / 24.9.
# A latent refine legitimately moves far below raw SeedVR2 on these metrics, so the frame floors sit
# under the measured refines and only a broken refine (wrong packet, blank tiles) trips them. The
# car tile's four centre cells ran 0.78-0.83 against 0.9997 background cells (drop 0.17-0.22) with a
# visibly re-imagined badge, so cell_drop_max 0.25 flags only a worse case; the -compare.mp4 is the
# real badge/plate review. Warns, never gates.
FIDELITY_DEFAULTS = {"enabled": True, "compare": True, "ssim_min": 0.85, "psnr_min": 22.0,
                     "cell_ssim_min": 0.70, "cell_drop_max": 0.25}
ATTENTION_DEFAULT = "Default"
FP16_ACCUMULATION_DEFAULT = "Default"
FP16_ACCUMULATION_OPTIONS = ("Default", "Enabled", "Disabled")

# Option strings to confirm against /object_info on the PC (see h3_preflight).
GEOMETRY_MODE_DIMS = "target_dimensions"        # MMH3H3LatentUpscalePrepare.geometry_mode
UPSCALER_MODE_DIMS = "target dimensions"        # MinimaxH3LatentUpscaler3D.mode
CREATE_TASK_FRAMES = "Video (optional frames)"  # MMH3Create.task
SETTINGS_RESOLUTION_CUSTOM = "Custom"           # MMH3H3GenerationSettings.resolution
PUT_ROLE = "auxiliary"                          # MMH3Put.role for the decoded import
LORAS_ACTION_NONE = "mark no LoRAs"             # MMH3GenerationLoRAs.action: record an explicit empty list
LOAD_FILE_NONE = "(none)"                       # MMH3Load.file when path_override is used
LATENT_ORIGIN_SAMPLER = "sampler_output"
LATENT_ORIGIN_DERIVED = "derived"
OP_GENERATE, OP_REFINE, OP_TILE = "generate", "latent_upscale_refine", "latent_upscale_native_tile_refine"
MODE_HIGH_SIGMA = "high_sigma_refine"
DECODED_PROMPT = ("The same footage at higher resolution: identical subject, framing, motion, colours "
                  "and sound, with finer detail.")
PROBE_CLASS = "PreviewAny"   # optional output node used to read strings back from /history


# --------------------------------------------------------------------------- provenance strings

def sampling_profile_json(fam, steps, turbo, shift_video=12.0, shift_audio=3.0, sampler="res_multistep",
                          scheduler="simple"):
    """The mmh3 v2 sampling contract for what graphs.build actually ran, so the
    refine tail follows the real trajectory. profile text matters: the refine
    node keys its denoise choice on "turbo (4" / "turbo (8"."""
    profile = f"turbo ({int(steps)} steps)" if turbo else f"standard ({int(steps)} steps)"
    return json.dumps({"version": 2, "contract": "mmh3_h3_sampling_preset_v2", "profile": profile,
                       "task_family": fam, "steps": int(steps), "video_shift": float(shift_video),
                       "audio_shift": float(shift_audio), "sampler": sampler, "scheduler": scheduler,
                       "sigma_preset": "scheduler_generated", "runtime_validated": True,
                       "source": "render-farm video_gen"})


def applied_loras_json(fam, steps, turbo):
    if not turbo:
        return "[]"
    name = graphs.TURBO_LORAS[fam].get(int(steps)) or graphs.TURBO_LORAS[fam][graphs.DEFAULT_TURBO_STEPS[fam]]
    return json.dumps([{"name": name, "strength_model": 1.0, "purpose": "acceleration",
                        "reapply_for_high_sigma": True, "loader": "LoraLoaderModelOnly",
                        "source": "render-farm video_gen"}])


def turbo_lora_name(fam="fl2va", steps=None):
    return graphs.TURBO_LORAS[fam].get(int(steps or graphs.DEFAULT_TURBO_STEPS[fam])) \
        or graphs.TURBO_LORAS[fam][graphs.DEFAULT_TURBO_STEPS[fam]]


# --------------------------------------------------------------------------- geometry

def _even(v):
    return int(v) // 2 * 2


def _up32(v):
    return int(math.ceil(int(v) / ALIGN)) * ALIGN


def requested_dims(src_w, src_h, u):
    """The exact output size the operator asked for: the same numbers
    segments.lanczos() would produce (short edge -> shorter_size, or x factor,
    both even)."""
    src_w, src_h = int(src_w), int(src_h)
    if u.get("shorter_size"):
        s = int(u["shorter_size"])
        if src_w >= src_h:
            return _even(round(src_w * s / src_h)), s
        return s, _even(round(src_h * s / src_w))
    f = float(u.get("factor") or 2.0)
    return _even(src_w * f), _even(src_h * f)


def target_dims(src_w, src_h, u):
    """{req: (w, h), refine: (w32, h32), scale: (sx, sy)} for one upscale.

    refine covers req (never smaller) on the 32 grid H3 samples on; the runner
    crops refine -> req. Scale must stay within the upscaler's 1x-4x per axis."""
    req = requested_dims(src_w, src_h, u)
    refine = (_up32(req[0]), _up32(req[1]))
    scale = (refine[0] / int(src_w), refine[1] / int(src_h))
    for axis, s in zip("wh", scale):
        if s < 1.0 - 1e-9:
            raise ValueError(f"h3_latent_upscale cannot downscale (axis {axis}: {int(src_w if axis == 'w' else src_h)} -> "
                             f"{refine[0] if axis == 'w' else refine[1]}); use lanczos for a smaller output")
        if s > 4.0 + 1e-9:
            raise ValueError(f"h3_latent_upscale is limited to 4x per axis (axis {axis} would be {s:.2f}x); "
                             f"upscale in two passes or lower shorter_size/factor")
    return {"req": req, "refine": refine, "scale": (round(scale[0], 4), round(scale[1], 4))}


def crop_args(refine, req):
    """ffmpeg crop filter string for the centre crop refine -> req, or None."""
    if tuple(refine) == tuple(req):
        return None
    x = (refine[0] - req[0]) // 2
    y = (refine[1] - req[1]) // 2
    return f"crop={req[0]}:{req[1]}:{x}:{y}"


def on_frame_grid(frames):
    return int(frames) >= 5 and (int(frames) - 5) % 17 == 0


def grid_frames(frames):
    """Largest 17k+5 count not above frames."""
    n = int(frames)
    return max(5, 5 + 17 * ((n - 5) // 17)) if n >= 5 else 0


# AV-exact H3 lengths: 17k+5 frames whose audio latent (40/s) lands on a whole frame at 24 fps,
# i.e. frames divisible by 3: 39, 90, 141, 192, 243, ... (mmh3_media continuation.py
# is_exact_h3_av_handover_boundary). PC 2026-09-10: MMH3H3DecodedUpscalePrepare (bca81b8c)
# slices the audio to frames*32000//24 samples, whose VAE encode floors to 800-sample latent
# frames, while its validator rounds (73 f -> 121 vs 122): every other grid length fails with
# "H3 AV duration mismatch". The decoded import therefore keeps the largest AV-exact length.
AV_BOUNDARY_MIN = 39
AV_BOUNDARY_STEP = 51


def av_boundary_frames(frames):
    """Largest AV-exact length (39 + 51k) not above frames; 0 when the clip is shorter than 39."""
    n = int(frames)
    if n < AV_BOUNDARY_MIN:
        return 0
    return AV_BOUNDARY_MIN + AV_BOUNDARY_STEP * ((n - AV_BOUNDARY_MIN) // AV_BOUNDARY_STEP)


# --------------------------------------------------------------------------- params

def validate_upscale_params(u, mode, src_dims=None, frames=None):
    """Normalise params.video_gen.upscale for method h3_latent_upscale.

    Raises ValueError with an operator-readable message. Returns a new dict
    with every knob filled in. mode is the job mode (upscale | t2v | i2v | r2v)."""
    u = dict(u or {})
    if (u.get("method") or "").lower() != METHOD:
        raise ValueError(f"validate_upscale_params: method must be {METHOD}, got {u.get('method')!r}")
    variant = (u.get("variant") or "tile").lower()
    if variant not in VARIANTS:
        raise ValueError(f"upscale.variant must be one of {VARIANTS}, got {variant!r}")
    if mode == "r2v" and variant != "decoded":
        raise ValueError("h3_latent_upscale after r2v generation is not supported yet (save_latent for r2v is "
                         "unverified); run it as a standalone upscale with variant 'decoded'")
    if mode == "upscale" and variant in ("tile", "full"):
        lat = u.get("latent")
        if not isinstance(lat, dict) or not lat.get("bucket") or not lat.get("path"):
            raise ValueError("h3_latent_upscale needs upscale.latent {bucket, path}: the outputs/<id>-latent.mmh3 "
                             "sidecar of a save_latent job (or the same packet filed as an asset). This clip has "
                             "no saved latent; use variant 'decoded' (re-encodes the mp4, lower fidelity) or lanczos")
    if u.get("factor") and u.get("shorter_size"):
        raise ValueError("upscale: choose factor or shorter_size, not both")
    if u.get("factor") is not None:
        f = float(u["factor"])
        if not 1.0 < f <= 4.0:
            raise ValueError(f"upscale.factor must be in (1, 4], got {f}")
    if u.get("shorter_size") is not None:
        s = int(u["shorter_size"])
        if not 480 <= s <= 2160:
            raise ValueError(f"upscale.shorter_size must be 480-2160, got {s}")
    if not u.get("factor") and not u.get("shorter_size"):
        u["shorter_size"] = 1080
    denoise = float(u.get("denoise") or 0.0)
    if denoise != 0.0 and not 0.05 <= denoise <= 0.5:
        raise ValueError(f"upscale.denoise must be 0 (source-aware) or 0.05-0.50, got {denoise}")
    steps = int(u.get("steps_override") or 0)
    if steps and not 1 <= steps <= 20:
        raise ValueError(f"upscale.steps_override must be 0 (source profile) or 1-20, got {steps}")
    if variant == "decoded":
        # An imported mp4 carries no sampling provenance; give the refine the
        # turbo-8 trajectory graphs.build uses, at the same 3-step tail.
        if not steps:
            steps = graphs.DEFAULT_TURBO_STEPS["fl2va"]
        if denoise == 0.0:
            denoise = 0.375
    for key in ("tile_width", "tile_height", "tile_overlap", "context_padding"):
        v = int(u.get(key) if u.get(key) is not None else TILE_DEFAULTS[key])
        if v % ALIGN or v < 0 or (key.startswith("tile_w") or key.startswith("tile_h")) and v < ALIGN:
            raise ValueError(f"upscale.{key} must be a positive multiple of {ALIGN}, got {v}")
        u[key] = v
    for key, options in TILE_MODE_OPTIONS.items():
        v = u.get(key) or TILE_DEFAULTS[key]
        if v not in options:
            raise ValueError(f"upscale.{key} must be one of {options}, got {v!r}")
        u[key] = v
    if u["overlap_mode"] == "context_only" and u["blend_mode"] != "hard":
        raise ValueError("upscale.blend_mode 'linear'/'half_cosine' needs overlap_mode 'reprocess' (context_only "
                         "overlap gives each tile exclusive ownership, so there is nothing to blend)")
    if u["overlap_mode"] == "reprocess" and (u["blend_mode"] == "hard" or u["tile_overlap"] <= 0):
        raise ValueError("upscale.overlap_mode 'reprocess' needs blend_mode 'linear' or 'half_cosine' and a "
                         "tile_overlap > 0")
    fp16 = u.get("fp16_accumulation") or FP16_ACCUMULATION_DEFAULT
    if fp16 not in FP16_ACCUMULATION_OPTIONS:
        raise ValueError(f"upscale.fp16_accumulation must be one of {FP16_ACCUMULATION_OPTIONS}, got {fp16!r}")
    attention = u.get("attention") or ATTENTION_DEFAULT
    if attention != ATTENTION_DEFAULT:
        raise ValueError("upscale.attention: only 'Default' is allowed (no SLA/VDN experimental backends)")
    fid = dict(FIDELITY_DEFAULTS)
    fid.update({k: v for k, v in (u.get("fidelity") or {}).items() if k in FIDELITY_DEFAULTS})
    out = dict(u)
    out.update({"method": METHOD, "variant": variant, "denoise": denoise, "steps_override": steps,
                "seed": int(u.get("seed") if u.get("seed") is not None else 0),
                "force_unload": u.get("force_unload", True) is not False,
                "attention": attention, "fp16_accumulation": fp16,
                "missing_audio_policy": u.get("missing_audio_policy") or "error", "fidelity": fid,
                "allow_large_full": bool(u.get("allow_large_full"))})
    if src_dims is not None:
        dims = target_dims(src_dims[0], src_dims[1], out)
        out["_dims"] = dims
        if frames is not None and variant == "full":
            pxf = dims["refine"][0] * dims["refine"][1] * int(frames)
            if pxf > FULL_MAX_PXF and not out["allow_large_full"]:
                raise ValueError(f"variant 'full' refines the whole {dims['refine'][0]}x{dims['refine'][1]} clip "
                                 f"in one pass: {int(frames)} frames = {pxf / 1e6:.0f} M pixel-frames, above the "
                                 f"{FULL_MAX_PXF / 1e6:.0f} M guard for a 16 GB card. Use variant 'tile' (default) "
                                 f"or a clip of at most {FULL_MAX_PXF // (dims['refine'][0] * dims['refine'][1])} "
                                 f"frames, or set allow_large_full after measuring")
    if variant != "decoded" and frames is not None and not on_frame_grid(frames):
        raise ValueError(f"source clip has {int(frames)} frames, which is not on H3's 17k+5 grid; the packet and "
                         f"the clip must come from the same generation")
    return out


# --------------------------------------------------------------------------- graph builders

def _loaders(g, fam="fl2va", lora=None):
    g["unet"] = {"class_type": "UNETLoader",
                 "inputs": {"unet_name": graphs.CHECKPOINTS[fam], "weight_dtype": "default"}}
    model_ref = ["unet", 0]
    if lora:
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model_ref, "lora_name": lora, "strength_model": 1.0}}
        model_ref = ["lora", 0]
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": graphs.TEXT_ENCODER, "type": "minimax", "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": graphs.VIDEO_VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": graphs.AUDIO_VAE}}
    return model_ref


def _av_decode(g, latent_ref, filename_prefix):
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": latent_ref, "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": latent_ref, "vae": ["avae", 0]}}
    g["video"] = {"class_type": "CreateVideo",
                  "inputs": {"images": ["dec", 0], "fps": float(graphs.FPS), "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": filename_prefix, "format": "mp4"}}


def build_generation_with_packet(mode, p, inputs, filename_prefix, packet_prefix):
    """graphs.build() plus the .mmh3 packet: same models, same sampler chain,
    same seed; MMH3Create -> MMH3H3GenerationSettings -> MMH3H3AutoCondition
    replaces MiniMaxH3ImageToVideo (AutoCondition wraps it), and
    MMH3PackH3Result -> MMH3Save records latent + video + audio + settings.

    Returns (graph, meta); meta adds packet_prefix / sampling_profile / loras.
    t2v and i2v only (the r2v reference autogrow keys are unverified)."""
    if mode not in ("t2v", "i2v"):
        raise ValueError(f"build_generation_with_packet supports t2v/i2v, got {mode!r}")
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("video_gen needs a non-empty prompt (H3 requires a text item)")
    fam = "fl2va"
    width, height = graphs.dims(p.get("resolution") or "480p", p.get("ratio") or "16:9")
    length = graphs.frames_for(p.get("duration_s", 5))
    turbo = p.get("turbo", True) is not False
    seed = int(p.get("seed") if p.get("seed") is not None else 0)
    if turbo:
        steps = int(p.get("steps") or graphs.DEFAULT_TURBO_STEPS[fam])
        lora = graphs.TURBO_LORAS[fam].get(steps) or graphs.TURBO_LORAS[fam][graphs.DEFAULT_TURBO_STEPS[fam]]
    else:
        steps = int(p.get("steps") or graphs.DEFAULT_FULL_STEPS)
        lora = None
    shift_video = float(p.get("shift_video", 12.0))
    shift_audio = float(p.get("shift_audio", 3.0))

    g = {}
    model_ref = _loaders(g, fam, lora)
    if p.get("shift_video") is not None or p.get("shift_audio") is not None:
        g["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                      "inputs": {"model": model_ref, "shift_video": shift_video, "shift_audio": shift_audio}}
        model_ref = ["shift", 0]

    create = {"prompt": prompt, "task": CREATE_TASK_FRAMES, "seed": seed, "name": os.path.basename(packet_prefix),
              "notes": f"render-farm video_gen {mode}"}
    if mode == "i2v" and not inputs.get("first_frame") and not inputs.get("last_frame"):
        raise ValueError("i2v needs first_frame and/or last_frame")
    frame_links = {}
    for key in ("first_frame", "last_frame"):
        name = inputs.get(key)
        if name:
            g[f"load_{key}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            frame_links[key] = [f"load_{key}", 0]
    create.update(frame_links)
    g["pkt_create"] = {"class_type": "MMH3Create", "inputs": create}
    g["pkt_settings"] = {"class_type": "MMH3H3GenerationSettings",
                         "inputs": {"packet": ["pkt_create", 0], "resolution": SETTINGS_RESOLUTION_CUSTOM,
                                    "resolution.width": width, "resolution.height": height,
                                    "duration_seconds": round(length / graphs.FPS, 3)}}
    # Literal overrides: the sampling canvas must be exactly what graphs.build
    # would use, whatever GenerationSettings rounds duration_seconds to.
    g["cond"] = {"class_type": "MMH3H3AutoCondition",
                 "inputs": {"packet": ["pkt_settings", 0], "prompt_override": "", "seed_override": -1,
                            "clip": ["clip", 0], "video_vae": ["vae", 0], "audio_vae": ["avae", 0],
                            "width_override": width, "height_override": height, "frames_override": length}}
    g["opt"] = {"class_type": "MMH3H3ModelOptimizations",
                "inputs": {"model": model_ref, "attention": ATTENTION_DEFAULT,
                           "fp16_accumulation": FP16_ACCUMULATION_DEFAULT, "sampling_profile_json": ""}}
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["cond", 0]}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sched", 0], "latent_image": ["cond", 1]}}
    _av_decode(g, ["sample", 0], filename_prefix)
    profile = sampling_profile_json(fam, steps, turbo, shift_video, shift_audio)
    loras = applied_loras_json(fam, steps, turbo)
    g["pkt_result"] = {"class_type": "MMH3PackH3Result",
                       "inputs": {"packet": ["cond", 3], "latent": ["sample", 0], "video": ["video", 0],
                                  "audio": ["adec", 0], "operation": OP_GENERATE, "mode": ["cond", 4],
                                  "status": ["cond", 5], "process_info_json": ["cond", 6],
                                  "generation_settings_json": ["pkt_settings", 4],
                                  "sampling_profile_json": profile, "optimization_profile_json": ["opt", 1],
                                  "applied_loras_json": loras, "latent_origin": LATENT_ORIGIN_SAMPLER,
                                  **frame_links}}
    g["pkt_save"] = {"class_type": "MMH3Save",
                     "inputs": {"packet": ["pkt_result", 0], "filename_prefix": packet_prefix, "target": "output",
                                "overwrite": False}}
    meta = {"family": fam, "width": width, "height": height, "length": length,
            "seconds": round(length / graphs.FPS, 2), "steps": steps, "turbo": turbo, "seed": seed,
            "packet_prefix": packet_prefix, "sampling_profile": json.loads(profile), "loras": json.loads(loras)}
    return g, meta


def _refine_tail(g, u, prep_key, packet_ref, refine_dims, filename_prefix, packet_prefix, variant, probes):
    """Everything after the upscaler for tile/full/decoded: LoRA re-apply,
    conditioning, refine settings, sampling, decode, save, optional packet."""
    decoded = variant == "decoded"
    g["loras"] = {"class_type": "MMH3H3RefineLoRAs",
                  "inputs": {"packet": [prep_key, 0], "model": ["unet", 0], "clip": ["clip", 0],
                             "unknown_policy": "continue_without_source_loras" if decoded else "error",
                             "missing_policy": "error",
                             "turbo_override": turbo_lora_name("fl2va", u["steps_override"]) if decoded else "",
                             "acceleration_policy": "preserve"}}
    g["opt"] = {"class_type": "MMH3H3ModelOptimizations",
                "inputs": {"model": ["loras", 0], "attention": u["attention"],
                           "fp16_accumulation": u["fp16_accumulation"], "sampling_profile_json": ""}}
    g["cond"] = {"class_type": "MMH3H3AutoCondition",
                 "inputs": {"packet": [prep_key, 0], "prompt_override": "", "seed_override": int(u["seed"]),
                            "clip": ["loras", 1], "video_vae": ["vae", 0], "audio_vae": ["avae", 0],
                            "width_override": [prep_key, 3], "height_override": [prep_key, 4],
                            "frames_override": [prep_key, 5]}}
    g["refine"] = {"class_type": "MMH3H3UpscaleRefineSampling",
                   "inputs": {"packet": packet_ref, "denoise_override": float(u["denoise"])}}
    g["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["opt", 0], "shift_video": ["refine", 3], "shift_audio": ["refine", 4]}}
    if variant == "tile":
        # Per-tile sampling cannot take the packet's first/last-frame conditioning: the core model
        # asserts the frame rows against the full-frame latent (comfy/ldm/minimax/model.py:700,
        # "shape mismatch ... [2006, 96] ... [308, 96]" on the 2026-09-10 PC run). The tile guider
        # therefore gets text-only conditioning at the refine size, built by the core node from the
        # packet's own prompt; AutoCondition still supplies the seed and the packet downstream.
        g["inspect"] = {"class_type": "MMH3Inspect", "inputs": {"packet": [prep_key, 0]}}
        g["cond_text"] = {"class_type": "MiniMaxH3ImageToVideo",
                          "inputs": {"clip": ["loras", 1], "vae": ["vae", 0], "prompt": ["inspect", 11],
                                     "width": [prep_key, 3], "height": [prep_key, 4], "length": [prep_key, 5]}}
        guider_cond = ["cond_text", 0]
    else:
        guider_cond = ["cond", 0]
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": ["shift", 0], "conditioning": guider_cond}}
    g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": ["refine", 5]}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": ["shift", 0], "scheduler": ["refine", 6],
                             "steps": int(u["steps_override"]) if u["steps_override"] else ["refine", 2],
                             "denoise": ["refine", 7]}}
    if variant == "full":
        g["target"] = {"class_type": "MMH3H3LatentUpscaleTarget",
                       "inputs": {"video_latent": ["up", 0], "audio_latent": [prep_key, 2]}}
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": ["cond", 2]}}
        g["sample"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                  "sigmas": ["sched", 0], "latent_image": ["target", 0]}}
        g["avsep"] = {"class_type": "MMH3H3AVSeparate", "inputs": {"av_latent": ["sample", 0]}}
        g["avcomb"] = {"class_type": "MMH3H3AVCombine",
                       "inputs": {"video_latent": ["avsep", 0], "audio_latent": ["avsep", 1]}}
        out_latent = ["avcomb", 0]
        operation = OP_REFINE
    else:
        g["dec_up"] = {"class_type": "VAEDecode", "inputs": {"samples": ["up", 0], "vae": ["vae", 0]}}
        g["tile"] = {"class_type": "MMH3H3NativeTileRefine",
                     "inputs": {"video": ["dec_up", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                "sigmas": ["sched", 0], "video_vae": ["vae", 0],
                                "source_audio_latent": [prep_key, 2], "frames": [prep_key, 5],
                                "seed": ["cond", 2], "tile_width": u["tile_width"], "tile_height": u["tile_height"],
                                "overlap": u["tile_overlap"], "context_padding": u["context_padding"],
                                "traversal": u["traversal"], "overlap_mode": u["overlap_mode"],
                                "blend_mode": u["blend_mode"], "context_source": u["context_source"]}}
        # No `packet` link on purpose: with a packet attached MMH3H3NativeTileRefine (bca81b8c,
        # nodes_upscale.py:590) demands an F16 control configuration the packet does not carry and
        # fails "Packet has no F16 control configuration". The reference F07 tile workflow leaves it
        # unlinked too; the result packet is built from the AutoCondition packet instead.
        out_latent = ["tile", 0]
        operation = OP_TILE
    _av_decode(g, out_latent, filename_prefix)
    if probes:
        g["probe_refine"] = {"class_type": PROBE_CLASS, "inputs": {"source": ["refine", 0]}}
        g["probe_prep"] = {"class_type": PROBE_CLASS, "inputs": {"source": [prep_key, 8]}}
        if variant != "full":
            g["probe_tile"] = {"class_type": PROBE_CLASS, "inputs": {"source": ["tile", 2]}}
    if packet_prefix:
        report = {"geometry_report_json": [prep_key, 8], "source_lora_report_json": ["loras", 4],
                  "process_loras_json": ["loras", 5], "upscaler_model": UPSCALER_WEIGHT, "device": "cuda",
                  "precision": "bf16", "sigma_profile": "source_aware", "refine_sampling_json": ["refine", 0]}
        if variant != "full":
            report.update({"tile_plan_json": ["tile", 2], "tile_run_report_json": ["tile", 3],
                           "native_tile_adapter_json": ["tile", 4]})
        g["report"] = {"class_type": "MMH3H3LatentUpscaleReport", "inputs": report}
        g["pkt_result"] = {"class_type": "MMH3PackH3Result",
                           "inputs": {"packet": ["cond", 3], "latent": out_latent, "video": ["video", 0],
                                      "audio": ["adec", 0], "operation": operation, "mode": MODE_HIGH_SIGMA,
                                      "status": ["report", 2], "process_info_json": ["report", 0],
                                      "applied_loras_json": ["report", 1], "optimization_profile_json": ["opt", 1],
                                      "latent_origin": LATENT_ORIGIN_DERIVED}}
        g["pkt_save"] = {"class_type": "MMH3Save",
                         "inputs": {"packet": ["pkt_result", 0], "filename_prefix": packet_prefix,
                                    "target": "output", "overwrite": False}}
    return operation


def _upscaler(g, u, prep_key):
    g["up"] = {"class_type": UPSCALER_CLASS,
               "inputs": {"latent": [prep_key, 1], "model_name": UPSCALER_WEIGHT, "mode": UPSCALER_MODE_DIMS,
                          "mode.width": [prep_key, 3], "mode.height": [prep_key, 4], "align": ALIGN,
                          "enable_temporal_chunking": True, "force_unload": bool(u["force_unload"]),
                          "device": "cuda", "precision": "bf16"}}


def build_latent_upscale(u, latent_abs_path, src_dims, filename_prefix, packet_prefix=None, probes=True):
    """F07 latent upscale of a saved packet. u must come from
    validate_upscale_params (variant tile | full). Returns (graph, meta)."""
    variant = u["variant"]
    if variant not in ("tile", "full"):
        raise ValueError(f"build_latent_upscale handles tile|full, got {variant!r}")
    if not os.path.isabs(latent_abs_path):
        raise ValueError("MMH3Load.path_override needs an absolute path")
    dims = u.get("_dims") or target_dims(src_dims[0], src_dims[1], u)
    w32, h32 = dims["refine"]
    g = {}
    g["pkt_load"] = {"class_type": "MMH3Load",
                     "inputs": {"file": LOAD_FILE_NONE, "verify": "on_access", "path_override": latent_abs_path}}
    _loaders(g, "fl2va", None)
    g["prep"] = {"class_type": "MMH3H3LatentUpscalePrepare",
                 "inputs": {"packet": ["pkt_load", 0], "geometry_mode": GEOMETRY_MODE_DIMS, "scale": 2.0,
                            "target_width": w32, "target_height": h32, "target_megapixels": 0.0, "align": ALIGN,
                            "enable_chunking": True}}
    _upscaler(g, u, "prep")
    operation = _refine_tail(g, u, "prep", ["pkt_load", 0], dims["refine"], filename_prefix, packet_prefix,
                             variant, probes)
    meta = {"method": METHOD, "variant": variant, "operation": operation, "recipe": _recipe(variant),
            "source": [int(src_dims[0]), int(src_dims[1])], "requested": list(dims["req"]),
            "refine": list(dims["refine"]), "scale": list(dims["scale"]), "crop": crop_args(dims["refine"], dims["req"]),
            "denoise": u["denoise"], "steps_override": u["steps_override"], "seed": u["seed"],
            "tiles": _tile_meta(u, dims["refine"]) if variant == "tile" else None,
            "upscaler": UPSCALER_WEIGHT, "packet_prefix": packet_prefix}
    return g, meta


def build_decoded_upscale(u, video_name, src_dims, filename_prefix, prompt=None, packet_prefix=None, probes=True):
    """Latent-space upscale of a clip that only exists as an mp4: the clip is
    imported into a fresh packet (MMH3Create + MMH3Put primary video),
    VAE-encoded by MMH3H3DecodedUpscalePrepare, then refined like the tile
    variant. Lower fidelity than a saved latent (VAE round trip); the only
    H3-native route for clips generated before latent capture."""
    if u["variant"] != "decoded":
        raise ValueError("build_decoded_upscale needs variant 'decoded'")
    dims = u.get("_dims") or target_dims(src_dims[0], src_dims[1], u)
    w32, h32 = dims["refine"]
    g = {}
    g["load"] = {"class_type": "LoadVideo", "inputs": {"file": video_name}}
    g["pkt_create"] = {"class_type": "MMH3Create",
                       "inputs": {"prompt": (prompt or "").strip() or DECODED_PROMPT, "task": CREATE_TASK_FRAMES,
                                  "seed": int(u["seed"]), "name": os.path.basename(filename_prefix),
                                  "notes": "render-farm video_gen decoded upscale import"}}
    # mmh3_media bca81b8c declares the advanced string inputs as required (defaults ""); a graph
    # that omits them is rejected by the preflight and by /prompt, so set them explicitly.
    g["pkt_put"] = {"class_type": "MMH3Put",
                    "inputs": {"packet": ["pkt_create", 0], "resource": ["load", 0], "role": PUT_ROLE, "order": -1,
                               "mode": "add", "primary": True, "resource_id": "", "name": "", "tags": "",
                               "descriptor_json": "", "extensions_json": ""}}
    # PC 2026-09-10: MMH3H3RefineLoRAs.turbo_override needs a recorded LoRA list ("Record source
    # LoRAs before overriding Turbo"); a created packet has none, so record an explicit empty list.
    # The refine-sampling node then infers the stock 20-step profile (no acceleration LoRA is
    # recorded on the packet it sees) and the graph pins steps_override / denoise itself.
    g["pkt_loras"] = {"class_type": "MMH3GenerationLoRAs",
                      "inputs": {"packet": ["pkt_put", 0], "action": LORAS_ACTION_NONE, "loras_list": "[]"}}
    _loaders(g, "fl2va", None)
    g["prep"] = {"class_type": "MMH3H3DecodedUpscalePrepare",
                 "inputs": {"packet": ["pkt_loras", 0], "video_vae": ["vae", 0], "audio_vae": ["avae", 0],
                            "missing_audio_policy": u["missing_audio_policy"], "geometry_mode": GEOMETRY_MODE_DIMS,
                            "scale": 2.0, "target_width": w32, "target_height": h32, "target_megapixels": 2.1,
                            "align": ALIGN, "enable_chunking": True}}
    _upscaler(g, u, "prep")
    operation = _refine_tail(g, u, "prep", ["prep", 0], dims["refine"], filename_prefix, packet_prefix,
                             "decoded", probes)
    meta = {"method": METHOD, "variant": "decoded", "operation": operation, "recipe": _recipe("decoded"),
            "source": [int(src_dims[0]), int(src_dims[1])], "requested": list(dims["req"]),
            "refine": list(dims["refine"]), "scale": list(dims["scale"]), "crop": crop_args(dims["refine"], dims["req"]),
            "denoise": u["denoise"], "steps_override": u["steps_override"], "seed": u["seed"],
            "tiles": _tile_meta(u, dims["refine"]), "upscaler": UPSCALER_WEIGHT, "packet_prefix": packet_prefix}
    return g, meta


def _recipe(variant):
    return {"tile": "F07 Native Tile Upscale (mmh3_media)", "full": "F07 Latent Upscale (mmh3_media)",
            "decoded": "decoded import + F07 native tile refine (mmh3_media, experimental)"}[variant]


def tile_count(refine, u):
    tw, th, ov = int(u["tile_width"]), int(u["tile_height"]), int(u["tile_overlap"])
    w, h = refine
    cols = max(1, math.ceil((w - ov) / max(1, tw - ov)))
    rows = max(1, math.ceil((h - ov) / max(1, th - ov)))
    return cols * rows


def _tile_meta(u, refine):
    out = {k: u[k] for k in ("tile_width", "tile_height", "tile_overlap", "context_padding")}
    out.update({k: u.get(k, TILE_DEFAULTS[k]) for k in TILE_MODE_OPTIONS})
    out["count"] = tile_count(refine, u)
    return out


# --------------------------------------------------------------------------- introspection

def links(graph):
    """Every [node_key, output_index] reference in an API graph."""
    out = []
    for key, node in graph.items():
        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], int):
                out.append((key, name, val[0], val[1]))
    return out


def check_links(graph):
    """Raise if any link points at a node key that is not in the graph."""
    bad = [(k, n, t) for k, n, t, _ in links(graph) if t not in graph]
    if bad:
        raise ValueError(f"graph links to missing nodes: {bad}")
    return True


def preflight_requirements(graph):
    """What h3_preflight.check verifies for this graph: the class set and the
    combo values it must find in /object_info (the upscaler weight above all)."""
    classes = sorted({n["class_type"] for n in graph.values()})
    enums = {}
    for key, node in graph.items():
        if node["class_type"] == UPSCALER_CLASS:
            enums[(UPSCALER_CLASS, "model_name")] = node["inputs"]["model_name"]
    return {"classes": classes, "enums": enums}
