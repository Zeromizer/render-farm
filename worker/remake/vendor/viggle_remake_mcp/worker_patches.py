# SPDX-License-Identifier: Apache-2.0
"""Patches on the MiniMax-H3 `ref2va` workflow of diffusers d6726f3, driven by the job in `J`.

- the prompt embeds are the job's, computed outside the pipeline (`J["embeds"]`)
- the target audio rows hold the job's soundtrack as a clean latent; only video rows are stepped
- each pinned ref is also a keyframe block inside the target's rotary span at its real frame (`J["pin"]`), cut out of
  the sequence once 1 - t reaches `J["keyframe_until"]`
- a prefix (`J["prefix_rows"]`) is the first target rows: held at the condition timestep, never stepped, restored
  un-noised before decode
- both schedulers take the sigma grid of the job's mode (`J["mode"]`); there is no default grid
- each block's feed-forward runs over the packed sequence `FF_ROWS` rows at a time (row-wise, so the same result)
"""
import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (_ROPE_FRAME_RESCALE, MiniMaxH3Ref2VAPrepareLayoutStep,
                                                                   MiniMaxH3SetTimestepsStep, _frame_position_grid)
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep, MiniMaxH3Ref2VALoopDenoiser
from diffusers.modular_pipelines.minimax_h3.encoders import (MiniMaxH3Ref2VAReferenceEncoderStep,
                                                             MiniMaxH3Ref2VATextEncoderStep, encode_vae_condition)

# 6-step grids in the video stream's sigma; each stream's own shift is applied in sigmas()
SIGMAS = {"swap": [1, .990476, .965243, .923077, .8, .631579, 0],
          "edit": [1, .983607, .96, .923077, .857143, .705882, 0]}
TEXT_TAG = 1
# Its three S x 28672 bf16 transients (projection, LoRA branch, their sum) are the largest allocation of a step:
# 32 GiB at 345 frames on the 736 canvas unsliced, 2.6 GiB at 16k rows
FF_ROWS = 16384
J = {}


def sigmas(mode, shift):
    sh = lambda u, s: s * u / (1 + (s - 1) * u)  # noqa: E731
    return [sh(x / (12 - 11 * x), shift) for x in SIGMAS[mode]]  # video sigma -> u -> this stream's sigma


def use_job_embeds(self, components, state):
    block_state = self.get_block_state(state)
    block_state.prompt_embeds = J["embeds"].to(components._execution_device, torch.bfloat16)
    block_state.text_token_tags = torch.full((J["embeds"].shape[1],), TEXT_TAG, dtype=torch.long)
    self.set_block_state(state, block_state)
    return components, state


_build_row_timesteps = MiniMaxH3SetTimestepsStep.build_row_timesteps


def row_timesteps(*a):
    # positional arg 6 = audio_timestep; H3's t = 1 is clean
    uniq, inv = _build_row_timesteps(*a[:6], 1.0, *a[7:])
    if not J["np"]:
        return uniq, inv
    row = uniq[inv]  # the prefix = the first target video rows (patchify is frame-major), at the condition timestep
    row[a[0][a[2]:a[2] + J["np"]]] = a[7]
    return torch.unique(row, sorted=True, return_inverse=True)


@torch.no_grad()
def video_only_step(self, components, block_state, i, t):
    n = block_state.num_condition_video_rows + J["np"]
    block_state.latents[n:] = components.scheduler.step(
        block_state.noise_pred[0, n:].float(), t, block_state.latents[n:], return_dict=False)[0]
    return components, block_state


_after_denoise = MiniMaxH3AfterDenoiseStep.__call__


@torch.no_grad()
def restore_prefix(self, components, state):
    if J["np"]:
        block_state = self.get_block_state(state)
        n = block_state.num_condition_video_rows
        block_state.latents[n:n + J["np"]] = J["prefix_rows"].to(block_state.latents.device, block_state.latents.dtype)
        self.set_block_state(state, block_state)
    return _after_denoise(self, components, state)


_encode_references = MiniMaxH3Ref2VAReferenceEncoderStep.__call__


def encode_references_then_keyframes(self, components, state):
    components, state = _encode_references(self, components, state)
    with torch.no_grad():
        kf = [encode_vae_condition(components.vae, torch.from_numpy(np.array(im)).to(components._execution_device)
                                   .permute(2, 0, 1)[None, :, None], components.pixel_mean, components.pixel_std,
                                   components.keyframe_encode_seed) for im in J["kf_images"]]
    state.set("condition_latents", list(state.get("condition_latents") or []) + kf)
    return components, state


_layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence


def layout_with_keyframes(text_token_tags, references, condition_latents, audio_condition_latents, num_latent_frames,
                          latent_height, latent_width, num_audio_latents, patch_size, audio_channels, audio_tag,
                          video_tag):
    k = len(J["pin"])
    pos, tags, vi, ai, ti, ncv, nca = _layout(text_token_tags, references, condition_latents[:len(condition_latents) - k],
                                              audio_condition_latents, num_latent_frames, latent_height, latent_width,
                                              num_audio_latents, patch_size, audio_channels, audio_tag, video_tag)
    if not k:
        return pos, tags, vi, ai, ti, ncv, nca
    s = text_token_tags.shape[0] + ncv + nca  # the first target audio row; its time is the target's rotary origin
    origin = float(pos[s, 0])
    blocks = []
    for anchor, lat in zip(J["pin"], condition_latents[-k:]):
        grid, _ = _frame_position_grid(lat.shape[3], lat.shape[4], patch_size[1], patch_size[2])
        p = torch.zeros(grid.shape[0], 3, dtype=torch.float64)
        p[:, 0] = origin + _ROPE_FRAME_RESCALE * float(anchor)
        p[:, 1:] = grid
        blocks.append(p)
    n = sum(b.shape[0] for b in blocks)
    pos = torch.cat([pos[:s], *blocks, pos[s:]])
    tags = torch.cat([tags[:s], torch.full((n,), video_tag, dtype=tags.dtype), tags[s:]])
    vi = torch.cat([vi[vi < s], torch.arange(s, s + n), vi[vi >= s] + n])
    ai = torch.where(ai >= s, ai + n, ai)
    return pos, tags, vi, ai, ti, ncv + n, nca


_denoise = MiniMaxH3Ref2VALoopDenoiser.__call__


def kf_release(self, components, block_state, i, t):
    # the cut keyframe rows get zero velocity back, which the scheduler never reads (conditioning rows)
    nkf = J["nkf"]
    if not nkf or 1 - float(t) >= J["keyframe_until"]:
        return _denoise(self, components, block_state, i, t)
    ncv = block_state.num_condition_video_rows
    f = block_state.denoiser_input_fields
    vi, ai = f["video_indices"], f["audio_indices"]
    lo, hi = int(vi[ncv - nkf]), int(vi[ncv - 1]) + 1
    assert hi - lo == nkf and hi == int(ai[block_state.num_condition_audio_rows]), (lo, hi, nkf)
    keep = torch.ones(f["position_ids"].shape[0], dtype=torch.bool, device=vi.device)
    keep[lo:hi] = False
    plan = list(block_state.row_timestep_plan)
    plan[i] = (plan[i][0], plan[i][1][keep])
    full = block_state.latents, f, block_state.row_timestep_plan
    block_state.latents = torch.cat([block_state.latents[:ncv - nkf], block_state.latents[ncv:]])
    block_state.denoiser_input_fields = dict(
        f, position_ids=f["position_ids"][keep], token_tags=f["token_tags"][keep],
        video_indices=torch.cat([vi[:ncv - nkf], vi[ncv:] - nkf]), audio_indices=ai - nkf * (ai >= hi))
    block_state.row_timestep_plan = plan
    components, block_state = _denoise(self, components, block_state, i, t)
    block_state.latents, block_state.denoiser_input_fields, block_state.row_timestep_plan = full
    p = block_state.noise_pred
    block_state.noise_pred = torch.cat([p[:, :ncv - nkf], p.new_zeros(1, nkf, p.shape[2]), p[:, ncv - nkf:]], 1)
    return components, block_state


def apply(pipe):
    MiniMaxH3Ref2VATextEncoderStep.__call__ = use_job_embeds
    MiniMaxH3SetTimestepsStep.build_row_timesteps = staticmethod(row_timesteps)
    MiniMaxH3LoopSchedulerStep.__call__ = video_only_step
    MiniMaxH3AfterDenoiseStep.__call__ = restore_prefix
    MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = encode_references_then_keyframes
    MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(layout_with_keyframes)
    MiniMaxH3Ref2VALoopDenoiser.__call__ = kf_release
    for block in pipe.transformer_ref.transformer_blocks:
        block.ff.forward = (lambda f: lambda x: torch.cat([f(c) for c in x.split(FF_ROWS, dim=-2)], dim=-2))(
            block.ff.forward)
    for sched, s in ((pipe.scheduler, 12.0), (pipe.audio_scheduler, 3.0)):
        sched.set_timesteps = (lambda f, s: lambda *a, **k: f(sigmas=sigmas(J["mode"], s), device=k.get("device")))(
            sched.set_timesteps, s)
