"""Aimotion H3 latent hold: chain MiniMax H3 shots on their real latents.

Two nodes, installed into ComfyUI/custom_nodes/aimotion_h3_hold by spikes/h3_chain/try_chain.py.

  AimH3SaveLatent  writes a sampled AV latent to output/aim_h3/<name>.pt (video + audio, CPU).
  AimH3Hold        splices frames of a saved latent into this shot's starting latent and sets
                   the noise mask to 0 there, so the sampler keeps them:
                     at=head: the source's LAST `frames` frames become this shot's first frames
                              (a continuation that starts inside the previous shot's motion);
                     at=tail: the source's FIRST `frames` frames become this shot's last frames
                              (the shot lands exactly on the next one, or on its own start for a loop).

H3 lays latents out as FRAME_PER_TOKEN (1, 4, 4, 4, 4): every 5 latents are a 17-frame block,
from frame 0, and a 5 + 17k clip ends in a 2-latent stub. With clip lengths and `frames` both
5 + 17k, the head and the tail both start on a block boundary, so whole latents line up.
ComfyUI's MiniMax H3 model runs masked tokens at the condition timestep and pastes the clean
latent back after the last step (model_base.MiniMaxH3.scale_latent_inpaint), so held frames
come out as the source's own latents rather than a redraw.

Audio is 40 latents/s against 24 video frames/s; `frames` / 24 * 40 audio latents are held too
when hold_audio is on.
"""
import os

import torch

import folder_paths
from comfy.nested_tensor import NestedTensor

BLOCK = 17
AUDIO_PER_FRAME = 40 / 24


def _frames_of(n_latents):
    return sum((1, 4, 4, 4, 4)[k % 5] for k in range(n_latents))


def _latents_in(frames):
    if frames % BLOCK != 5:
        raise ValueError(f"frames must be 5 + 17k (5, 22, 39, 56, ...), got {frames}")
    return (frames - 5) // BLOCK * 5 + 2


def _path(name):
    d = os.path.join(folder_paths.get_output_directory(), "aim_h3")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.pt")


def _parts(samples):
    if isinstance(samples, NestedTensor):
        return list(samples.unbind())
    raise ValueError("expected a MiniMax H3 AV latent (NestedTensor of video + audio)")


class AimH3SaveLatent:
    CATEGORY = "latent/aimotion"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "name": ("STRING", {"default": "shot"})}}

    def save(self, latent, name):
        video, audio = _parts(latent["samples"])
        torch.save({"video": video.detach().cpu(), "audio": audio.detach().cpu()}, _path(name))
        print(f"[AimH3SaveLatent] {name}: video {tuple(video.shape)} audio {tuple(audio.shape)}")
        return (latent,)


class AimH3Hold:
    CATEGORY = "latent/aimotion"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "hold"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "source": ("STRING", {"default": "shot"}),
            "frames": ("INT", {"default": 39, "min": 5, "max": 4096}),
            "at": (["head", "tail"],),
            "hold_audio": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "hold_video": ("BOOLEAN", {"default": True}),
            "source_from": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 17,
                                    "tooltip": "tail only: take the source's frames from here (a multiple of 17)"}),
        }}

    def hold(self, latent, source, frames, at, hold_audio, hold_video=True, source_from=0):
        video, audio = (t.clone() for t in _parts(latent["samples"]))
        src = torch.load(_path(source), map_location="cpu")
        sv, sa = src["video"].to(video.dtype), src["audio"].to(audio.dtype)
        if sv.shape[1] != video.shape[1] or sv.shape[3:] != video.shape[3:]:
            raise ValueError(f"source {tuple(sv.shape)} does not match this shot {tuple(video.shape)}")
        n = _latents_in(frames) if hold_video else 0
        T, Ts = video.shape[2], sv.shape[2]
        if n > min(T, Ts):
            raise ValueError(f"{frames} frames do not fit in {_frames_of(T)} / {_frames_of(Ts)} frame clips")
        if source_from % BLOCK:
            raise ValueError(f"source_from must be a multiple of {BLOCK}, got {source_from}")
        l0 = source_from // BLOCK * 5          # latent index where that block starts
        a0 = int(round(source_from * AUDIO_PER_FRAME))
        if at == "tail" and l0 + n > Ts:
            raise ValueError(f"source frames {source_from}+{frames} run past the source's {_frames_of(Ts)} frames")
        a = int(round(frames * AUDIO_PER_FRAME)) if hold_audio else 0
        a = min(a, audio.shape[-1], sa.shape[-1] - (a0 if at == "tail" else 0))

        old = latent.get("noise_mask")
        if isinstance(old, NestedTensor):
            vm, am = (m.clone() for m in old.unbind())
        else:
            vm, am = torch.ones_like(video), torch.ones_like(audio)

        if at == "head":
            if n:
                video[:, :, :n] = sv[:, :, Ts - n:]
                vm[:, :, :n] = 0.0
            if a:
                audio[..., :a] = sa[..., sa.shape[-1] - a:]
                am[..., :a] = 0.0
            where = f"frames 0..{frames - 1} <- source frames {_frames_of(Ts) - frames}..{_frames_of(Ts) - 1}"
        else:
            if n:
                video[:, :, T - n:] = sv[:, :, l0:l0 + n]
                vm[:, :, T - n:] = 0.0
            if a:
                audio[..., audio.shape[-1] - a:] = sa[..., a0:a0 + a]
                am[..., am.shape[-1] - a:] = 0.0
            f0 = _frames_of(T - n)
            where = f"frames {f0}..{f0 + frames - 1} <- source frames {source_from}..{source_from + frames - 1}"
        print(f"[AimH3Hold] {at} {source}: {where}, {n} video latents, {a} audio latents")

        out = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
        out["samples"] = NestedTensor((video, audio))
        out["noise_mask"] = NestedTensor((vm, am))
        return (out,)


NODE_CLASS_MAPPINGS = {"AimH3SaveLatent": AimH3SaveLatent, "AimH3Hold": AimH3Hold}
NODE_DISPLAY_NAME_MAPPINGS = {"AimH3SaveLatent": "Aimotion H3 Save Latent",
                              "AimH3Hold": "Aimotion H3 Hold (head / tail)"}
