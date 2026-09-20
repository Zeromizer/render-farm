"""API-format ComfyUI graphs for the audio_gen runner, one per model key.

A graph is a JSON file exported from a workflow that was HEARD to work on this
box ("Save (API format)"), not one written from a node schema: <model>.api.json
next to this module. build() fills it in by node CLASS and input NAME rather
than by node id, so re-exporting the workflow after rearranging it in the UI
needs no code change here. Anything a model does not have (no ABC node, no cfg
input) is simply not set.

yue2_inst: YuE2 3B (int8) + the instrumental LoRA on the CLIP slot (ComfyUI maps
the YuE2 language model to CLIP, so that is strength_clip 1.0 / strength_model 0).
Three stages in one graph: YuE2GenerateABC writes a score, YuE2GenerateMusic
writes music tokens conditioned on it (mode "full"), then a KSampler + audio VAE
turns those into sound. It takes no lyrics: the `lyrics` input carries timed
section tags such as "[intro 0:00-0:03]", the only handle on length and structure.

THE EXPORT MUST BE FLATTENED. The stock template feeds style, lyrics and seed in
through PrimitiveStringMultiline / SeedNode wires, and routes the ABC score
through a ComfySwitchNode whose default is OFF (an empty abc silently forces
mode "off", so mode=full does nothing). build() only writes LITERAL inputs and
never rewires, so the committed JSON has to carry style/lyrics/seed as literals
on the two YuE2 nodes, a literal seed on the KSampler, and abc wired straight
from the ABC node. build() refuses an export where that is not true rather than
generating the template placeholder song.
"""
import copy
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# model key -> what the runner needs to know about it
MODELS = {
    "yue2_inst": {
        "file": "yue2_inst.api.json",
        # Measured on the 4080 SUPER, 2026-09-20, 15/30/45 s beds: wall = 55 + 2.8 x
        # duration. The two token stages dominate (~21 token/s: a score of 1300-2150
        # tokens, then 25 music tokens per audio second); the 32 diffusion steps are
        # 5-10 s. Peak VRAM 5-10 GB, growing with length and score size.
        "realtime_factor": 2.8,
        "load_seconds": 55.0,
        # The model does NOT end on its last section tag: 8 of 8 measured takes ran to
        # the token budget, so max_duration IS the length and the runner's trim + fade
        # IS the ending. The headroom only keeps the cut off the final beat.
        "duration_headroom": 2.0,
    },
}

_TEXT_NODES = ("YuE2GenerateABC", "YuE2GenerateMusic")


def abc_token_cap(duration_s):
    """max_abc_tokens for a bed of this length.

    The node's default is 8192 and 3 of 7 measured seeds ran away to it: the score
    never emitted its end token, which cost ~300 s and changed nothing audible (the
    truncated score still produced music). A finished score was 1300-1500 tokens at
    15 s and ~2150 at 30-45 s, so this leaves real scores room and bounds a runaway
    at under a minute of extra sampling."""
    return int(min(8192, max(3000, 1500 + 40 * float(duration_s))))


def load(model):
    spec = MODELS.get(model)
    if spec is None:
        raise RuntimeError(f"audio_gen model must be one of {sorted(MODELS)}, got {model!r}")
    path = os.path.join(_HERE, spec["file"])
    if not os.path.exists(path):
        raise RuntimeError(
            f"audio_gen graph for {model!r} is not installed: {path} is missing. Export the working "
            f"workflow from ComfyUI with 'Save (API format)' and commit it there.")
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    if not isinstance(graph, dict) or not all(isinstance(n, dict) and "class_type" in n for n in graph.values()):
        raise RuntimeError(f"{path} is not an API-format graph (export with 'Save (API format)', not 'Save')")
    return graph, spec


def build(model, style, lyrics, duration_s, seed, prefix, cfg_scale=None):
    """Return (graph, meta). Raises if the export has no generate or save node."""
    base, spec = load(model)
    graph = copy.deepcopy(base)
    touched = {"text": 0, "music": 0, "save": 0, "sampler": 0, "cfg": 0}
    unset = []
    for node in graph.values():
        cls = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        if cls in _TEXT_NODES:
            touched["text"] += 1
            for key, value in (("style", style), ("lyrics", lyrics), ("seed", int(seed))):
                if isinstance(inputs.get(key), list):   # a wire: whatever feeds it would win
                    unset.append(f"{cls}.{key}")
                else:
                    inputs[key] = value
            if "mode" in inputs and not isinstance(inputs["mode"], list):
                inputs["mode"] = "full"
        if cls == "YuE2GenerateABC":
            inputs["max_abc_tokens"] = abc_token_cap(duration_s)
        if cls == "YuE2GenerateMusic":
            touched["music"] += 1
            # 0.04 s grid (the node's own step); never below what was asked for.
            cap = round((float(duration_s) + spec["duration_headroom"]) / 0.04) * 0.04
            inputs["max_duration"] = round(cap, 2)
            # Only where the export has the input. The v0.36.0 release tag this box runs has no
            # cfg_scale on the node (it arrived after the tag), and ComfyUI rejects a prompt
            # carrying an input the node does not declare. Newer ComfyUI: re-export with it.
            if cfg_scale is not None and "cfg_scale" in inputs:
                inputs["cfg_scale"] = float(cfg_scale)
                touched["cfg"] = 1
            if not isinstance(inputs.get("abc"), list):
                unset.append("YuE2GenerateMusic.abc (must be wired from YuE2GenerateABC)")
        if cls in ("KSampler", "KSamplerAdvanced"):
            # One seed drives all three stages, so a retry changes the whole take.
            key = "seed" if cls == "KSampler" else "noise_seed"
            if isinstance(inputs.get(key), list):
                unset.append(f"{cls}.{key}")
            else:
                inputs[key] = int(seed)
                touched["sampler"] += 1
        if cls.startswith("SaveAudio"):
            touched["save"] += 1
            inputs["filename_prefix"] = prefix
    if not touched["music"]:
        raise RuntimeError(f"{spec['file']} has no YuE2GenerateMusic node")
    if not touched["save"]:
        raise RuntimeError(f"{spec['file']} has no SaveAudio node, so nothing would come back")
    if unset:
        raise RuntimeError(f"{spec['file']} is not flattened, so this request would not reach the model: "
                           f"{', '.join(unset)}. See the note at the top of audiogen/graphs.py.")
    meta = {"model": model, "nodes": len(graph), **touched}
    return graph, meta


def hint(model, duration_s):
    """comfy_client.wait() ETA seed. Most of the time goes in the autoregressive
    token stages, not the 32 diffusion steps, so until a tqdm line appears it is
    one long 'step' sized from the length asked for (25 music tokens per second)."""
    spec = MODELS[model]
    return {"steps": 1, "step_seconds": max(10.0, float(duration_s) * spec["realtime_factor"]),
            "load_seconds": spec["load_seconds"], "tail_seconds": 8.0}
