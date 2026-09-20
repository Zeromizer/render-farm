"""API-format ComfyUI graphs for the audio_gen runner, one per model key.

A graph is a JSON file exported from a workflow that was HEARD to work on this
box ("Save (API format)"), not one written from a node schema: <model>.api.json
next to this module. build() fills it in by node CLASS and input NAME rather
than by node id, so re-exporting the workflow after rearranging it in the UI
needs no code change here. Anything a model does not have (no ABC node, no cfg
input) is simply not set.

yue2_inst: YuE2 3B (int8) + the instrumental LoRA on the CLIP slot. The model
writes an ABC score first (mode "full") and then the audio conditioned on it.
It takes no lyrics: the `lyrics` input carries timed section tags such as
"[intro 0:00-0:03]" which are also the only handle on length and structure.
"""
import copy
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# model key -> what the runner needs to know about it
MODELS = {
    "yue2_inst": {
        "file": "yue2_inst.api.json",
        # Seconds of wall clock per second of audio on the 4080 SUPER, for the
        # ETA before ComfyUI logs a step. Measured value goes here after Step 0.
        "realtime_factor": 0.5,
        "load_seconds": 50.0,
        # The model stops on its own near the last tag; the cap only has to be
        # past it. The runner trims to the exact length afterwards.
        "duration_headroom": 2.0,
    },
}

_TEXT_NODES = ("YuE2GenerateABC", "YuE2GenerateMusic")


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
    touched = {"text": 0, "music": 0, "save": 0}
    for node in graph.values():
        cls = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        if cls in _TEXT_NODES:
            touched["text"] += 1
            for key, value in (("style", style), ("lyrics", lyrics), ("seed", int(seed))):
                if key in inputs and not isinstance(inputs[key], list):   # a list is a wire, leave it
                    inputs[key] = value
            if "mode" in inputs and not isinstance(inputs["mode"], list):
                inputs["mode"] = "full"
        if cls == "YuE2GenerateMusic":
            touched["music"] += 1
            # 0.04 s grid (the node's own step); never below what was asked for.
            cap = round((float(duration_s) + spec["duration_headroom"]) / 0.04) * 0.04
            inputs["max_duration"] = round(cap, 2)
            if cfg_scale is not None:
                inputs["cfg_scale"] = float(cfg_scale)
        if cls.startswith("SaveAudio"):
            touched["save"] += 1
            inputs["filename_prefix"] = prefix
    if not touched["music"]:
        raise RuntimeError(f"{spec['file']} has no YuE2GenerateMusic node")
    if not touched["save"]:
        raise RuntimeError(f"{spec['file']} has no SaveAudio node, so nothing would come back")
    meta = {"model": model, "nodes": len(graph), **touched}
    return graph, meta


def hint(model, duration_s):
    """comfy_client.wait() ETA seed. The generation is autoregressive, not a fixed
    number of diffusion steps, so until a tqdm line appears it is one long 'step'."""
    spec = MODELS[model]
    return {"steps": 1, "step_seconds": max(10.0, float(duration_s) * spec["realtime_factor"]),
            "load_seconds": spec["load_seconds"], "tail_seconds": 8.0}
