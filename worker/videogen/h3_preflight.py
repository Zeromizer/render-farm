"""Fail-loud preflight for the mmh3 graphs against ComfyUI's /object_info.

The brief's rule is "never fall back silently": a missing node pack, a missing
upscaler weight or a wrong combo string must fail the job with one message
that says what to install. /object_info lists every registered node class
with its input schema, so all of that is checkable before /prompt is called.

Pure functions (no HTTP): the runner passes the /object_info dict in, the
tests pass fakes. Two shapes of input spec are handled:
  required/optional[name] = [ [options...], {cfg} ]   # combo
  required/optional[name] = [ "TYPE", {cfg} ]         # typed input
Dotted DynamicCombo keys ("mode.width") cannot be validated from the schema
and are left to /prompt's own validation, which reports node_errors loudly.
"""
from videogen import graphs_h3

INSTALL_HINTS = {
    "mmh3": (f"install {graphs_h3.MMH3_REPO} into ComfyUI/custom_nodes/{graphs_h3.MMH3_NODE_DIR} "
             f"(pip install -r requirements.txt in the ComfyUI venv) and restart ComfyUI"),
    "upscaler": (f"install {graphs_h3.UPSCALER_REPO} into ComfyUI/custom_nodes/{graphs_h3.UPSCALER_NODE_DIR} "
                 f"and restart ComfyUI"),
    "weight": (f"download {graphs_h3.UPSCALER_WEIGHT} from {graphs_h3.UPSCALER_WEIGHTS_URL} into "
               f"ComfyUI/models/{graphs_h3.UPSCALER_FOLDER}/ and restart ComfyUI"),
    "core": "update ComfyUI (core node missing)",
}
OPTIONAL_CLASSES = (graphs_h3.PROBE_CLASS,)


def _inputs_of(spec):
    """{name: [type_or_options, cfg]} over required + optional."""
    out = {}
    inp = (spec or {}).get("input") or {}
    for section in ("required", "optional"):
        for name, val in (inp.get(section) or {}).items():
            out[name] = val if isinstance(val, (list, tuple)) else [val, {}]
    return out


def _options(val):
    """The combo option list of an input spec, or None when it is not a combo."""
    if not val:
        return None
    head = val[0]
    if isinstance(head, (list, tuple)):
        return list(head)
    # v3 nodes may serialise combos as {"type": "COMBO", "options": [...]}
    if isinstance(head, dict) and isinstance(head.get("options"), (list, tuple)):
        return list(head["options"])
    if isinstance(head, str) and head == "COMBO" and len(val) > 1 and isinstance(val[1], dict):
        opts = val[1].get("options")
        return list(opts) if isinstance(opts, (list, tuple)) else None
    return None


def _hint_for(cls):
    if cls.startswith("MMH3"):
        return INSTALL_HINTS["mmh3"]
    if cls == graphs_h3.UPSCALER_CLASS:
        return INSTALL_HINTS["upscaler"]
    return INSTALL_HINTS["core"]


def check(object_info, graph, optional_classes=OPTIONAL_CLASSES):
    """Compare a graph with /object_info.

    Returns {"missing_classes": [cls...], "dropped_optional": [node_key...],
             "bad_enum": [{node, class, input, value, options}],
             "unknown_inputs": [{node, class, input}], "missing_required": [{node, class, input}]}"""
    info = object_info or {}
    res = {"missing_classes": [], "dropped_optional": [], "bad_enum": [], "unknown_inputs": [],
           "missing_required": []}
    seen_missing = set()
    for key, node in graph.items():
        cls = node["class_type"]
        spec = info.get(cls)
        if spec is None:
            if cls in optional_classes:
                res["dropped_optional"].append(key)
            elif cls not in seen_missing:
                seen_missing.add(cls)
                res["missing_classes"].append(cls)
            continue
        inputs = _inputs_of(spec)
        required = set(((spec.get("input") or {}).get("required") or {}).keys())
        given = node.get("inputs") or {}
        for name in required:
            if name not in given and not any(k.startswith(name + ".") for k in given):
                res["missing_required"].append({"node": key, "class": cls, "input": name})
        for name, val in given.items():
            if "." in name:
                continue
            if name not in inputs:
                res["unknown_inputs"].append({"node": key, "class": cls, "input": name})
                continue
            if isinstance(val, list):
                continue  # a link; types are ComfyUI's job
            opts = _options(inputs[name])
            if opts is not None and val not in opts:
                res["bad_enum"].append({"node": key, "class": cls, "input": name, "value": val,
                                        "options": opts[:25]})
    return res


def error_message(res):
    """One operator-readable message, or None when the graph can run."""
    parts = []
    if res["missing_classes"]:
        by_hint = {}
        for cls in res["missing_classes"]:
            by_hint.setdefault(_hint_for(cls), []).append(cls)
        for hint, classes in by_hint.items():
            parts.append(f"missing node classes {', '.join(classes)}: {hint}")
    for e in res["bad_enum"]:
        if e["class"] == graphs_h3.UPSCALER_CLASS and e["input"] == "model_name":
            parts.append(f"{e['class']}.model_name has no {e['value']!r} (installed: {e['options'] or 'none'}): "
                         f"{INSTALL_HINTS['weight']}")
        else:
            parts.append(f"{e['class']}.{e['input']} does not accept {e['value']!r} on this install; options are "
                         f"{e['options']} (fix the constant in videogen/graphs_h3.py)")
    for e in res["missing_required"]:
        parts.append(f"{e['class']} ({e['node']}) requires input {e['input']!r}, which the graph does not set "
                     f"(node pack version mismatch; compare videogen/graphs_h3.py with the installed nodes)")
    if not parts:
        return None
    return "h3_latent_upscale preflight failed: " + "; ".join(parts)


def strip_optional(graph, dropped):
    """Remove optional probe nodes that this ComfyUI does not have."""
    if not dropped:
        return graph
    return {k: v for k, v in graph.items() if k not in set(dropped)}


def run(object_info, graph, optional_classes=OPTIONAL_CLASSES):
    """check + strip; raises RuntimeError with the combined message.
    Returns (graph_without_missing_probes, result)."""
    res = check(object_info, graph, optional_classes)
    msg = error_message(res)
    if msg:
        raise RuntimeError(msg)
    return strip_optional(graph, res["dropped_optional"]), res
