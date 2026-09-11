"""Settings validation for recipe text_overlay_v2 (docs/aftereffects-text-overlay-v2.md).

Envelope (schema_version 1, assets, scope) is validated by schema.py; this
module validates and normalizes `settings` for v2: ordered layers with
frame-aligned timing, anchors, independent scale, text stretch/stroke,
clip, track mattes, built-in effects and keyframes with easing.

Normalization gives the recipe one canonical shape: frames AND seconds for
every time, explicit anchors, opacity keyframes generated from fades, ease
always as {"in": [speed, influence], "out": [...]} or "hold" / "linear".
"""
import math
import re

from aftereffects import schema as s1
from aftereffects.errors import AEError

MAX_LAYERS = 60
MAX_EFFECTS = 8
MAX_KEYS = 200
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_KEYPROP_RE = re.compile(r"^(position|scale|rotation|opacity|brightness|stretch|effects\.(\d)\.([a-z_]+))$")

BLENDS = {"normal": "NORMAL", "multiply": "MULTIPLY", "screen": "SCREEN", "add": "ADD", "overlay": "OVERLAY",
          "lighten": "LIGHTEN", "darken": "DARKEN"}
MATTE_MODES = {"alpha": "ALPHA", "alpha_inverted": "ALPHA_INVERTED", "luma": "LUMA", "luma_inverted": "LUMA_INVERTED"}
EASE_PRESETS = {"linear": "linear", "hold": "hold",
                "ease_in": {"in": [0.0, 33.3], "out": [0.0, 0.1]},
                "ease_out": {"in": [0.0, 0.1], "out": [0.0, 33.3]},
                "ease_in_out": {"in": [0.0, 33.3], "out": [0.0, 33.3]}}

# effect type -> (AE match name, {param: (default, lo, hi | "color" | enum tuple)})
EFFECTS = {
    "glow": ("ADBE Glo2", {"color": ("#FFFFFF", "color"), "radius": (10.0, 0, 500), "intensity": (1.0, 0, 4),
                           "threshold": (0.0, 0, 100), "based_on": ("alpha", ("alpha", "color")),
                           "composite": ("on_top", ("on_top", "behind"))}),
    "bevel_highlight": ("ADBE Bevel Alpha", {"thickness": (2.0, 0, 100), "angle": (90.0, -360, 360),
                                             "color": ("#FFFFFF", "color"), "intensity": (0.5, 0, 1)}),
    "drop_shadow": ("ADBE Drop Shadow", {"color": ("#000000", "color"), "opacity": (50.0, 0, 100),
                                         "distance": (8.0, 0, 500), "softness": (20.0, 0, 500), "direction": (135.0, -360, 360)}),
    "gaussian_blur": ("ADBE Gaussian Blur 2", {"radius": (10.0, 0, 500)}),
    "fill": ("ADBE Fill", {"color": ("#FFFFFF", "color"), "opacity": (100.0, 0, 100)}),
    "tint": ("ADBE Tint", {"black": ("#000000", "color"), "white": ("#FFFFFF", "color"), "amount": (100.0, 0, 100)}),
    "choker": ("ADBE Simple Choker", {"amount": (0.0, -100, 100)}),
    "exposure": ("ADBE Exposure2", {"stops": (0.0, -10, 10)}),
    "roughen_edges": ("ADBE Roughen Edges", {"border": (8.0, 0, 100), "scale": (100.0, 0, 300),
                                             "complexity": (2, 1, 10), "seed": (0, 0, 100000)}),
}
# keyframable effect params (numbers only)
EFFECT_KEYABLE = {"glow": ("radius", "intensity", "threshold"), "bevel_highlight": ("thickness", "angle", "intensity"),
                  "drop_shadow": ("opacity", "distance", "softness", "direction"), "gaussian_blur": ("radius",),
                  "fill": ("opacity",), "tint": ("amount",), "choker": ("amount",), "exposure": ("stops",),
                  "roughen_edges": ("border", "scale")}

CAPABILITIES = {
    "layers": ["text", "shape", "image"],
    "time_unit": "frames (in_f/out_f, key f); seconds accepted",
    "features": ["ordered_layers", "anchor", "independent_scale", "rotation", "blend", "motion_blur", "clip",
                 "matte", "text_stretch", "text_stroke", "box_text", "fade_in_f", "fade_out_f", "keyframes",
                 "easing_presets", "easing_explicit", "brightness_keyframes", "svg_footage", "proof_frames"],
    "effects": sorted(EFFECTS),
    "keyframe_properties": ["position", "scale", "rotation", "opacity", "brightness", "stretch", "effects.<i>.<param>"],
    "blend_modes": sorted(BLENDS), "matte_modes": sorted(MATTE_MODES), "ease": sorted(EASE_PRESETS) + ["{in:[speed,influence],out:[...]}"],
    "bounds": {"max_layers": MAX_LAYERS, "max_effects_per_layer": MAX_EFFECTS, "max_keys_per_property": MAX_KEYS,
               "max_text_chars": s1.MAX_TEXT_CHARS, "max_assets": s1.MAX_ASSETS, "stretch": [0.1, 10],
               "stroke_width": [0, 50], "scale_percent": [1, 1000], "glow_radius": [0, 500], "max_proof_frames": 12},
    "not_supported": ["third-party plugins", "arbitrary expressions", "cubic-bezier easing (use explicit speed/influence)",
                      "inner shadow (use bevel_highlight)", "text on path", "3D layers"],
}


def _bad(msg, **detail):
    raise AEError("INVALID_REQUEST", msg, detail)


def _num(v, name, lo, hi, integer=False):
    return s1._num(v, name, lo, hi, allow_float=not integer)


def _color(v, name):
    return s1._color(v, name)


def _composition(v):
    comp = s1._composition(v)
    mb = (v or {}).get("motion_blur") or {}
    if mb and not isinstance(mb, dict):
        _bad("composition.motion_blur must be an object")
    comp["motion_blur"] = {
        "enabled": bool(mb.get("enabled", False)),
        "shutter_angle": _num(mb.get("shutter_angle", 180), "motion_blur.shutter_angle", 0, 720),
        "shutter_phase": _num(mb.get("shutter_phase", -90), "motion_blur.shutter_phase", -360, 360),
        "samples_per_frame": _num(mb.get("samples_per_frame", 16), "motion_blur.samples_per_frame", 2, 64, integer=True),
        "adaptive_sample_limit": _num(mb.get("adaptive_sample_limit", 128), "motion_blur.adaptive_sample_limit", 16, 256, integer=True),
    }
    return comp


def _frame(v, name, fps, frames, spec, key_s):
    """Frames win; seconds are converted; result is an int in [0, frames]."""
    if v is not None:
        f = _num(v, name, 0, frames, integer=True)
    elif spec.get(key_s) is not None:
        f = int(round(_num(spec[key_s], key_s, 0, frames / fps) * fps))
    else:
        return None
    return int(f)


def _anchor(v, name, kind):
    default = {"x": "center", "y": "baseline" if kind == "text" else "center"}
    if v is None:
        return default
    if not isinstance(v, dict):
        _bad(f"{name} must be an object")
    out = {}
    for axis, words in (("x", ("left", "center", "right")), ("y", ("top", "center", "bottom", "baseline"))):
        a = v.get(axis, default[axis])
        if isinstance(a, str):
            if a not in words:
                _bad(f"{name}.{axis} must be one of {words} or a number")
            if a == "baseline" and kind != "text":
                _bad(f"{name}.y baseline is for text layers only")
            out[axis] = a
        else:
            out[axis] = _num(a, f"{name}.{axis}", -20000, 20000)
    return out


def _ease(v, name):
    if v is None:
        return "linear"
    if isinstance(v, str):
        if v not in EASE_PRESETS:
            _bad(f"{name} must be one of {sorted(EASE_PRESETS)} or {{in, out}}")
        return EASE_PRESETS[v]
    if not isinstance(v, dict):
        _bad(f"{name} must be a preset name or {{in: [speed, influence], out: [...]}}")
    out = {}
    for side in ("in", "out"):
        pair = v.get(side, [0.0, 0.1])
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            _bad(f"{name}.{side} must be [speed, influence]")
        out[side] = [_num(pair[0], f"{name}.{side}[0]", -100000, 100000), _num(pair[1], f"{name}.{side}[1]", 0.1, 100)]
    return out


def _keyframes(v, name, layer, comp, effects):
    if v is None:
        return {}
    if not isinstance(v, dict):
        _bad(f"{name} must be an object of property -> keys")
    out = {}
    for prop, keys in v.items():
        m = _KEYPROP_RE.match(str(prop))
        if not m:
            _bad(f"{name}.{prop}: unknown keyframe property")
        if prop == "stretch" and layer["kind"] != "text":
            _bad(f"{name}.stretch is for text layers only")
        if prop.startswith("effects."):
            idx, param = int(m.group(2)), m.group(3)
            if idx >= len(effects):
                _bad(f"{name}.{prop}: layer has {len(effects)} effect(s)")
            if param not in EFFECT_KEYABLE.get(effects[idx]["type"], ()):
                _bad(f"{name}.{prop}: {param} is not keyframable on {effects[idx]['type']}")
        if not isinstance(keys, list) or not keys or len(keys) > MAX_KEYS:
            _bad(f"{name}.{prop} must be a list of 1..{MAX_KEYS} keys")
        norm, last_f = [], -1
        for i, k in enumerate(keys):
            if not isinstance(k, dict):
                _bad(f"{name}.{prop}[{i}] must be an object")
            f = _frame(k.get("f"), f"{name}.{prop}[{i}].f", comp["fps"], comp["frames"], k, "t")
            if f is None:
                _bad(f"{name}.{prop}[{i}] needs f (frame)")
            if not (layer["in_f"] <= f <= layer["out_f"]):
                raise AEError("KEYFRAME_INVALID", f"{name}.{prop}[{i}] frame {f} outside the layer's {layer['in_f']}..{layer['out_f']}")
            if f <= last_f:
                _bad(f"{name}.{prop}: frames must increase")
            last_f = f
            val = k.get("v")
            if prop in ("position", "scale", "stretch"):
                lim = {"position": (-20000, 20000), "scale": (1, 1000), "stretch": (0.1, 10)}[prop]
                val = s1._pair(val, f"{name}.{prop}[{i}].v", *lim)
            elif prop == "opacity":
                val = _num(val, f"{name}.{prop}[{i}].v", 0, 100)
            elif prop == "brightness":
                val = _num(val, f"{name}.{prop}[{i}].v", 0.1, 4)
            elif prop == "rotation":
                val = _num(val, f"{name}.{prop}[{i}].v", -36000, 36000)
            else:
                _, lo, hi = EFFECTS[effects[int(m.group(2))]["type"]][1][m.group(3)][:3] + ((None,) * 3)[:0] if False else (None, None, None)
                spec = EFFECTS[effects[int(m.group(2))]["type"]][1][m.group(3)]
                val = _num(val, f"{name}.{prop}[{i}].v", spec[1], spec[2])
            norm.append({"f": f, "t": round(f / comp["fps"], 6), "v": val, "ease": _ease(k.get("ease"), f"{name}.{prop}[{i}].ease")})
        out[prop] = norm
    return out


def _effects(v, name):
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > MAX_EFFECTS:
        _bad(f"{name} must be a list of at most {MAX_EFFECTS}")
    out = []
    for i, e in enumerate(v):
        if not isinstance(e, dict) or e.get("type") not in EFFECTS:
            raise AEError("EFFECT_MISSING", f"{name}[{i}]: type must be one of {sorted(EFFECTS)}, got {(e or {}).get('type')!r}")
        match, params = EFFECTS[e["type"]]
        norm = {"type": e["type"], "match_name": match}
        for p, spec in params.items():
            raw = e.get(p, spec[0])
            if spec[1] == "color":
                norm[p] = _color(raw, f"{name}[{i}].{p}")
            elif isinstance(spec[1], tuple):
                if raw not in spec[1]:
                    _bad(f"{name}[{i}].{p} must be one of {spec[1]}")
                norm[p] = raw
            else:
                norm[p] = _num(raw, f"{name}[{i}].{p}", spec[1], spec[2], integer=isinstance(spec[0], int))
        for p in e:
            if p != "type" and p not in params:
                _bad(f"{name}[{i}]: unknown parameter {p!r} for {e['type']}")
        out.append(norm)
    return out


def _clip(v, name, comp):
    if v is None:
        return None
    if not isinstance(v, dict) or not v:
        _bad(f"{name} must be an object with top/bottom/left/right")
    out = {}
    for k in v:
        if k not in ("top", "bottom", "left", "right"):
            _bad(f"{name}.{k}: unknown edge")
        out[k] = _num(v[k], f"{name}.{k}", -20000, 20000)
    left, right = out.get("left", 0), out.get("right", comp["width"])
    top, bottom = out.get("top", 0), out.get("bottom", comp["height"])
    if not (right > left and bottom > top):
        _bad(f"{name}: empty clip rectangle")
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _stroke(v, name, max_w=50):
    if v is None:
        return None
    if not isinstance(v, dict):
        _bad(f"{name} must be an object")
    return {"color": _color(v.get("color", "#FFFFFF"), name + ".color"),
            "width": _num(v.get("width", 1), name + ".width", 0, max_w),
            "over_fill": bool(v.get("over_fill", False))}


def validate_settings(settings, asset_names):
    if not isinstance(settings, dict):
        _bad("settings must be an object")
    comp = _composition(settings.get("composition"))
    raw_layers = settings.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers or len(raw_layers) > MAX_LAYERS:
        _bad(f"settings.layers must be a list of 1..{MAX_LAYERS} layers (bottom first)")
    layers, ids = [], set()
    fps, frames = comp["fps"], comp["frames"]
    for i, spec in enumerate(raw_layers):
        name = f"layers[{i}]"
        if not isinstance(spec, dict):
            _bad(f"{name} must be an object")
        lid = s1._str(spec.get("id"), name + ".id", _ID_RE)
        if lid in ids:
            _bad(f"duplicate layer id {lid!r}")
        ids.add(lid)
        kind = spec.get("kind")
        if kind not in ("text", "shape", "image"):
            _bad(f"{name}.kind must be text|shape|image")
        L = {"id": lid, "kind": kind}
        in_f = _frame(spec.get("in_f"), name + ".in_f", fps, frames, spec, "in_s")
        out_f = _frame(spec.get("out_f"), name + ".out_f", fps, frames, spec, "out_s")
        L["in_f"] = 0 if in_f is None else in_f
        L["out_f"] = frames if out_f is None else out_f
        if not L["out_f"] > L["in_f"]:
            _bad(f"{name}: out_f ({L['out_f']}) must be after in_f ({L['in_f']})")
        L["in_s"], L["out_s"] = round(L["in_f"] / fps, 6), round(L["out_f"] / fps, 6)
        L["position"] = s1._pair(spec.get("position", [comp["width"] / 2, comp["height"] / 2]), name + ".position", -20000, 20000)
        L["anchor"] = _anchor(spec.get("anchor"), name + ".anchor", kind)
        L["scale"] = s1._pair(spec.get("scale", [100, 100]), name + ".scale", 1, 1000)
        L["rotation"] = _num(spec.get("rotation", 0), name + ".rotation", -36000, 36000)
        L["opacity"] = _num(spec.get("opacity", 100), name + ".opacity", 0, 100)
        blend = spec.get("blend", "normal")
        if blend not in BLENDS:
            _bad(f"{name}.blend must be one of {sorted(BLENDS)}")
        L["blend"] = blend
        L["motion_blur"] = bool(spec.get("motion_blur", False))
        L["clip"] = _clip(spec.get("clip"), name + ".clip", comp)
        matte = spec.get("matte")
        if matte is not None:
            if not isinstance(matte, dict) or not isinstance(matte.get("layer"), str):
                _bad(f"{name}.matte must be {{layer, mode, keep_matte_visible}}")
            if matte.get("mode", "alpha") not in MATTE_MODES:
                _bad(f"{name}.matte.mode must be one of {sorted(MATTE_MODES)}")
            L["matte"] = {"layer": matte["layer"], "mode": matte.get("mode", "alpha"),
                          "keep_matte_visible": bool(matte.get("keep_matte_visible", True))}
        L["effects"] = _effects(spec.get("effects"), name + ".effects")
        span = L["out_f"] - L["in_f"]
        fi = _frame(spec.get("fade_in_f"), name + ".fade_in_f", fps, span, spec, "fade_in_s") or 0
        fo = _frame(spec.get("fade_out_f"), name + ".fade_out_f", fps, span, spec, "fade_out_s") or 0
        if fi + fo > span:
            _bad(f"{name}: fades ({fi}+{fo} frames) exceed the layer's {span} frames")
        L["keyframes"] = _keyframes(spec.get("keyframes"), name + ".keyframes", L, comp, L["effects"])
        if (fi or fo) and "opacity" not in L["keyframes"]:
            keys = []
            if fi:
                keys += [{"f": L["in_f"], "v": 0.0}, {"f": L["in_f"] + fi, "v": L["opacity"]}]
            else:
                keys += [{"f": L["in_f"], "v": L["opacity"]}]
            if fo:
                keys += [{"f": L["out_f"] - fo, "v": L["opacity"]}, {"f": L["out_f"], "v": 0.0}]
            seen = set()
            keys = [k for k in keys if not (k["f"] in seen or seen.add(k["f"]))]
            L["keyframes"]["opacity"] = [dict(k, t=round(k["f"] / fps, 6), ease="linear") for k in keys]

        if kind == "text":
            text = spec.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > s1.MAX_TEXT_CHARS:
                _bad(f"{name}.text must be a non-empty string of at most {s1.MAX_TEXT_CHARS} characters")
            if any(ord(c) < 32 and c != "\n" for c in text):
                _bad(f"{name}.text contains control characters")
            L["text"] = text
            L["font"] = s1._str(spec.get("font"), name + ".font", s1._FONT_RE)
            L["size"] = _num(spec.get("size", 72), name + ".size", 4, 2000)
            L["fill"] = spec.get("fill", True) is not False
            L["color"] = _color(spec.get("color", "#FFFFFF"), name + ".color")
            L["stretch"] = s1._pair(spec.get("stretch", [1, 1]), name + ".stretch", 0.1, 10)
            L["stroke"] = _stroke(spec.get("stroke"), name + ".stroke")
            if not L["fill"] and not L["stroke"]:
                _bad(f"{name}: text with fill false needs a stroke")
            just = spec.get("justify", "center")
            if just not in ("left", "center", "right"):
                _bad(f"{name}.justify must be left|center|right")
            L["justify"] = just
            for k, lo, hi in (("tracking", -500, 500), ("leading", 0, 5000), ("baseline_shift", -2000, 2000)):
                if spec.get(k) is not None:
                    L[k] = _num(spec[k], f"{name}.{k}", lo, hi)
            if spec.get("box") is not None:
                b = spec["box"]
                if not isinstance(b, dict):
                    _bad(f"{name}.box must be {{width, height}}")
                L["box"] = {"width": _num(b.get("width"), name + ".box.width", 1, 20000),
                            "height": _num(b.get("height"), name + ".box.height", 1, 20000)}
        elif kind == "shape":
            sh = spec.get("shape", "rect")
            if sh not in ("rect", "ellipse"):
                _bad(f"{name}.shape must be rect|ellipse")
            L["shape"] = sh
            L["size"] = s1._pair(spec.get("size"), name + ".size", 1, 20000)
            L["radius"] = _num(spec.get("radius", 0), name + ".radius", 0, 10000)
            L["color"] = None if spec.get("color", "#000000") is None else _color(spec.get("color", "#000000"), name + ".color")
            L["stroke"] = _stroke(spec.get("stroke"), name + ".stroke", max_w=500)
            if L["color"] is None and not L["stroke"]:
                _bad(f"{name}: shape needs a fill colour or a stroke")
        else:
            asset = s1._str(spec.get("asset"), name + ".asset", s1._NAME_RE)
            if asset not in asset_names:
                _bad(f"{name}.asset {asset!r} is not in the request's assets manifest")
            L["asset"] = asset
            if spec.get("size") is not None:
                L["size"] = s1._pair(spec["size"], name + ".size", 1, 20000)
            fit = spec.get("fit", "contain")
            if fit not in ("contain", "stretch"):
                _bad(f"{name}.fit must be contain|stretch")
            L["fit"] = fit
        layers.append(L)

    for L in layers:
        if L.get("matte"):
            if L["matte"]["layer"] not in ids or L["matte"]["layer"] == L["id"]:
                raise AEError("MATTE_TARGET_MISSING", f"layer {L['id']!r}: matte.layer {L['matte']['layer']!r} is not another layer")
    extras = settings.get("output_extras") or {}
    proof = []
    if extras:
        if not isinstance(extras, dict):
            _bad("settings.output_extras must be an object")
        pf = extras.get("proof_frames_f") or []
        if not isinstance(pf, list) or len(pf) > CAPABILITIES["bounds"]["max_proof_frames"]:
            _bad(f"output_extras.proof_frames_f: at most {CAPABILITIES['bounds']['max_proof_frames']} frames")
        proof = sorted({_num(f, "proof_frames_f[]", 0, frames - 1, integer=True) for f in pf})
    return {"composition": comp, "layers": layers, "output_extras": {"proof_frames_f": proof}}


def text_layers(settings):
    return [L for L in settings["layers"] if L["kind"] == "text"]


def verify_inspect(settings, comp_summary):
    """Problems between the request and the reopened project's layer summary."""
    problems = []
    by_name = {L.get("name"): L for L in (comp_summary.get("layers") or [])}
    # clipped layers are precomposed: the outer layer is named "<id>" and holds a mask
    for L in settings["layers"]:
        got = by_name.get(L["id"])
        if not got:
            problems.append(f"layer {L['id']!r} missing")
            continue
        if L.get("clip"):
            if got.get("kind") != "precomp" or not got.get("masks"):
                problems.append(f"layer {L['id']!r}: expected a masked precomp for clip, got {got.get('kind')} masks={got.get('masks')}")
            inner = (got.get("inner") or {})
            if L["kind"] == "text" and inner.get("kind") != "text":
                problems.append(f"layer {L['id']!r}: clipped precomp does not hold the text layer")
            src = inner if inner else got
        else:
            src = got
            if got.get("kind") != {"image": "footage"}.get(L["kind"], L["kind"]):
                problems.append(f"layer {L['id']!r}: kind {got.get('kind')} != {L['kind']}")
        if L["kind"] == "text":
            if (src.get("text") or "").replace("\r", "\n") != L["text"]:
                problems.append(f"layer {L['id']!r}: text differs ({src.get('text')!r})")
            if src.get("font") != L["font"]:
                problems.append(f"layer {L['id']!r}: font {src.get('font')} != {L['font']}")
            st = src.get("stretch")
            if st and (abs(st[0] - L["stretch"][0]) > 0.01 or abs(st[1] - L["stretch"][1]) > 0.01):
                problems.append(f"layer {L['id']!r}: stretch {st} != {L['stretch']}")
            if bool(L.get("stroke")) != bool(src.get("stroke")):
                problems.append(f"layer {L['id']!r}: stroke {'missing' if L.get('stroke') else 'unexpected'}")
        want_fx = [e["match_name"] for e in L["effects"]] + (["ADBE Exposure2"] if "brightness" in L["keyframes"] else [])
        got_fx = [e for e in (src.get("effects") or []) if e != "ADBE Effect Built In Params"]
        if got_fx[:len(want_fx)] != want_fx:
            problems.append(f"layer {L['id']!r}: effects {got_fx} != {want_fx}")
        if L.get("matte"):
            if not got.get("matte") or got["matte"].get("layer") != L["matte"]["layer"]:
                problems.append(f"layer {L['id']!r}: matte {got.get('matte')} != {L['matte']}")
        if L["motion_blur"] and not got.get("motion_blur"):
            problems.append(f"layer {L['id']!r}: motion blur switch off")
        for prop, keys in L["keyframes"].items():
            n = (got.get("keys") or {}).get(prop)
            if n is None and src is not got:
                n = (src.get("keys") or {}).get(prop)
            if n != len(keys):
                problems.append(f"layer {L['id']!r}: {prop} has {n} keys, expected {len(keys)}")
        if abs((got.get("in_s") or 0) - L["in_s"]) > 0.5 / settings["composition"]["fps"] or \
           abs((got.get("out_s") or 0) - L["out_s"]) > 0.5 / settings["composition"]["fps"]:
            problems.append(f"layer {L['id']!r}: timing {got.get('in_s')}..{got.get('out_s')} != {L['in_s']}..{L['out_s']}")
    mb = settings["composition"]["motion_blur"]
    if mb["enabled"] and not comp_summary.get("motion_blur"):
        problems.append("comp motion blur off")
    if mb["enabled"] and comp_summary.get("shutter_angle") is not None and abs(comp_summary["shutter_angle"] - mb["shutter_angle"]) > 0.5:
        problems.append(f"comp shutter angle {comp_summary.get('shutter_angle')} != {mb['shutter_angle']}")
    return problems
