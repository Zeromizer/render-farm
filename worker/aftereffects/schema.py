"""Request schema for engine "aftereffects", schema_version 1.

The request lives at params.aftereffects on the farm_render_jobs row:

  {
    "schema_version": 1,
    "recipe": "text_overlay_v1",
    "recipe_revision": "9f3c2a1b0d4e",          # optional pin; mismatch fails
    "composition": "Main",
    "settings": { ...recipe settings, see TEXT_OVERLAY_V1 below... },
    "assets": [ {"name": "logo", "bucket": "assets", "path": "sha256/<hex>",
                 "sha256": "<hex>", "size": 1234, "kind": "image"} ],
    "output_profile": "prores4444_alpha",
    "org_id": "<uuid>", "job_id": "<uuid>",     # platform scope (required)
    "order_id": "<uuid>", "version": 3          # optional, provenance only
  }

Validation here is a data check: it bounds sizes, counts, enums and paths so
the trusted recipe only ever sees well-formed values. It does NOT make
arbitrary JSX safe - only the recipes in recipes/ are ever executed.
"""
import math
import re

from aftereffects.errors import AEError

SCHEMA_VERSION = 1

RECIPES = {
    "text_overlay_v1": {"author": "text_overlay_v1.jsx", "revise": "revise_v1.jsx"},
    # v2: ordered layers, frames, anchors, stretch/stroke, clip, mattes, effects,
    # keyframes (docs/aftereffects-text-overlay-v2.md). Revisions are new jobs.
    "text_overlay_v2": {"author": "text_overlay_v2.jsx", "revise": None},
}

OUTPUT_PROFILES = {
    # master: ProRes 4444 RGB+A .mov; review: VP9 alpha .webm; bundle: .zip
    "prores4444_alpha": {"master_ext": "mov", "master_mime": "video/quicktime",
                         "review_ext": "webm", "review_mime": "video/webm"},
}

FPS_ALLOWED = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)
MAX_LAYERS = 40
MAX_TEXTS, MAX_SHAPES, MAX_IMAGES, MAX_ASSETS = 20, 20, 10, 20
MAX_TEXT_CHARS = 500

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FONT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_. ]{0,62}$")
_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
_UUIDISH_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]{0,255}$")
_OPT_TEXT_RE = re.compile(r"^[A-Za-z0-9 _\-]{1,64}$")


def _bad(msg, **detail):
    raise AEError("INVALID_REQUEST", msg, detail)


def _num(v, name, lo, hi, allow_float=True):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _bad(f"{name} must be a number")
    if not allow_float and int(v) != v:
        _bad(f"{name} must be an integer")
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        _bad(f"{name} must be finite")
    if not (lo <= v <= hi):
        _bad(f"{name} must be within [{lo}, {hi}], got {v}")
    return int(v) if not allow_float else float(v)


def _pair(v, name, lo, hi):
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        _bad(f"{name} must be [x, y]")
    return [_num(v[0], name + "[0]", lo, hi), _num(v[1], name + "[1]", lo, hi)]


def _str(v, name, regex, maxlen=None):
    if not isinstance(v, str):
        _bad(f"{name} must be a string")
    if maxlen is not None and len(v) > maxlen:
        _bad(f"{name} longer than {maxlen} characters")
    if regex and not regex.match(v):
        _bad(f"{name} has an unexpected format: {v[:40]!r}")
    return v


def _color(v, name):
    s = _str(v, name, _HEX_RE)
    return "#" + s.lstrip("#").upper()


def _shadow(v, name):
    if v is None or v is False:
        return None
    if v is True:
        return {"opacity": 50, "distance": 8, "softness": 20}
    if not isinstance(v, dict):
        _bad(f"{name} must be true/false or an object")
    return {"opacity": _num(v.get("opacity", 50), name + ".opacity", 0, 100),
            "distance": _num(v.get("distance", 8), name + ".distance", 0, 200),
            "softness": _num(v.get("softness", 20), name + ".softness", 0, 500)}


def _layer_common(spec, name, comp, out, seen_ids):
    if not isinstance(spec, dict):
        _bad(f"{name} must be an object")
    lid = _str(spec.get("id"), name + ".id", _ID_RE)
    if lid in seen_ids:
        _bad(f"duplicate layer id {lid!r}")
    seen_ids.add(lid)
    out["id"] = lid
    w, h, dur = comp["width"], comp["height"], comp["duration_s"]
    out["position"] = _pair(spec.get("position", [w / 2, h / 2]), name + ".position", -3 * max(w, h), 4 * max(w, h))
    in_s = _num(spec.get("in_s", 0), name + ".in_s", 0, dur)
    out_s = _num(spec.get("out_s", dur), name + ".out_s", 0, dur)
    if not out_s > in_s:
        _bad(f"{name}: out_s ({out_s}) must be after in_s ({in_s})")
    out["in_s"], out["out_s"] = in_s, out_s
    span = out_s - in_s
    out["fade_in_s"] = _num(spec.get("fade_in_s", 0), name + ".fade_in_s", 0, span)
    out["fade_out_s"] = _num(spec.get("fade_out_s", 0), name + ".fade_out_s", 0, span)
    if out["fade_in_s"] + out["fade_out_s"] > span:
        _bad(f"{name}: fades ({out['fade_in_s']}+{out['fade_out_s']}s) exceed the layer's {span:.3f}s on screen")
    out["opacity"] = _num(spec.get("opacity", 100), name + ".opacity", 0, 100)
    if "slide_from" in spec:
        out["slide_from"] = _pair(spec["slide_from"], name + ".slide_from", -4000, 4000)
        out["slide_s"] = _num(spec.get("slide_s", out["fade_in_s"] or 0.4), name + ".slide_s", 0.01, span)
    out["scale"] = _num(spec.get("scale", 100), name + ".scale", 1, 1000)
    if "scale_from" in spec:
        out["scale_from"] = _num(spec["scale_from"], name + ".scale_from", 0, 1000)
        out["scale_s"] = _num(spec.get("scale_s", out["fade_in_s"] or 0.4), name + ".scale_s", 0.01, span)
    sh = _shadow(spec.get("shadow"), name + ".shadow")
    if sh:
        out["shadow"] = sh
    return out


def _composition(v):
    if not isinstance(v, dict):
        _bad("settings.composition must be an object")
    width = _num(v.get("width"), "composition.width", 16, 8192, allow_float=False)
    height = _num(v.get("height"), "composition.height", 16, 8192, allow_float=False)
    if width % 2 or height % 2:
        _bad("composition width/height must be even (VP9 alpha review needs 4:2:0-safe dimensions)")
    fps = _num(v.get("fps"), "composition.fps", 1, 120)
    if not any(abs(fps - a) < 1e-6 for a in FPS_ALLOWED):
        _bad(f"composition.fps must be one of {FPS_ALLOWED}")
    fps = next(a for a in FPS_ALLOWED if abs(fps - a) < 1e-6)
    duration = _num(v.get("duration_s"), "composition.duration_s", 0.1, 120)
    frames = int(round(duration * fps))
    if frames < 1:
        _bad("composition is shorter than one frame")
    return {"width": width, "height": height, "fps": fps, "duration_s": duration, "frames": frames}


def validate_settings_text_overlay_v1(settings, asset_names):
    if not isinstance(settings, dict):
        _bad("settings must be an object")
    comp = _composition(settings.get("composition"))
    seen = set()
    texts, shapes, images = [], [], []

    raw_texts = settings.get("texts") or []
    if not isinstance(raw_texts, list) or len(raw_texts) > MAX_TEXTS:
        _bad(f"settings.texts must be a list of at most {MAX_TEXTS}")
    for i, spec in enumerate(raw_texts):
        name = f"texts[{i}]"
        out = _layer_common(spec, name, comp, {}, seen)
        text = spec.get("text")
        if not isinstance(text, str) or not text.strip():
            _bad(f"{name}.text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            _bad(f"{name}.text longer than {MAX_TEXT_CHARS} characters")
        if any(ord(c) < 32 and c not in "\n" for c in text):
            _bad(f"{name}.text contains control characters")
        out["text"] = text
        out["font"] = _str(spec.get("font"), name + ".font", _FONT_RE)
        out["size"] = _num(spec.get("size", 72), name + ".size", 4, 1000)
        out["color"] = _color(spec.get("color", "#FFFFFF"), name + ".color")
        just = spec.get("justify", "center")
        if just not in ("left", "center", "right"):
            _bad(f"{name}.justify must be left|center|right")
        out["justify"] = just
        if "tracking" in spec:
            out["tracking"] = _num(spec["tracking"], name + ".tracking", -500, 500)
        texts.append(out)

    raw_shapes = settings.get("shapes") or []
    if not isinstance(raw_shapes, list) or len(raw_shapes) > MAX_SHAPES:
        _bad(f"settings.shapes must be a list of at most {MAX_SHAPES}")
    for i, spec in enumerate(raw_shapes):
        name = f"shapes[{i}]"
        out = _layer_common(spec, name, comp, {}, seen)
        kind = spec.get("kind", "rect")
        if kind not in ("rect", "ellipse"):
            _bad(f"{name}.kind must be rect|ellipse")
        out["kind"] = kind
        out["size"] = _pair(spec.get("size"), name + ".size", 1, 8192)
        out["radius"] = _num(spec.get("radius", 0), name + ".radius", 0, 4096)
        out["color"] = _color(spec.get("color", "#000000"), name + ".color")
        shapes.append(out)

    raw_images = settings.get("images") or []
    if not isinstance(raw_images, list) or len(raw_images) > MAX_IMAGES:
        _bad(f"settings.images must be a list of at most {MAX_IMAGES}")
    for i, spec in enumerate(raw_images):
        name = f"images[{i}]"
        out = _layer_common(spec, name, comp, {}, seen)
        asset = _str(spec.get("asset"), name + ".asset", _NAME_RE)
        if asset not in asset_names:
            _bad(f"{name}.asset {asset!r} is not in the request's assets manifest")
        out["asset"] = asset
        images.append(out)

    if len(texts) + len(shapes) + len(images) > MAX_LAYERS:
        _bad(f"more than {MAX_LAYERS} layers")
    return {"composition": comp, "texts": texts, "shapes": shapes, "images": images}


def validate_assets(raw):
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_ASSETS:
        _bad(f"assets must be a list of at most {MAX_ASSETS}")
    out, names = [], set()
    for i, a in enumerate(raw):
        if not isinstance(a, dict):
            _bad(f"assets[{i}] must be an object")
        name = _str(a.get("name"), f"assets[{i}].name", _NAME_RE)
        if name in names:
            _bad(f"duplicate asset name {name!r}")
        names.add(name)
        entry = {"name": name,
                 "bucket": _str(a.get("bucket"), f"assets[{i}].bucket", _NAME_RE),
                 "path": _str(a.get("path"), f"assets[{i}].path", _PATH_RE)}
        if ".." in entry["path"].split("/"):
            _bad(f"assets[{i}].path must not contain '..'")
        kind = a.get("kind", "image")
        if kind not in ("image", "video", "audio"):
            _bad(f"assets[{i}].kind must be image|video|audio")
        entry["kind"] = kind
        if a.get("sha256") is not None:
            entry["sha256"] = _str(str(a["sha256"]).lower(), f"assets[{i}].sha256", _SHA_RE)
        if a.get("size") is not None:
            entry["size"] = _num(a["size"], f"assets[{i}].size", 0, 4 * 1024 ** 3, allow_float=False)
        out.append(entry)
    return out


def validate_request(params):
    """Validate params.aftereffects; return the normalized request."""
    if not isinstance(params, dict):
        _bad("params.aftereffects must be an object")
    ver = params.get("schema_version")
    if ver != SCHEMA_VERSION:
        _bad(f"schema_version must be {SCHEMA_VERSION}, got {ver!r}")
    recipe = params.get("recipe")
    if recipe not in RECIPES:
        _bad(f"recipe must be one of {sorted(RECIPES)}, got {recipe!r}")
    profile = params.get("output_profile", "prores4444_alpha")
    if profile not in OUTPUT_PROFILES:
        _bad(f"output_profile must be one of {sorted(OUTPUT_PROFILES)}")
    req = {
        "schema_version": SCHEMA_VERSION,
        "recipe": recipe,
        "composition": _str(params.get("composition", "Main"), "composition", _OPT_TEXT_RE),
        "output_profile": profile,
        "org_id": _str(params.get("org_id"), "org_id", _UUIDISH_RE),
        "job_id": _str(params.get("job_id"), "job_id", _UUIDISH_RE),
    }
    if params.get("recipe_revision") is not None:
        req["recipe_revision"] = _str(params["recipe_revision"], "recipe_revision", re.compile(r"^[0-9a-f]{12}$"))
    if params.get("order_id") is not None:
        req["order_id"] = _str(params["order_id"], "order_id", _UUIDISH_RE)
    if params.get("version") is not None:
        req["version"] = _num(params["version"], "version", 0, 10 ** 6, allow_float=False)
    req["assets"] = validate_assets(params.get("assets"))
    names = {a["name"] for a in req["assets"]}
    if recipe == "text_overlay_v1":
        req["settings"] = validate_settings_text_overlay_v1(params.get("settings"), names)
    elif recipe == "text_overlay_v2":
        from aftereffects import schema_v2
        req["settings"] = schema_v2.validate_settings(params.get("settings"), names)
    return req


def text_layers(request):
    """[{id, text, font}] for either recipe."""
    st = request["settings"]
    if request["recipe"] == "text_overlay_v2":
        return [{"id": L["id"], "text": L["text"], "font": L["font"]} for L in st["layers"] if L["kind"] == "text"]
    return [{"id": t["id"], "text": t["text"], "font": t["font"]} for t in st["texts"]]


def layer_ids(request):
    st = request["settings"]
    if request["recipe"] == "text_overlay_v2":
        return [L["id"] for L in st["layers"]]
    return [x["id"] for group in ("texts", "shapes", "images") for x in st[group]]


def validate_changes(changes, request):
    """Revision changes for revise_v1.jsx: [{id, text?, in_s?, out_s?}]."""
    if RECIPES[request["recipe"]]["revise"] is None:
        _bad(f"recipe {request['recipe']} has no revision script; submit changed settings as a new job")
    if not isinstance(changes, list) or not changes or len(changes) > MAX_LAYERS:
        _bad("changes must be a non-empty list")
    ids = {t["id"]: "text" for t in request["settings"]["texts"]}
    ids.update({s["id"]: "shape" for s in request["settings"]["shapes"]})
    ids.update({i["id"]: "image" for i in request["settings"]["images"]})
    dur = request["settings"]["composition"]["duration_s"]
    out = []
    for i, ch in enumerate(changes):
        if not isinstance(ch, dict):
            _bad(f"changes[{i}] must be an object")
        lid = _str(ch.get("id"), f"changes[{i}].id", _ID_RE)
        if lid not in ids:
            _bad(f"changes[{i}].id {lid!r} is not a layer of this request")
        entry = {"id": lid}
        if "text" in ch:
            if ids[lid] != "text":
                _bad(f"changes[{i}]: {lid!r} is not a text layer")
            if not isinstance(ch["text"], str) or not ch["text"].strip() or len(ch["text"]) > MAX_TEXT_CHARS:
                _bad(f"changes[{i}].text must be a non-empty string of at most {MAX_TEXT_CHARS} characters")
            entry["text"] = ch["text"]
        if "in_s" in ch:
            entry["in_s"] = _num(ch["in_s"], f"changes[{i}].in_s", 0, dur)
        if "out_s" in ch:
            entry["out_s"] = _num(ch["out_s"], f"changes[{i}].out_s", 0, dur)
        if "in_s" in entry and "out_s" in entry and not entry["out_s"] > entry["in_s"]:
            _bad(f"changes[{i}]: out_s must be after in_s")
        if len(entry) == 1:
            _bad(f"changes[{i}] changes nothing")
        out.append(entry)
    return out
