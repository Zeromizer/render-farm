"""Machine-readable host + recipe capability report for the platform's preflight.

  ..\\.venv\\Scripts\\python.exe -m aftereffects.capabilities            # print JSON
  ..\\.venv\\Scripts\\python.exe -m aftereffects.capabilities --write    # also docs/aftereffects-capabilities.json

Fonts come from the Windows font tables (PostScript names), the AE version
from AfterFX.exe, recipe revisions from the recipe files. Feature lists are
declared next to the recipes so a change there changes the report.
"""
import datetime
import hashlib
import json
import os
import platform
import sys

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from aftereffects import ae_host, fonts, media, schema  # noqa: E402
from aftereffects.pipeline import recipe_revision  # noqa: E402

RECIPE_FEATURES = {
    "text_overlay_v1": {
        "layers": ["text", "shape", "image"],
        "features": ["fade_in_s", "fade_out_s", "slide_from", "uniform_scale", "scale_from", "drop_shadow",
                     "text_tracking", "justify"],
        "effects": ["drop_shadow"],
        "keyframe_properties": [],
        "time_unit": "seconds",
        "bounds": {"max_layers": schema.MAX_LAYERS, "max_texts": schema.MAX_TEXTS, "max_shapes": schema.MAX_SHAPES,
                   "max_images": schema.MAX_IMAGES, "max_assets": schema.MAX_ASSETS, "max_text_chars": schema.MAX_TEXT_CHARS},
    },
}
try:  # v2 declares itself when present
    from aftereffects import schema_v2  # noqa: E402
    RECIPE_FEATURES["text_overlay_v2"] = schema_v2.CAPABILITIES
except ImportError:
    pass


def report():
    install = ae_host.find_after_effects()
    names = sorted(fonts.installed_postscript_names(refresh=True))
    recipes = {}
    for name, feats in RECIPE_FEATURES.items():
        if name in schema.RECIPES:
            recipes[name] = dict(feats, revision=recipe_revision(name))
    r = {
        "host": {"kind": "aftereffects" if install else "none",
                 "version": install["version"] if install else None,
                 "dir": install["dir"] if install else None,
                 "os": f"{platform.system()} {platform.release()}",
                 "ffmpeg": media.version()},
        "fonts": names,
        "recipes": recipes,
        "svg_import": True,      # probed on AE 26.5: ImportOptions.canImportAs(FOOTAGE) is true for SVG
        "motion_blur": True,
        "output_profiles": sorted(schema.OUTPUT_PROFILES),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    body = json.dumps({k: v for k, v in r.items() if k != "generated_at"}, sort_keys=True).encode()
    r["capabilities_sha256"] = hashlib.sha256(body).hexdigest()
    return r


def main():
    r = report()
    text = json.dumps(r, indent=1)
    if "--write" in sys.argv:
        out = os.path.join(os.path.dirname(_WORKER_DIR), "docs", "aftereffects-capabilities.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {out}")
    print(text)


if __name__ == "__main__":
    main()
