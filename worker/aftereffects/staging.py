"""Asset staging for the aftereffects engine: download each request asset
into the attempt's inputs/ directory and give it the suffix its CONTENT earns.

Storage objects are content-addressed (assets/sha256/<hex>, no suffix), so the
name tells nothing. The video_gen sniffer (videogen/media_type.py) recognises
raster/video/audio magic and ffprobes the file; it has no notion of SVG, and
ffmpeg cannot decode SVG, so an SVG wear texture died there with a bare
RuntimeError before pipeline._stage_assets could rasterize it (job 8ed3206f,
2026-09-11). SVG is therefore recognised here, by content, before that sniffer
runs; everything else still goes through it unchanged.

Failure codes (aftereffects/errors.py): ASSET_MISSING when the object cannot
be fetched, ASSET_INVALID when the bytes are not a decodable file of the
declared kind (a video where an image was declared, an SVG declared as video,
an unrecognised or truncated file).

Runner and smoke share this module so `aftereffects/smoke.py --asset` walks
the exact path a farm job walks (bare name, no suffix, sniffed).
"""
import os

from aftereffects import media
from aftereffects.errors import AEError

ASSET_KINDS = {"image": ("image",), "video": ("video",), "audio": ("audio",)}


def typed_asset(path, kind, name):
    """Return `path` renamed with the suffix its content earns, validated as
    `kind` (image | video | audio). SVG (image only) gets .svg and no ffprobe;
    pipeline._stage_assets rasterizes it before After Effects sees it."""
    from videogen import media_type

    if kind not in ASSET_KINDS:
        raise AEError("INVALID_REQUEST", f"asset {name!r}: unknown kind {kind!r}")
    if media.is_svg(path):
        if kind != "image":
            raise AEError("ASSET_INVALID", f"asset {name!r}: SVG given where {kind} is expected")
        w, h = media.svg_size(path, default=(0, 0))
        if w <= 0 or h <= 0:
            raise AEError("ASSET_INVALID", f"asset {name!r}: SVG declares no usable width/height or viewBox")
        target = path if path.lower().endswith(".svg") else os.path.splitext(path)[0] + ".svg"
        if os.path.abspath(target) != os.path.abspath(path):
            if os.path.exists(target):
                os.remove(target)
            os.replace(path, target)
        return target
    try:
        return media_type.ensure_extension(path, ASSET_KINDS[kind], name=name)
    except RuntimeError as e:
        msg = str(e)
        prefix = f"video_gen input {name}: "
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
        raise AEError("ASSET_INVALID", f"asset {name!r} ({kind}): {msg}")


def stage_assets(request, inputs_dir, log, download):
    """{name: local path} for every request asset. `download(bucket, path,
    inputs_dir, name, log)` must return the local file it wrote."""
    local = {}
    for a in request["assets"]:
        try:
            p = download(a["bucket"], a["path"], inputs_dir, a["name"], log)
        except AEError:
            raise
        except Exception as e:   # storage / network errors: not the request's fault, retryable
            raise AEError("ASSET_MISSING", f"asset {a['name']!r}: {a['bucket']}/{a['path']} could not be fetched: {e}")
        local[a["name"]] = typed_asset(p, a["kind"], a["name"])
    return local
