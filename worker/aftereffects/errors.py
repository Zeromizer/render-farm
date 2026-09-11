"""Error codes the aftereffects engine reports. The row's `error` column
carries "<CODE>: message" so the platform can branch on the prefix."""

CODES = (
    "AE_NOT_INSTALLED",      # no AfterFX.exe / aerender.exe on this host
    "AE_SLOT_TIMEOUT",       # another AE job held the single slot too long
    "AE_NOT_ISOLATED",       # the script landed in an instance with an open project
    "AE_NO_MANIFEST",        # AE exited without writing the manifest (file access pref?)
    "INVALID_REQUEST",       # schema validation failed
    "RECIPE_REVISION_MISMATCH",
    "ASSET_MISSING",
    "ASSET_HASH_MISMATCH",
    "ASSET_INVALID",         # downloaded bytes are not a decodable file of the declared kind
    "FONT_MISSING",
    "EFFECT_MISSING",
    "OM_TEMPLATE_MISSING",
    "JSX_ERROR",             # the recipe threw (line/file/message attached)
    "RENDER_FAILED",         # aerender exited nonzero / produced nothing
    "RENDER_INCOMPLETE",     # frame count or duration off
    "ALPHA_MISSING",         # output carries no alpha channel / alpha is flat
    "VERIFY_FAILED",         # dimensions/fps/layers do not match the request
    "CANCELED",
    "TIMEOUT",
    "CLAIM_SUPERSEDED",      # the row was reclaimed by another attempt
    # text_overlay_v2
    "MATTE_TARGET_MISSING",  # matte.layer is not another layer of the request
    "KEYFRAME_INVALID",      # key outside the layer's frames / unknown property
    "ANCHOR_UNRESOLVED",     # layer has no bounds at its in frame (empty text)
)


class AEError(RuntimeError):
    def __init__(self, code, message, detail=None):
        if code not in CODES:
            raise ValueError(f"unknown AE error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self):
        d = {"code": self.code, "message": self.message}
        d.update(self.detail)
        return d
