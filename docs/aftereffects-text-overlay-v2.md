# text_overlay_v2: request contract (draft for laptop review, 2026-09-11)

Second recipe of the `aftereffects` engine. Same envelope as v1
(`params.aftereffects`, `schema_version: 1`, `recipe: "text_overlay_v2"`,
`composition`, `assets[]`, `output_profile`, `org_id`, `job_id`, `order_id`,
`version`, optional `recipe_revision`); only `settings` differs.
`text_overlay_v1` and its pin `69d6535b68b6` are untouched. Same five
outputs, same alpha/editability/provenance/scope/cancellation contract, same
error codes plus the ones listed at the end.

Everything below was probed on the host (AE 26.5.0.89, 2026-09-11) before
being written down: SVG imports natively as footage, `Montserrat-ExtraBold`
resolves (installed per-user from the OFL Montserrat 7.222 static TTFs),
TextDocument exposes stroke and independent horizontal/vertical scale,
`setTrackMatte` and masks and `precompose` work, Glow / Bevel Alpha /
Exposure / Fill / Tint / Gaussian Blur / Simple Choker / Roughen Edges /
Motion Blur effects exist, comp shutter angle/phase/samples are scriptable,
temporal easing is settable per key. Not available: Inner Shadow, Levels,
Hue/Saturation, Directional Blur, Pixel Motion Blur.

## settings

```json
{
  "composition": {
    "width": 1080, "height": 1920, "fps": 30, "duration_s": 16,
    "motion_blur": {"enabled": true, "shutter_angle": 180, "shutter_phase": -90,
                    "samples_per_frame": 16, "adaptive_sample_limit": 128}
  },
  "layers": [ ...bottom-up z order, see below... ]
}
```

`layers` replaces v1's three lists: one ordered array, first entry at the
bottom. Every layer:

| field | rule |
| --- | --- |
| `id` | `^[a-z][a-z0-9_]{0,31}$`, unique; becomes the AE layer name |
| `kind` | `text` \| `shape` \| `image` |
| `in_f`, `out_f` | integer frames, `0 <= in_f < out_f <= frames`; hard cuts land on these frames exactly. `in_s`/`out_s` (seconds) accepted instead; frames win if both given |
| `position` | `[x, y]` comp px of the layer's **anchor** (see `anchor`) |
| `anchor` | `{"x": "left"\|"center"\|"right", "y": "top"\|"center"\|"bottom"\|"baseline"}` (baseline: text only). Resolved from the layer's own bounds at `in_f` (AE `sourceRectAtTime`), so `position` + `anchor: top/center` puts the visual top-centre of the stretched word at `position`. Default: text `center/baseline`, shape/image `center/center`. Numeric px offsets in layer space are also accepted (`{"x": 12.5, "y": -40}`) |
| `scale` | `[sx, sy]` percent, independent (layer transform). Default `[100, 100]` |
| `rotation` | degrees, default 0 |
| `opacity` | 0..100 |
| `blend` | `normal` \| `multiply` \| `screen` \| `add` \| `overlay` \| `lighten` \| `darken` |
| `motion_blur` | bool; the layer's motion-blur switch (comp `motion_blur.enabled` must be true for it to render; `Best Settings` renders "On for checked layers") |
| `clip` | `{"top": px, "bottom": px, "left": px, "right": px}` in **comp** coordinates (any subset). Implemented as a rectangular mask on the layer after it is precomposed at comp size, so keyframed motion inside the layer never moves the clip. Alpha-preserving: outside is transparent, nothing opaque is added |
| `matte` | `{"layer": "<id>", "mode": "alpha"\|"alpha_inverted"\|"luma"\|"luma_inverted", "keep_matte_visible": true}` : this layer is masked by another layer's alpha/luma (AE track matte). With `keep_matte_visible` the matte layer still renders itself, which is how texture-inside-glyphs works: the texture layer is matted by the visible text layer |
| `effects` | ordered list, max 8, see Effects |
| `keyframes` | `{ "<property>": [ {"f": frame, "v": value, "ease": ...}, ... ] }`, see Keyframes |
| `fade_in_f`, `fade_out_f` | frame counts; shorthand that generates opacity keys (kept from v1 as `fade_in_s`/`fade_out_s` too) |

### text

| field | rule |
| --- | --- |
| `text` | 1..500 chars, `\n` for line breaks |
| `font` | PostScript name; must be in the host inventory (`docs/aftereffects-capabilities.json`) or the job fails `FONT_MISSING` before AE launches; AE re-checks; **no substitution** |
| `size` | px, 4..2000 |
| `color` | `#RRGGBB` fill; `fill: false` for stroke-only text |
| `stretch` | `[hx, vy]` factors applied as TextDocument horizontal/vertical scale (1 = none, 0.1..10): the glyphs themselves are stretched, the text stays editable; the approved 1.38 x 4.8 words are this, not `scale` |
| `stroke` | `{"color": "#RRGGBB", "width": px 0..50, "over_fill": false}`. Width 0 is accepted but After Effects stores its own minimum, 0.01 px, and the manifest reports that value; omit `stroke` entirely for no stroke |
| `justify` | `left` \| `center` \| `right` (horizontal anchor of a point-text line) |
| `tracking`, `leading`, `baseline_shift` | numbers (AE units) |
| `box` | optional `{"width": px, "height": px}` turns the layer into box text (wrapping) |

### shape

`kind: "shape"`, `shape: "rect" | "ellipse"`, `size: [w, h]`, `radius`
(rect corners), `color` (fill or `null`), `stroke: {color, width}`. Anchor at
the geometric centre unless `anchor` says otherwise.

### image

`kind: "image"`, `asset: "<name from assets[]>"`. PNG/JPG/WebP/SVG/MOV/MP4
footage. **SVG never reaches After Effects**: AE 26.5 accepts SVG footage
but crashed (heap corruption) on the second wear-texture import, so the
worker rasterizes every SVG to a transparent PNG at the SVG's declared
width/height (viewBox fallback) with Chrome headless before authoring, and
records it in `provenance.inputs.<name>.rasterized_from`. The approved wear
SVGs (1080x1400) come out as 1080x1400 RGBA PNGs in ~1.5 s. `size` optional
`[w, h]` px to fit (uniform unless `fit: "stretch"`).

Assets arrive from content-addressed storage (`assets/sha256/<hex>`, no
suffix), so the worker types every asset by content, never by name
(`worker/aftereffects/staging.py`): an SVG is recognised by its document
prolog + `<svg` root (UTF-8, optional BOM/XML prolog/comments/DOCTYPE,
within the first 8 KB) and must declare a width/height or viewBox; PNG /
JPEG / WebP / video / audio by magic bytes plus an ffprobe decode. Bytes that
match neither, or that match a different kind than declared, fail with
`ASSET_INVALID` before After Effects launches (job 8ed3206f failed here on
2026-09-11 because SVG went through the raster sniffer; fixed in the
worker, storage contract unchanged).

### Effects (built-in AE effects only, in the order given)

| `type` | params | AE effect |
| --- | --- | --- |
| `glow` | `color` `#RRGGBB`, `radius` px 0..500, `intensity` 0..4, `threshold` 0..100 (default 0), `based_on` `alpha`\|`color`, `composite` `on_top`\|`behind` | Glow (`ADBE Glo2`) with A&B colours set to `color`; stack several for the 10px + 24px halos |
| `bevel_highlight` | `thickness` px, `angle` deg, `color`, `intensity` 0..1 | Bevel Alpha (the "inset 0 2px #fff7" highlight approximation; documented as such) |
| `drop_shadow` | `color`, `opacity`, `distance`, `softness`, `direction` | Drop Shadow |
| `gaussian_blur` | `radius` | Gaussian Blur |
| `fill` | `color`, `opacity` | Fill (recolour a texture inside a matte) |
| `tint` | `black`, `white`, `amount` | Tint |
| `choker` | `amount` px (negative spreads) | Simple Choker (edge tightening for matted textures) |
| `exposure` | `stops` | Exposure (static; keyframed brightness uses the `brightness` keyframe property below) |
| `roughen_edges` | `border`, `scale`, `complexity`, `seed` | Roughen Edges (optional printed-ink edge) |

Anything else fails `EFFECT_MISSING`. Third-party plugins are never used.

### Keyframes

`keyframes: { "<property>": [ {"f": 0, "v": [103, 103], "ease": "ease_out"}, {"f": 6, "v": [100, 100], "ease": "hold"} ] }`

| property | value | notes |
| --- | --- | --- |
| `position` | `[x, y]` | comp px of the anchor |
| `scale` | `[sx, sy]` | percent |
| `rotation` | deg | |
| `opacity` | 0..100 | |
| `brightness` | multiplier, 0.1..4 | one Exposure effect is added; exposure = log2(v) in stops, so 1.12 is exactly +12 % linear light. Used for the lamp pulse: `[{f:0,v:1},{f:2,v:1.12},{f:8,v:1}]` |
| `stretch` | `[hx, vy]` | text only, via Source Text keys (hold interpolation; AE cannot tween text properties) |
| `effects.<index>.<param>` | number | e.g. `effects.0.intensity` keyframes on the first glow |

Frames are integers within the layer's `in_f..out_f`; at most 200 keys per
property. `ease` per key: `linear` (default), `hold` (value jumps at this key
and holds until the next: the hard cuts), `ease_in`, `ease_out`, `ease_in_out`
(speed 0, influence 33.3 on the eased side), or explicit
`{"in": [speed, influence], "out": [speed, influence]}` (influence 0.1..100).
`"ease_out"` on the first key of the 103→100 settle plus `"hold"` on the
last key gives "settle, then sharp hold, no overshoot"; the manifest reports
the resulting AE ease values per key so the platform can pin them. A CSS
cubic-bezier is not reproduced exactly; state it as `{"in": …, "out": …}` if
the approximation matters.

### Host capability report

`python -m aftereffects.capabilities --json` writes
`docs/aftereffects-capabilities.json` (committed on every recipe change):

```json
{"host": {"kind": "aftereffects", "version": "26.5.0.89", "os": "Windows 11"},
 "fonts": ["Arial-BoldMT", "ArialMT", "Montserrat-Bold", "Montserrat-ExtraBold", "SegoeUI", "..."],
 "recipes": {"text_overlay_v1": {"revision": "69d6535b68b6", "features": [...]},
             "text_overlay_v2": {"revision": "<12 hex>", "features": ["stretch", "stroke", "matte", "clip", "glow", ...],
                                 "effects": ["glow", "bevel_highlight", ...], "keyframe_properties": [...],
                                 "bounds": {"max_layers": 60, "max_keys_per_property": 200, ...}}},
 "svg_import": true, "motion_blur": true, "generated_at": "..."}
```

The platform must preflight fonts and features against this file (or the
same JSON returned by the worker in every manifest under
`provenance.capabilities_sha256`) before recommending a design.

## Example: approved r9 lettering scene (colour word BLUE)

```json
{
  "schema_version": 1, "recipe": "text_overlay_v2", "composition": "Main",
  "org_id": "...", "job_id": "...", "order_id": "e506f8ec-...", "version": 1,
  "assets": [
    {"name": "wear_light", "bucket": "assets", "path": "sha256/...", "sha256": "...", "kind": "image"}
  ],
  "settings": {
    "composition": {"width": 1080, "height": 1920, "fps": 30, "duration_s": 16,
                    "motion_blur": {"enabled": true, "shutter_angle": 180, "shutter_phase": -90}},
    "layers": [
      {"id": "word_blue", "kind": "text", "text": "BLUE", "font": "Montserrat-ExtraBold", "size": 231,
       "color": "#30579A", "stretch": [1.38, 4.8], "justify": "center",
       "position": [540, 270], "anchor": {"x": "center", "y": "top"},
       "in_f": 88, "out_f": 132, "clip": {"bottom": 1250}},
      {"id": "wear_blue", "kind": "image", "asset": "wear_light", "position": [540, 700],
       "in_f": 88, "out_f": 132, "blend": "normal",
       "matte": {"layer": "word_blue", "mode": "alpha", "keep_matte_visible": true}},
      {"id": "lamp_red", "kind": "shape", "shape": "ellipse", "size": [58, 52], "color": "#D82026",
       "position": [964, 455], "in_f": 0, "out_f": 387, "motion_blur": false,
       "effects": [{"type": "glow", "color": "#D82026", "radius": 10, "intensity": 0.62},
                   {"type": "glow", "color": "#D82026", "radius": 24, "intensity": 0.22},
                   {"type": "bevel_highlight", "thickness": 2, "angle": 90, "color": "#FFFFFF", "intensity": 0.47}],
       "keyframes": {"brightness": [{"f": 0, "v": 1.0}, {"f": 2, "v": 1.12}, {"f": 8, "v": 1.0}]}},
      {"id": "num_3", "kind": "text", "text": "3", "font": "Montserrat-ExtraBold", "size": 680, "color": "#D3DAE0",
       "position": [315, 505], "anchor": {"x": "center", "y": "top"}, "in_f": 0, "out_f": 27, "motion_blur": true,
       "keyframes": {"scale": [{"f": 0, "v": [103, 103], "ease": "ease_out"}, {"f": 6, "v": [100, 100], "ease": "hold"}]}}
    ]
  }
}
```

The `clip` on `word_blue` also clips the matted `wear_blue` because the wear
only exists where the text's alpha is. Note that the wear asset is matted by
the text, so its own `position`/`size` place the texture, not the word.

## Verification additions (v2)

Beyond v1's checks, the reopen step confirms per layer: kind, font,
stretch, stroke, matte target + mode, mask presence, effect match names in
order, keyframe count per property, motion-blur switch, comp shutter
settings. The manifest carries all of it under `inspect.layers[]`, plus
`easing_resolved` (the AE speed/influence per key). A contact sheet at the
proof frame times can be requested with `output_extras.proof_frames_f: [...]`
(max 12; each also uploaded as `outputs/<id>-proof-<fffff>.png`, the frame
number zero-padded to five digits: frame 42 is `<id>-proof-00042.png`).

## New error codes

`ASSET_INVALID` (downloaded bytes are not a decodable file of the declared
kind; `ASSET_MISSING` now also covers a failed fetch from storage),
`MATTE_TARGET_MISSING` (matte.layer not in `layers`), `EFFECT_MISSING`
(reused), `KEYFRAME_INVALID` (frame outside the layer, unknown property),
`ANCHOR_UNRESOLVED` (empty bounds at `in_f`, e.g. empty text).

## Open points for the laptop before implementation starts

1. `layers[]` single ordered array (v2) vs three lists (v1): confirm.
2. Frames (`in_f`/`out_f`, key `f`) as the primary time unit: confirm.
3. Easing: named presets + explicit `{in, out}` speed/influence; no
   cubic-bezier field. Confirm or ask for a bezier field with a documented
   approximation.
4. Lamp geometry in the example is illustrative; the approved values
   (housing x914 y408 100x244 r24 #212326, lamps x930 y424/498/572 68x62
   with 5 px border) map to shapes with `stroke` for the dark border.
5. Wear textures: native SVG import first; say if you would rather ship
   PNG rasters as the authorized matte assets.
