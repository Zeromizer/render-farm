# aftereffects engine: platform integration contract (v1)

Farm-side implementation: `worker/runners/aftereffects.py` + `worker/aftereffects/`
on branch `feat/aftereffects-engine` of `Zeromizer/render-farm`. Measured on the
render PC with After Effects 26.5 (2026) on 2026-09-11. Nothing in this document
is activated on the shared queue yet; see "Activation" at the end.

## Engine name and row shape

`farm_render_jobs` row:

| column | value |
| --- | --- |
| `engine` | `aftereffects` |
| `repo_url` | `-` (no clone) |
| `git_ref` | `main` (unused) |
| `priority` | `100` like renders (a job blocks the worker ~30 s per 5 s overlay) |
| `timeout_minutes` | `20` suggested (author + render + encode; the slot wait is inside it) |
| `params` | `{"aftereffects": <request>}` |

Only a worker advertising the `aftereffects` capability claims the row
(`claim_farm_job(p_capabilities)`). The claim function deployed today has no
such filter and claims everything, so **do not insert `aftereffects` rows
before the migration below is applied and the new worker code runs**: the
current worker would claim the row and fail it with `unknown engine`.

## Request: `params.aftereffects` (schema_version 1)

Validated by `worker/aftereffects/schema.py` (`validate_request`). Invalid
requests fail fast with `INVALID_REQUEST: <reason>` and nothing runs.

```json
{
  "schema_version": 1,
  "recipe": "text_overlay_v1",
  "recipe_revision": "5b1e0a9c2d44",
  "composition": "Main",
  "output_profile": "prores4444_alpha",
  "org_id": "0f4c…", "job_id": "7a21…", "order_id": "c3d9…", "version": 3,
  "assets": [
    {"name": "logo", "bucket": "assets", "path": "sha256/ab12…", "sha256": "ab12…", "size": 48211, "kind": "image"}
  ],
  "settings": {
    "composition": {"width": 1080, "height": 1920, "fps": 30, "duration_s": 5},
    "shapes": [
      {"id": "pill", "kind": "rect", "size": [720, 150], "radius": 75, "color": "#FF6A00",
       "position": [540, 1500], "in_s": 0.6, "out_s": 5.0, "fade_in_s": 0.3, "fade_out_s": 0.4,
       "scale_from": 60, "scale_s": 0.45}
    ],
    "texts": [
      {"id": "headline", "text": "Motorised Blinds\nFrom $299", "font": "Arial-BoldMT", "size": 110,
       "color": "#FFFFFF", "position": [540, 760], "justify": "center", "in_s": 0.2, "out_s": 5.0,
       "fade_in_s": 0.4, "fade_out_s": 0.4, "slide_from": [0, 80], "shadow": true},
      {"id": "cta", "text": "Book a free measure", "font": "ArialMT", "size": 56, "color": "#FFFFFF",
       "position": [540, 1500], "in_s": 0.8, "out_s": 5.0, "fade_in_s": 0.3, "fade_out_s": 0.4}
    ],
    "images": [
      {"id": "logo_layer", "asset": "logo", "position": [540, 300], "scale": 40, "in_s": 0, "out_s": 5, "fade_in_s": 0.3}
    ]
  }
}
```

Field rules (all bounds enforced):

| field | rule |
| --- | --- |
| `schema_version` | must be `1` |
| `recipe` | `text_overlay_v1` (the only recipe today) |
| `recipe_revision` | optional 12-hex pin (sha256 over `lib.jsx` + the recipe's author/revise scripts); the worker's current value is returned in every manifest under `provenance.recipe_revision` and by `pipeline.recipe_revision("text_overlay_v1")`. Mismatch fails `RECIPE_REVISION_MISMATCH`. Pin it once the laptop has locked a version; leave it out while iterating. |
| `composition` | comp name, `[A-Za-z0-9 _-]{1,64}`, default `Main` |
| `output_profile` | `prores4444_alpha` (master `.mov` ProRes 4444 yuva444p10 straight alpha + review `.webm` VP9 yuva420p) |
| `org_id`, `job_id` | required, `[A-Za-z0-9_-]{1,64}`; provenance only (farm rows have no tenant column, same as matte / video_gen) |
| `order_id`, `version` | optional, provenance only |
| `assets[]` | ≤ 20; `name` `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` unique; `bucket` + `path` (no `..`); optional `sha256` (verified after download; mismatch fails), `size`; `kind` image\|video\|audio (the bytes are sniffed and must decode as that kind) |
| `settings.composition` | `width`/`height` even ints 16–8192; `fps` ∈ {23.976, 24, 25, 29.97, 30, 50, 59.94, 60}; `duration_s` 0.1–120 |
| layer common | `id` `^[a-z][a-z0-9_]{0,31}$` unique across texts/shapes/images; `position` [x,y]; `in_s` < `out_s` within the comp; `fade_in_s` + `fade_out_s` ≤ on-screen span; `opacity` 0–100; `slide_from` [dx,dy] (±4000) + `slide_s`; `scale` 1–1000, `scale_from` + `scale_s`; `shadow` true or `{opacity, distance, softness}` (built-in Drop Shadow) |
| `texts[]` | ≤ 20; `text` 1–500 chars (newlines allowed, no other control chars); `font` PostScript name (checked against the Windows font tables before AE launches, then inside AE); `size` 4–1000; `color` `#RRGGBB`; `justify` left\|center\|right; `tracking` ±500 |
| `shapes[]` | ≤ 20; `kind` rect\|ellipse; `size` [w,h]; `radius` (rect); `color` |
| `images[]` | ≤ 10; `asset` must name an entry of `assets[]` |
| total layers | ≤ 40 |

Z-order: images at the bottom, then shapes, then texts (each list bottom-up).
Text values are data: nothing from the request is evaluated as script.

Coordinate semantics (measured on the first demo job, 2026-09-11; not a
recipe change): a text layer is AE **point text**, so `position` is the
layer's anchor, which sits on the **baseline** of the first line, not the
visual centre of the glyphs. `justify` only affects the horizontal anchor
(`center` puts the anchor mid-line). To centre a one-line text visually at
`y`, send roughly `y + 0.35 * size`; for `N` lines the block grows downward
by about `1.2 * size` per extra line. Shapes and images are anchored at
their centre. Position, `slide_from` and `size` are in comp pixels; scale is
percent; times are seconds; colours `#RRGGBB`.

## Result

On success the row is `done` with `output_path = outputs/<farm_job_id>.mov`
(`output_ext = mov`, `signed_url` as usual). The sidecars are uploaded **before**
the master and the row only turns `done` after all five objects exist:

| object | MIME | content |
| --- | --- | --- |
| `outputs/<id>.mov` | `video/quicktime` | ProRes 4444 (`prores_ks` profile 4444, `yuva444p10le`, decodes as 12-bit), **straight alpha**, comp size/fps, exact frame count |
| `outputs/<id>-review.webm` | `video/webm` | VP9 `yuva420p` alpha review copy (crf 30), same frames |
| `outputs/<id>-bundle.zip` | `application/zip` | `project.aep`, `assets/<name>.<ext>`, `settings.json` (the validated request), `recipes/*.jsx` (the exact scripts that authored it) |
| `outputs/<id>-contact.png` | `image/png` | 6 sampled frames over a light row and a dark row |
| `outputs/<id>-manifest.json` | `application/json` | the result manifest below |

The AE preview is rendered media. Browser-side editing of layers is **not**
offered; a change is a new farm job (full re-author from changed `settings`),
or, for the PC-side revision path, the `revise_v1` recipe applied to the
bundle's `.aep` (used by the smoke test; not exposed as a queue engine in v1).

### Manifest (`outputs/<id>-manifest.json`)

```json
{
  "ok": true, "name": "<farm_job_id>", "workspace": "...\\attempt-1",
  "project": ".../project.aep", "bundle": "...", "master": "...", "review": "...", "contact_sheet": "...",
  "checks": {
    "codec": "prores", "profile": "4444", "pix_fmt": "yuva444p12le",
    "width": 1080, "height": 1920, "fps": 30.0, "frames": 150, "duration_s": 5.0,
    "alpha": {"frames": 150, "scale": "0..255", "mean_alpha_min": 0.0, "mean_alpha_max": 19.8, "mean_alpha_avg": 16.4,
              "frames_with_transparent_pixels": 150, "frames_with_opaque_pixels": 125,
              "frames_fully_transparent": 8, "frames_fully_opaque": 0,
              "raw_mid_frame": {"frame": 75, "min": 0, "max": 255}},
    "source": {"kind": "mov", "pix_fmt": "bgra", "codec": "rawvideo", "alpha_from_ae": "premultiplied",
               "unpremultiplied": true, "aerender": {"format": "AVI", "channels": "RGB + Alpha", "depth": "Millions of Colors+"}}
  },
  "review_info": {"codec": "vp9", "pix_fmt": "yuva420p", "raw_alpha_frame": {"frame": 75, "min": 0, "max": 255}, "size": 434550},
  "inspect": {"editable_layers": [
      {"index": 1, "name": "cta", "kind": "text", "text": "Book a free measure", "font": "ArialMT", "font_size": 56,
       "in_s": 0.8, "out_s": 5.0, "position_keys": 0, "opacity_keys": 4},
      {"index": 2, "name": "headline", "kind": "text", "text": "Motorised Blinds\rFrom $299", "font": "Arial-BoldMT", "...": "..."},
      {"index": 3, "name": "pill", "kind": "shape", "in_s": 0.6, "out_s": 5.0, "opacity_keys": 4}],
    "compositions": ["Main"], "ae": {"app_version": "26.5x89", "build_number": "89"}},
  "timings": {"author_s": 5.0, "render_s": 9.5, "encode_master_s": 2.3, "encode_review_s": 3.2,
              "verify_s": 2.5, "contact_sheet_s": 0.3, "verify_review_s": 0.2, "inspect_s": 5.0, "total_s": 28.5},
  "provenance": {
    "engine": "aftereffects", "schema_version": 1, "recipe": "text_overlay_v1", "recipe_revision": "5b1e0a9c2d44",
    "settings_sha256": "…", "inputs": {"logo": {"sha256": "…", "size": 48211, "kind": "image"}},
    "host": {"kind": "aftereffects", "version": "26.5.0.89", "dir": "C:\\Program Files\\Adobe\\Adobe After Effects 2026\\Support Files",
             "om_template": "Lossless with Alpha", "om_kind": "mov", "rs_template": "Best Settings"},
    "ae": {"app_version": "26.5x89"}, "ffmpeg": "9.0.1-full_build-www.gyan.dev",
    "output_settings": {"om_template": "Lossless with Alpha", "rs_template": "Best Settings",
                        "master": {"codec": "prores_ks", "profile": "4444", "pix_fmt": "yuva444p10le", "alpha": "straight"},
                        "review": {"codec": "libvpx-vp9", "pix_fmt": "yuva420p"}},
    "scope": {"org_id": "…", "job_id": "…", "order_id": "…", "version": 3},
    "fonts": {"Arial-BoldMT": {"available": true, "method": "app.fonts"}, "ArialMT": {"available": true, "method": "app.fonts"}}
  }
}
```

AE text layers store newlines as `\r`; the request's `\n` is mapped on the way in
and compared back on inspection.

## Progress and phases

`phase` on the row (`progress` 0–100 alongside): `downloading` → `validating`
→ `preflight` → `authoring` → `rendering` (aerender frame progress drives
20–70 %) → `encoding` → `verifying` → `inspecting` → `bundling` →
`uploading` → `done`. `heartbeat_at` keeps ticking through AE launches.

## Error codes

The row's `error` is `"<CODE>: message"`; branch on the prefix.

| code | meaning |
| --- | --- |
| `INVALID_REQUEST` | schema violation (message names the field) |
| `RECIPE_REVISION_MISMATCH` | pinned revision ≠ worker's recipe |
| `ASSET_MISSING` / `ASSET_HASH_MISMATCH` | input not in the bucket / not staged / wrong bytes |
| `FONT_MISSING` | PostScript name not installed on the host (checked before AE launches, and again by AE) |
| `EFFECT_MISSING` / `OM_TEMPLATE_MISSING` | built-in effect or output template absent in this AE |
| `AE_NOT_INSTALLED` / `AE_SLOT_TIMEOUT` / `AE_NOT_ISOLATED` / `AE_NO_MANIFEST` | host problems: no AE, slot held > `AE_SLOT_WAIT_SECONDS`, script landed in a foreign instance (refused, nothing touched), AE exited without its manifest (file-access preference) |
| `JSX_ERROR` | recipe threw: message carries `(file:line)` and the AE message |
| `RENDER_FAILED` / `RENDER_INCOMPLETE` | aerender error / wrong frame count or duration |
| `ALPHA_MISSING` / `VERIFY_FAILED` | output has no alpha or is flat; size/fps/layers differ from the request |
| `CLAIM_SUPERSEDED` | the row was reclaimed by another attempt before publish; nothing uploaded |
| `TIMEOUT` | `timeout_minutes` reached (processes killed) |

Cancellation: `cancel_requested` is honoured within ~5 s in every phase; the
AfterFX/aerender tree of this job is killed; the row ends `canceled`.

## Capability advertisement and migrations

Worker: `render_worker.capabilities()` returns `["aftereffects"]` when
`AfterFX.exe` + `aerender.exe` are found (or `AE_FAKE_HOST=1` in tests) and
passes it to `db.claim_job(caps)` → `claim_farm_job(p_capabilities)`. Before
the migration exists the worker detects the missing parameter once and falls
back to the plain call (gated rows are then unclaimable by everyone).

Migration (`docs/aftereffects-claiming.sql`, also folded into `schema.sql`):

1. `farm_engine_capabilities(engine, capability)` with `('aftereffects','aftereffects')`.
2. `drop function claim_farm_job()` + `create function claim_farm_job(p_capabilities text[] default '{}')`
   that excludes gated engines unless the capability is passed. The default
   makes every already-deployed worker (which calls it with no argument) safe.
3. `claim_farm_preview_job()` (perf/storyboard-batches lane, hyperframes stills only) untouched.

No change to `farm_render_jobs` columns, buckets or the MCP job helpers. The
laptop MCP gained `engine: "aftereffects"` + an `aftereffects` object param.

## Host facts (AE 26.5 on this PC) the platform should know

- Scripts run in a fresh instance (`AfterFX.exe -m -noui -r job.jsx`) with a
  per-launch `AE_JOB_TOKEN`; the recipe refuses to run (and never quits the
  app) when the token is missing, so an operator's open AE is never touched.
- "Allow Scripts to Write Files and Access Network": the recipe sets it
  in-memory (legacy pref section; the v2 section is refused by AE). Calling
  `preferences.saveToDisk()/reload()` **turns the permission off again** and
  every file write silently yields 0 bytes; the recipes never do that.
- `Lossless with Alpha` on Windows writes an **AVI** (raw BGRA, premultiplied,
  1.2 GB per 5 s at 1080x1920); the pipeline converts to straight alpha with a
  `blend=divide` graph (ffmpeg's `unpremultiply` filter is a no-op on it) and
  deletes the intermediate once the master exists. `-outputSettings "Color:"`
  is read-only in aerender 26.5.
- Timings (5 s, 1080x1920, 30 fps, 3 layers): author 5.0 s, aerender 9.5 s
  (~4 s of it launch), ProRes 2.3 s, VP9 3.2 s, verify 2.5 s, reopen check
  5.0 s, bundle < 0.1 s → 28.5 s; a text+timing revision 29.2 s. Budget ~35 s
  + queue wait per overlay; longer comps scale with aerender time.
- RAM: AE + aerender peak a few GB; the H3 video_gen job pauses TTS and takes
  the whole GPU — AE work is CPU/RAM only, but both share the one worker, so
  queue order decides. The AE slot mutex only serializes AE jobs among AE-capable
  processes.

## Activation (nothing of this is done yet)

1. Apply `docs/aftereffects-claiming.sql` in the Supabase SQL editor (Aimotion
   project). Verify with the probe row described in the file; delete it.
2. Merge `feat/aftereffects-engine` into `main`, `git pull` on the render PC,
   restart the worker (kill the real `python.exe` worker child, never the
   pythonw stub; the supervisor relaunches it). The worker log then shows
   `capabilities=['aftereffects']`.
3. Laptop: extend `submit_render_job` callers / `lib/queue/supabase-queue.ts`
   to insert `engine: "aftereffects"`, `repo_url: "-"`, `params.aftereffects`;
   extend `lib/overlay/tools.ts` prompts with the request schema above; read
   the five outputs by name from `output_path`.
4. First integrated test: the demo Blinds ad overlay with the approved
   price/CTA through the demo environment, then alpha verified in the actual
   final-composition engine plus timing and final audio. Production unchanged.
