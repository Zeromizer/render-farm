# Brief: connect Aimotion Render Platform to the local `video_gen` engine (MiniMax H3)

For the laptop Claude session working in the render-platform repo (the Vercel app at
render-platform-sandy.vercel.app). Written 2026-09-06 from the render PC after the engine was
verified end to end. Goal: give the platform a way to generate AI b-roll clips on the home
RTX 4080 instead of (or before) spending Seedance credits.

## What already exists and works (nothing to build on the PC side)

- **Queue**: `farm_render_jobs` table in the Aimotion Supabase project `otznmoiakqhtoeannldu`
  (same table the platform already files `matte`, `asset_check`, `reference_extract` jobs into).
- **Engine name**: `video_gen`. It is a NO_CLONE engine: `repo_url` must be the placeholder `"-"`.
- **Worker**: `c:\Coding\render-farm\worker` on the PC, always on (Startup shortcut + supervisor).
  It lazily starts a headless ComfyUI at `C:\ComfyUI`, generates, uploads, and frees the GPU.
- **Output**: one mp4 with native synchronized stereo audio at `outputs/<farm_job_id>.mp4` in the
  private `renders` bucket, plus `signed_url` (7-day) written onto the row. Identical shape to a
  matte or a Remotion render, so whatever the platform does with a finished matte works unchanged.
- **Measured on the 4080** (832x480, turbo): 3 s clip ~80-90 s, 10 s clip ~190 s. r2v ~50 s for 3 s.
  SeedVR2 x2 upscale (optional `upscale` key) adds ~150 s per 3 s of video; use it for hero shots, not every draft.
  Cancel via `cancel_requested=true` lands within ~3 s.

## Insert a job (exactly like matte)

```sql
insert into farm_render_jobs (engine, repo_url, git_ref, priority, timeout_minutes, params)
values ('video_gen', '-', 'main', 100, 60, '{"video_gen": { ... }}');
```

`params.video_gen` contract:

| key | type | notes |
|---|---|---|
| `org_id`, `job_id` | uuid | **Required for scoping.** `farm_render_jobs` has no tenant column; the platform must scope reads on `params.video_gen.job_id` (and org) exactly the way `wait_for_matte` scopes on `params.matte.job_id`, or any uuid can buy a signed URL to another org's clip. |
| `prompt` | string | **Required for t2v/i2v/r2v.** Describe shot, motion, and the sound you want; the model generates audio. Max ~7000 chars. |
| `mode` | `t2v` \| `i2v` \| `r2v` \| `upscale` | Optional. Inferred: `upscale` if `source` given, `i2v` if `first_frame`/`last_frame` given, `r2v` if any `ref_*` given, else `t2v`. |
| `duration_s` | 1-15 | Default 5. Snapped UP to H3's frame grid (3→3.04 s, 5→5.17 s, 10→10.13 s). |
| `resolution` | `480p` \| `768p` | Default `480p`. 768p (1344x768) works up to 10 s (243 frames: 522 s t2v, 315 s r2v turbo). **15 s at 768p OOMs** on the 16 GB card (thrashed 20 min, then failed); keep 768p jobs at 10 s or less, use 480p for longer clips. |
| `ratio` | `16:9` `9:16` `1:1` `4:3` `3:4` `21:9` | Default `16:9`. Use `9:16` for the vertical ads. |
| `seed` | int | Default 0. Change it to reroll. |
| `turbo` | bool | Default true (distilled LoRA, 8 steps t2v/i2v, 4 steps r2v). `false` = 20-step full schedule, ~2.5x slower, for hero shots. |
| `steps` | int | Optional override. |
| `first_frame`, `last_frame` | `{bucket, path}` | i2v: still(s) to animate from / end on. png/jpg/webp. |
| `ref_images[]` | `[{bucket, path}]` max 9 | r2v: character/style references. |
| `ref_videos[]` | `[{bucket, path}]` max 3 | r2v: 2-15 s clips (mp4/mov/webm). |
| `ref_audios[]` | `[{bucket, path}]` max 3 | r2v: reference audio (wav/mp3/flac). |
| `ref_image_size` | `match` \| `max` | r2v only. `max` is several times slower. |
| `source` | `{bucket, path}` | `mode: "upscale"` only: an existing clip to upscale with SeedVR2 (no generation). |
| `upscale` | object | Optional resize pass. `method`: `lanczos` (**default**: plain ffmpeg resize, instant, no GPU, faithful to the generated frames) or `seedvr2` (3B restoration model, ~55 s per second of video, hero shots only). `factor` (default 2) or `shorter_size` (px) for both. SeedVR2 extras: `blend` (share of SeedVR2 vs lanczos, default 0.5: the 2026-09-06 chihuahua A/B showed raw SeedVR2 etches fur and invents speckle on clean 768p footage), `color_correction` (wavelet default), `temporal_overlap`, `frames_per_chunk`, `seed`, `segment_frames`. On generation modes the base clip is kept at `outputs/<id>-base.mp4`. SeedVR2 ceiling ~2 MP/frame, so 768p sources use `shorter_size: 1080`. Third method `h3_latent_upscale` (2026-09-10): refine in H3's own latent space, see **Latent upscale** below. |
| `save_latent` | bool | Default false. Generation modes (t2v/i2v/turntable): also save the joint AV latent as an `.mmh3` packet, `outputs/<id>-latent.mmh3` (turntable: `outputs/<id>-<segment>-latent.mmh3`, repaired segments `-<segment>-piece<k>-latent.mmh3`, listed in the manifest). Prerequisite for `upscale.method: h3_latent_upscale` (tile/full). Fails loudly when the mmh3_media node pack is not installed on the PC. Not for r2v yet. Added 2026-09-10; see **Latent upscale** below. |
| `turntable` | object | **Car/product 360 for background removal** (`mode: "turntable"`, inferred when present). Two-anchor: `front` + `rear` photos -> 2 x 180 degrees. Four-anchor: `front` + `left` + `rear` + `right` -> 4 x 90 degrees (`left`/`right` are the VEHICLE's sides; one without the other is rejected). Plus `car`, `details`, `seconds_per_half` (10) / `seconds_per_quarter` (5), `segments`, `ready_segments`, `resolution` (768p), `seed`, `fps` (60), `shorter_size` (1080). Full contract, examples, staged workflow and timeouts in **Car 360** below. |

Inputs are storage objects the worker downloads with the service key. Use the content-addressed
`assets` bucket the platform already writes: `{"bucket": "assets", "path": "sha256/<hex>"}` from
`rp_assets.storage_path`. The platform resolves `asset_id` -> `storage_path` and checks org
ownership before filing the job, same as it does for `params.matte.source`.

**Extensionless objects are fine (fixed 2026-09-06).** A real job failed with
`video_gen input front: unsupported type ''` because `sha256/<hex>` has no suffix. The worker now
reads the type from the bytes (PNG/JPEG/WebP, MP4/MOV/WebM/MKV, WAV/MP3/FLAC/OGG/M4A), confirms the
file decodes with ffprobe, and names the local copy accordingly. This applies to every
`video_gen` input: `first_frame`, `last_frame`, `ref_*`, `source`, the turntable photos and
`ready_segments`. A file whose bytes are not a decodable media of the expected kind still fails
with a clear error (`... is video (.mp4) where image is expected`, `... does not decode as video`).
Do not rename or copy content-addressed objects for the farm.

## Read progress and the result

Poll the row or, better, subscribe to it (Supabase Realtime, `postgres_changes` on
`farm_render_jobs` with `filter: id=eq.<farm_job_id>`) and render a live card. Columns:

- `status`: `pending` → `processing` → `done` | `failed` | `canceled`.
- `progress`: 0-100, monotonic within a job, real (sampling steps are read from ComfyUI).
- `phase`: human-readable text, meant to be shown verbatim. Formats you will see (2026-09-06):
  - while pending: `queued: 2 ahead, starts in ~15 min` or `queued: next up` (the worker
    refreshes every ~30 s from the (priority, created_at) order and per-job estimates)
  - generation: `generate: loading model - ~9 min left`, then `generate: sampling 3/8 - ~5 min left`,
    `generate: decoding - ~30 s left`; with a resize pass the text ends `· job ~6 min - then upscale`
  - upscale: `upscale 1/4: ...` per 3 s segment (SeedVR2) or `upscale: lanczos`
  - turntable, two-anchor: `turntable half 1 (front to rear): sampling 5/8 - ~4 min left - job ~14 min - then 1 half + join`,
    then `turntable half 2 (rear to front): ...`, then `joining pieces`
  - turntable, four-anchor: `quarter 1/4 (front to left): sampling 3/8 - ~3 min left - job ~18 min - then 3 quarters + join`,
    `quarter 2/4 (left to rear): ...`, `quarter 3/4 (rear to right): ...`, `quarter 4/4 (right to front): ...`,
    then `joining quarters`. A reused segment produces no quarter phase; a single-segment job says `quarter 1/4 ...`
    then `... - then join + 60 fps`.
  - both: a segment may show `... try 2` (backdrop drift, reseeded) or `... repair (4s)` (mid-turn cut, remainder
    regenerated from the last clean frame to the same end photo), `<label>: checking for cuts`, then
    `checking segment transitions`, `joining ...`, `interpolating`, `encoding`, `uploading` (sidecars), `uploading` (result)
  - `uploading`, then `done`.
  Progress is measured: the generation spans 5-85 % split evenly across the segments still to generate, retries stay
  inside their segment's span, post-processing is 85-95 %, the result upload 99 %.
  The `~N min left` / `job ~N min` parts are estimates (ETA); parse `~(\d+) (s|min)` if you want a number.
- `heartbeat_at`: refreshed every 15 s while the worker is alive; stale > 5 min = the worker died,
  the farm reclaims the job.
- `error`: set with `failed`. `output_path` + `signed_url` (7-day) with `done`. Cancel by setting
  `cancel_requested = true`; the worker interrupts within ~3 s and the row ends `canceled`.

On `done`: download `signed_url` (or sign `output_path` in `renders` yourself) and file the clip
into the asset library the way matte outputs are filed (content-addressed into `assets`, an
`rp_assets` row, poster frame if you make them). The worker does **not** write `rp_assets`.

## Car 360 (most Aimotion jobs are cars)

### What the model can and cannot take (verified on the installed nodes)

`MiniMaxH3ImageToVideo` accepts exactly a `first_frame` and a `last_frame`; `MiniMaxH3ReferenceToVideo`
takes reference images but no frame anchors. There is no node that conditions one clip on four stills.
So a photo steers the model **only as the end frame of one segment and the start frame of the next**;
between anchors the wheels, doors and profile are the model's guess, guided by the prompt. Four anchors
help because every guess is now bounded by two real views 90 degrees apart instead of 180. Generation
completing is not proof of product accuracy: review the native segments before assembling.

### Photos

Straight-on views, same distance and focal length, plain backdrop, whole car in frame with space around
it. `left`/`right` are the **vehicle's** left and right (the side to a seated driver's left), not the
viewer's. The worker pads every photo onto the generation canvas at **one common scale** (no crop, no
stretch, letterboxed in the photo's own background colour): a side view is naturally wider than a
front view and must not be enlarged to match. Rejections come back on `error` with the numbers:
`anchor photos are not at the same scale: subject height front 405 px, left 620 px ...`,
`front photo: the subject touches the left edge ...`, `the car would fill only 30% of the canvas height ...`.

### Rotation

Clockwise as seen from above, always, in both variants:

| variant | segments (in order) | anchors per segment | default seconds |
|---|---|---|---|
| two-anchor (`front`, `rear`) | `front_to_rear`, `rear_to_front` | front->rear, rear->front | `seconds_per_half` 10 (4-10) |
| four-anchor (`front`, `left`, `rear`, `right`) | `front_to_left`, `left_to_rear`, `rear_to_right`, `right_to_front` | front->left, left->rear, rear->right, right->front | `seconds_per_quarter` 5 (3-10) |

Repairs regenerate only the remainder of the same segment, from its last clean frame to the segment's own
end photo, with the same clockwise wording, so a repair cannot leave its quarter or reverse.

### Example: full four-anchor rotation

```json
{"video_gen": {"org_id": "...", "job_id": "...", "mode": "turntable", "turntable": {
  "front": {"bucket": "assets", "path": "sha256/aaaa..."},
  "left":  {"bucket": "assets", "path": "sha256/bbbb..."},
  "rear":  {"bucket": "assets", "path": "sha256/cccc..."},
  "right": {"bucket": "assets", "path": "sha256/dddd..."},
  "car": "A pearl white Dongfeng 007 fastback sedan",
  "details": "The number plates read exactly 007 in the slanted angular Dongfeng wordmark; the badge is the Dongfeng twin-swallow emblem.",
  "seconds_per_quarter": 5, "resolution": "768p", "seed": 21, "fps": 60, "shorter_size": 1080
}}}
```
`timeout_minutes` 180 (MCP default for four-anchor; two-anchor stays 150). Expect ~5-6 min per 5 s quarter
at 768p plus one extra quarter per retry/repair, then ~1-2 min of post: 25-45 min in practice.

### Example: one quarter for review (same pipeline, no loop)

```json
{"video_gen": {"mode": "turntable", "turntable": {
  "front": {...}, "left": {...}, "right": {...}, "car": "...", "details": "...",
  "segments": ["front_to_left"], "seconds_per_quarter": 5
}}}
```
`segments` must be consecutive names of the variant's sequence (`["left_to_rear", "rear_to_right"]` is fine,
`["front_to_left", "rear_to_right"]` is rejected). Both side photos are still required whenever either
is given; the job downloads only the anchors its segments need (`front`, `left` here). The result is the
requested span at 60 fps, **not** labelled a 360 and not looped (`manifest.complete: false`).
`timeout_minutes` 60 is plenty for one quarter.

### Outputs (every turntable job)

| object in `renders` | what |
|---|---|
| `outputs/<id>.mp4` | the finished clip: constant-speed remap, RIFE to `fps`, lanczos to `shorter_size` (1890x1080 from a 1344x768 canvas). The resize adds no detail; the detail is the 768p canvas's |
| `outputs/<id>-<segment>.mp4` | one per segment in the job, **native 24 fps at canvas size**, after drift/cut repair. This is the unit of retry and reuse |
| `outputs/<id>-joined24.mp4` | the requested segments joined on their shared frames, native rate, before remap |
| `outputs/<id>-manifest.json` | `variant`, `segments_in_order`, `complete`, `loops`, per-segment `{bucket, path, frames, seconds, fps, reused, attempts, repairs}`, `seams` (`frame_diff` of the shared frame, `motion_jump` at the cut), `defects` (text), `timing_s`, `canvas`, `output` |
| `outputs/<id>-piece1..2.mp4` | two-anchor only, the same files as the segment sidecars under the pre-2026-09-06 names |

The manifest is the storage-reference contract: read it, show `defects`, and file the segment clips into
the org's library like any other output if you want them to outlive the `renders` retention.

### Retry one segment, then assemble without regenerating

1. Job A: full four-anchor rotation (or a quarter). Review `outputs/A-<segment>.mp4` per segment.
2. Job B: regenerate only the bad one, e.g. `"segments": ["left_to_rear"], "seed": 22` (same photos).
3. Job C: assemble the approved clips:
```json
{"video_gen": {"mode": "turntable", "turntable": {
  "ready_segments": {
    "front_to_left":  {"bucket": "renders", "path": "outputs/A-front_to_left.mp4"},
    "left_to_rear":   {"bucket": "renders", "path": "outputs/B-left_to_rear.mp4"},
    "rear_to_right":  {"bucket": "assets",  "path": "sha256/eeee..."},
    "right_to_front": {"bucket": "renders", "path": "outputs/A-right_to_front.mp4"}
  },
  "fps": 60, "shorter_size": 1080
}}}
```
`ready_segments` values are ordinary `{bucket, path}` inputs: a previous job's segment sidecar in `renders`,
or the same clip filed into the org's `assets` (content-addressed, extensionless is fine). Scope them the
way you scope every other input (org ownership of the asset / of the job that produced the sidecar).
Mixing is allowed: list some segments in `ready_segments`, give photos for the rest, and only the missing
ones are generated. A job with every segment ready uses no GPU (join, remap, RIFE, encode: ~2 min) and
needs no photos or `car`. Rules: ready clips must be the **native 24 fps segment files**, all on one
canvas, and each seam's shared frame must match (`frame_diff` <= 6, i.e. made from the same padded
anchors); otherwise the job fails with `segments X and Y do not meet ...`.

### Defects the worker measures and reports (manifest `defects`, log)

- backdrop drift and mid-turn cuts are fixed automatically (reseed / repair) and counted per segment;
- a seam whose motion jump exceeds 2.5x the steady-state motion is listed (visible pop);
- the loop seam is measured the same way when the sequence is complete;
- a left photo narrower than the front photo is flagged (probably the wrong photo in the slot).

Nothing measures badge/plate fidelity or wheel geometry between anchors: that stays a human review of the
native segments. Levers that help: exact plate wording in `details`, `turbo: false` is **not** available in
turntable mode (kept at the 8-step turbo schedule for time), a different `seed`, and real side photos.

### Quick two-anchor recipe (unchanged)

Front + rear photos, `mode: "turntable"`, `timeout_minutes` 150. Result loops at 60 fps 1890x1080; sidecars
`outputs/<id>-front_to_rear.mp4`, `-rear_to_front.mp4` (also as `-piece1/2.mp4`), `-joined24.mp4`, `-manifest.json`.
Plain generation defaults for everything else: `upscale: {method: "lanczos"}` when a 1080p master is needed
(instant, faithful); `method: "seedvr2"` with the default `blend` 0.5 only for hero shots.

## Latent upscale (H3 native), added 2026-09-10, validated on the render PC 2026-09-10

`upscale.method: "h3_latent_upscale"` upsamples a clip by re-running a short refine tail of the
original sampling on its saved latent (einhorn13/mmh3_media F07 + `MinimaxH3LatentUpscaler3D`),
so the output looks natively generated at the higher size: no sharpening halos, minimal drift. It is
opt-in per job, never the default, and never falls back to seedvr2/lanczos: a missing latent, node
pack, weight, wrong frame grid or OOM fails the row with a clear `error`.

Request shapes:

```json
{"video_gen": {"org_id": "...", "job_id": "...", "prompt": "...", "ratio": "9:16", "save_latent": true}}
```
```json
{"video_gen": {"org_id": "...", "job_id": "...", "mode": "upscale",
  "source": {"bucket": "assets", "path": "sha256/<clip>"},
  "upscale": {"method": "h3_latent_upscale", "variant": "tile", "shorter_size": 1080,
              "latent": {"bucket": "renders", "path": "outputs/<gen-id>-latent.mmh3"}}}}
```
```json
{"video_gen": {"mode": "upscale", "source": {...older clip, native 24 fps...},
  "upscale": {"method": "h3_latent_upscale", "variant": "decoded", "shorter_size": 1080}}}
```

| `upscale.*` key | notes |
|---|---|
| `variant` | `tile` (default; `MMH3H3NativeTileRefine`, 640x384 tiles, fits 16 GB), `full` (whole frame in one pass; measured limit 73 frames at 1080p on 16 GB: 3 s fits in 5.3 min, 5 s kills ComfyUI; `allow_large_full` bypasses the guard), `decoded` (no packet: the mp4 is VAE-encoded back into latent space; lower fidelity, experimental; 24 fps native clips only, and the clip is trimmed to the largest AV-exact length 39/90/141/192/243 frames because the node pack's decoded import rejects every other 17k+5 length) |
| `latent` | `{bucket, path}` of the clip's `.mmh3` packet: the `outputs/<id>-latent.mmh3` sidecar or the same file filed as an asset. Required for tile/full in upscale mode; on a generation job the packet just made is used and `save_latent` is forced on. Scope it like `ready_segments` (org ownership of the asset / of the job that produced the sidecar). The worker checks the packet's canvas and frame count against the source clip. |
| `shorter_size` / `factor` | as for the other methods (default `shorter_size: 1080`). Per-axis scale must stay within 1x-4x. The refine runs on the 32-aligned cover of the request (1088x1888 for 480x832 -> 1080) and the result is centre-cropped to the exact size. |
| `denoise` | 0 = source-aware (0.375 for the worker's turbo-8 packets, 0.5 for turbo-4, 0.25 otherwise); else 0.05-0.50. Lower = less drift. |
| `steps_override` | 0 = the packet's own profile (8 steps res_multistep/simple, shift 12/3); else 1-20. `decoded` defaults to 8 steps at denoise 0.375. |
| `seed`, `force_unload` (true), `attention` (Default only), `fp16_accumulation` (Default/Enabled/Disabled), `tile_width` 640, `tile_height` 384, `tile_overlap` 64, `context_padding` 64 (multiples of 32), `overlap_mode` reprocess/context_only, `blend_mode` half_cosine/linear/hard (defaults reprocess + half_cosine: seam-free; the F07 pair context_only + hard shows hard seams), `traversal`, `context_source`, `missing_audio_policy` (decoded), `prompt` (decoded) | |
| `fidelity` | `{enabled: true, compare: true, ssim_min: 0.85, psnr_min: 22, cell_ssim_min: 0.70, cell_drop_max: 0.25}` (PC-anchored 2026-09-10: lanczos 0.996/52.8 dB, raw SeedVR2 0.956/35.7, tile refine 0.922/25.2, full refine 0.919/24.9; the floors sit under a healthy refine and only a broken one trips them). ffmpeg ssim/psnr of the result against a lanczos resize of the source, globally and on a 4x4 grid; a cell far below its frame's mean is the badge/plate/wheel drift signal. Warns, never gates. |

Outputs in `renders` (besides `outputs/<id>.mp4`):

| object | what |
|---|---|
| `outputs/<id>-latent.mmh3` | the generation packet (`save_latent`); ZIP with `packet.json` (schema v2) + safetensors latent + media |
| `outputs/<id>-<segment>-latent.mmh3` | per turntable quarter; repaired quarters: `-<segment>-piece<k>-latent.mmh3`, see manifest `segments.<name>.latent` / `latent_pieces` / `latent_note` |
| `outputs/<id>-upscale.json` | provenance: source (path, sha256, size), latent (path, sha256, packet summary), recipe/variant, upscaler weight (+sha256), target/refine/crop, refine settings actually used, tiles, seed, ComfyUI/python/torch versions, node repo git heads, timing per phase, VRAM (min free / peak used estimate), fidelity summary, warnings |
| `outputs/<id>-fidelity.json` | `frames, ssim{mean,min,p05,per_frame}, psnr{...}, grid{cell_mean,cell_min}, flags{frames_below_ssim, frames_below_psnr, cells_low, cells_drift}, thresholds, verdict ok|review` |
| `outputs/<id>-compare.mp4` | side by side, lanczos-resized source (left) and the upscaled clip (right), source audio |
| `outputs/<id>-upscaled-latent.mmh3` | the refined packet when `save_latent` is on for the upscale job |

Phase text: `generate: preflight`, `uploading latent packet`, `h3 upscale: preflight`,
`h3 upscale: loading model - ~N min left`, `h3 upscale: sampling i/n ...` (the step counter restarts
per tile), `h3 upscale: decoding`, `upscale: cropping to the requested size`, `upscale: fidelity
check`, `uploading sidecars`. Progress budget as for seedvr2 (56-90).

Errors you will see verbatim on `error` (never a silent fallback):
`h3_latent_upscale needs upscale.latent {bucket, path}: the outputs/<id>-latent.mmh3 sidecar ...`,
`h3_latent_upscale preflight failed: missing node classes MMH3Load, ...: install https://github.com/einhorn13/mmh3_media ...`,
`... MinimaxH3LatentUpscaler3D.model_name has no 'minimax_h3_latent_upscaler_3d_bf16.safetensors' ...: download ... into ComfyUI/models/latent_upscale_models/`,
`latent packet is WxHxNf but the source clip is ...: pass the latent of the same generation`,
`source clip has N frames, which is not on H3's 17k+5 grid`, `variant 'decoded' needs a 24 fps source`,
`variant 'full' refines the whole ... clip in one pass: ... above the ... guard`,
`h3_latent_upscale ran out of VRAM: ... try variant 'tile' ...`.

Cost, measured on the RTX 4080 SUPER 2026-09-10: 480p 9:16 5 s -> 1080p tile = 27 min (12 tiles, 8
steps of ~10 s plus ~50 s per tile; peak VRAM 14.9-15.2 GB = the whole card), 3 s = 18 min; `full`
3 s = 5.3 min. The worker's estimate/ETA is calibrated to these. `timeout_minutes` 90 standalone /
120 with generation. Quality: far sharper than lanczos with no SeedVR2 speckle, but the tile refine
re-imagines small emblems (the Proton badge came out as a different emblem in every tile run) since
per-tile sampling cannot carry the image conditioning; `full` kept it. For badge/plate/logo shots use
`full` (<= 73 frames) or review `-compare.mp4`.

Policy (2026-09-11): production runs `variant: "full"` only (now the default) within the 73-frame
guard; `tile`, `decoded` and `allow_large_full` are dev-only (worker started with `H3_TILE_DEV=1`,
tile capped at 5 s) and the platform should not offer them. Pass `fidelity.critical_cells` with the
[row, col] cells (4x4 grid) that hold the badge, grille, wheels and plate: a drop of more than 0.10
below the frame mean in one of them fails the job (`error` starts with `h3_latent_upscale fidelity
FAIL`), with `-compare.mp4` and `-fidelity.json` still uploaded for review; the other cells warn at 0.25. Not available inside turntable jobs (upscale
the native segment clips as standalone jobs). Not for r2v generations yet (use `decoded`).

Platform notes: file `-latent.mmh3` as an asset (`kind: other`, `source_url: minimax:h3:<task>:latent`)
so it outlives render retention; resolve `upscale.latent` from that or from the producing task's
sidecar after the same org/job scoping as `ready_segments`; show `-compare.mp4` and the fidelity
verdict to the operator, whose rejection is simply the existing lanczos path. Finishing does not
touch storyboard/creative approval (it is a farm task, not a build order). The PC-side checklist
is `docs/h3-latent-upscale-pc-handoff.md`.

## Suggested platform work

1. **Tool for the agents**: a `generate_video` local tool (or extend the existing video-order flow
   that today talks to Seedance) that files a `video_gen` job and returns the farm job id; plus
   `wait_for_video` / `video_status` that poll the row scoped by `params.video_gen.job_id`.
   The `generate_image` tool in `C:\rp-relay\src\local-tools.js` is the closest pattern for the
   tool surface, and the matte filing code in the platform is the pattern for the queue side.
2. **Routing**: default new b-roll requests to `video_gen` (free, ~1.5-3 min), keep Seedance as an
   explicit opt-in for when quality or turnaround demands it.
3. **Defaults for ads**: `resolution: "480p"`, `ratio: "9:16"`, `duration_s: 5`, `turbo: true`,
   `priority: 100`, `timeout_minutes: 60`. A video job blocks the single render worker for its
   duration, so if renders must not wait, file video jobs at a lower priority number than renders
   only when they are urgent, otherwise use 150+.
4. **AGENTS.md / DIRECTOR.md**: tell builders the tool exists, that audio comes with the clip, and
   that a good prompt names the subject, the motion, the camera move and the sounds.

## Things to know

- Repo state: everything here is on `Zeromizer/render-farm` main. The platform does not need that repo
  (it inserts rows directly), but the laptop MCP (`submit_render_job` with a `video_gen` object, including
  `turntable.left/right/segments/ready_segments`) needs a pull + Claude Code restart after each push.
- License: MiniMax H3's community license excludes US use of the weights and their outputs. This
  was a deliberate choice; keep it in mind when deciding what the clips ship in.
- 16 GB VRAM: a 10 s 480p clip peaks with ~2 GB to spare. Do not file 768p jobs longer than ~5 s
  without testing first; they will likely fail with an OOM error on the row rather than hang.
- The worker pauses the OmniVoice/Chatterbox TTS workers for the duration of each video job and
  relaunches them after. If TTS and video jobs are ever filed together, expect TTS to lag by the
  video job's runtime.
- Logos, badges, number plates and other text are the weak spot of every video model, H3 included: on the
  Proton e.MAS 7 turntable test (r2v with 4 clean reference photos) the body, paint, wheels and light bar
  held up, the plate lettering and the badge smeared. Levers, in order of cost: `ref_image_size: "max"`,
  `turbo: false` (20 steps, ~2.5x slower), 768p, naming the exact text in the prompt. None guarantee it;
  for a hero shot plan to composite the real badge/plate in post or crop them out of frame.
- PC-side test without touching the platform:
  `c:\Coding\render-farm\worker\insert_test_job.py --engine video_gen --params "{\"prompt\": \"...\"}"`
