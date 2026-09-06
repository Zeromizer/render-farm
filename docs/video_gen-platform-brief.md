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
| `upscale` | object | Optional resize pass. `method`: `lanczos` (**default**: plain ffmpeg resize, instant, no GPU, faithful to the generated frames) or `seedvr2` (3B restoration model, ~55 s per second of video, hero shots only). `factor` (default 2) or `shorter_size` (px) for both. SeedVR2 extras: `blend` (share of SeedVR2 vs lanczos, default 0.5: the 2026-09-06 chihuahua A/B showed raw SeedVR2 etches fur and invents speckle on clean 768p footage), `color_correction` (wavelet default), `temporal_overlap`, `frames_per_chunk`, `seed`, `segment_frames`. On generation modes the base clip is kept at `outputs/<id>-base.mp4`. SeedVR2 ceiling ~2 MP/frame, so 768p sources use `shorter_size: 1080`. |
| `turntable` | object | **Car/product 360 for background removal** (`mode: "turntable"`, inferred when present). `front`, `rear`: `{bucket, path}` straight-on photos, same scale and framing, plain background; `car`: one line (colour, make, model, body); `details`: things to hold (badge, plate text); `seconds_per_half` (default 10, max 10 at 768p), `resolution` (default 768p), `seed`, `fps` (default 60), `shorter_size` (default 1080). The worker generates two anchored 180-degree image-to-video halves between the two photos (seams pixel-exact, loops), reseeds a half whose scene drifts off the backdrop, repairs a half that teleports mid-turn, joins on shared frames, remaps to constant speed and interpolates with RIFE. **25-45 GPU minutes at 768p; file with `timeout_minutes` 120+ (MCP default 150).** Result `outputs/<id>.mp4` (60 fps, 1080p) plus sidecars `outputs/<id>-piece1..N.mp4` (the 24 fps halves) and `outputs/<id>-joined24.mp4`. Proven on the Proton e.MAS 7 2026-09-06. |

Inputs are storage objects the worker downloads with the service key. Use the content-addressed
`assets` bucket the platform already writes: `{"bucket": "assets", "path": "sha256/<hex>"}` from
`rp_assets.storage_path`. The platform resolves `asset_id` -> `storage_path` and checks org
ownership before filing the job, same as it does for `params.matte.source`.

## Read progress and the result

Poll the row (or subscribe): `status` (`pending` → `processing` → `done` | `failed` | `canceled`),
`phase` (`downloading inputs`, `starting comfyui`, `uploading inputs`, `queued`, `running`,
`fetching`, `uploading`, `done`), `progress` 0-100 (coarse), `error`, `output_path`, `signed_url`,
`signed_url_expires_at`, `heartbeat_at`. Cancel with `update ... set cancel_requested = true`.

On `done`: download `signed_url` (or sign `output_path` in `renders` yourself) and file the clip
into the asset library the way matte outputs are filed (content-addressed into `assets`, an
`rp_assets` row, poster frame if you make them). The worker does **not** write `rp_assets`.

## Car 360 recipe (most Aimotion jobs are cars)

1. Get two studio photos of the car: straight-on front and straight-on rear, same distance, plain backdrop,
   the whole car in frame. File them into `assets` like any upload.
2. Insert one `video_gen` job with `mode: "turntable"` (table above), `priority` 100-150, `timeout_minutes` 150.
3. Poll the row: `phase` walks through `turntable half 1 (front to rear): running`, `half 2`, possible
   `try 2` / `repair` steps, `joining pieces`, `remapping...`, `encoding`.
4. On `done`, download `output_path` (loops, 60 fps, 1890x1080) and key the flat backdrop out downstream; there is
   no floor contact shadow. The sidecars are there if the editor wants the raw 24 fps halves.
5. Plain generation defaults for everything else: `upscale: {method: "lanczos"}` when a 1080p master is needed
   (instant, faithful); `method: "seedvr2"` with the default `blend` 0.5 only for hero shots.

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

- Repo state: the `video_gen` code lives in `c:\Coding\render-farm` on the PC and is **not yet
  committed or pushed**. The platform does not need that repo (it inserts rows directly), but if
  the laptop wants the updated `submit_render_job` MCP tool (`engine: "video_gen"` with a
  `video_gen` object), the PC needs to commit and push and the laptop pull + restart Claude Code.
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
