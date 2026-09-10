# render-farm

Personal GPU render farm: dispatch Remotion/Blender renders and Python
scripts from a laptop (via Claude Code MCP tools) to a home PC with an
RTX 4080 SUPER. Supabase
("Aimotion" project) is the only rendezvous — job queue table + private
Storage bucket. Both ends talk outbound HTTPS only; no inbound connections,
no admin rights needed anywhere.

```
LAPTOP (Claude Code + mcp/)                     HOME PC (worker/)
submit_render_job ──► farm_render_jobs table ◄── claim_farm_job() RPC (poll 3s)
sync_assets       ──► 'assets' bucket       ◄──  content-addressed download cache
get_job_status    ◄── progress/heartbeat    ◄──  heartbeat thread + stdout parsing
download_result   ◄── 'renders' bucket      ◄──  upload + signed URL
```

Transport is **git** for code: push your project, submit a job referencing
repo+ref; the worker clones/fetches, `npm ci` (Remotion, lockfile-hash
cached), renders with GPU (`--gl=angle` / OPTIX), uploads the result.
Large media assets skip git via the assets bucket (see below).

## Layout

- `schema.sql` — `farm_render_jobs` table + `claim_farm_job()` + `reclaim_stale_farm_jobs()` (apply in Supabase SQL editor)
- `setup_supabase.py` — creates/updates the private `renders` + `assets` buckets (2GB file_size_limit)
- `worker/` — Python worker for the render PC (see below)
- `mcp/` — Node stdio MCP server for the laptop (see `mcp/README.md`)

## Render PC setup

```sh
py -3.11 -m venv .venv
.venv\Scripts\pip install -r worker\requirements.txt
copy .env.example .env   # fill SUPABASE_SERVICE_KEY
.venv\Scripts\python setup_supabase.py
worker\start-worker.bat  # foreground test
powershell -File worker\make_startup_shortcut.ps1  # auto-start at logon
```

Job lifecycle: `pending → processing → done | failed | canceled`, with
heartbeat every 15s, stale-job reclaim (5 min), 2 attempts max, per-job
timeout (default 120 min), cancellation within ~5s, and startup cleanup of
old repo caches (14d) and work dirs (2d).

## Assets bucket (added 2026-08-12)

Large media (b-roll video, images, audio) no longer needs to be committed to
the video repo. The laptop's `sync_assets` MCP tool hashes local files
(SHA-256) and uploads only missing hashes to the private `assets` bucket at
`sha256/<hex>` — sync once, reference forever. It returns a manifest
(`[{path, sha256, size}, ...]`, paths repo-root-relative like
`public/broll-1.mp4`) that is passed as the `assets` param of
`submit_render_job`.

Before rendering (phase `syncing_assets`), the worker downloads any hashes
missing from its content-addressed cache (`%LOCALAPPDATA%\render-farm\assets`,
streamed + hash-verified) and hardlinks them into the checkout at the manifest
paths. Cache entries untouched for 30 days are evicted at worker startup
(`ASSET_CACHE_MAX_AGE_DAYS` env to change). A manifest entry whose hash is not
in the bucket fails the job fast with a "run sync_assets first" error.

Size limits: `setup_supabase.py` sets a 2GB per-bucket `file_size_limit`, but
the **project-global upload cap** (Supabase Dashboard → Storage → Settings;
50 MB on the Free plan) still applies on top — raise it there if big uploads
413.

## Draft mode (added 2026-08-12)

Remotion jobs accept `quality: "draft" | "final"` (default final, unchanged
behavior). Draft renders with `--scale=0.5` and `--crf=32` (CRF-capable codecs)
and injects `{"quality": "draft"}` into the input props. For the full speedup,
the composition should halve fps in draft via `calculateMetadata`:

```tsx
const FPS = 30;
const DURATION_IN_FRAMES = 1360;

<Composition ... fps={FPS} durationInFrames={DURATION_IN_FRAMES}
  defaultProps={{ quality: "final" as "final" | "draft" }}
  calculateMetadata={({ props }) =>
    props.quality === "draft"
      ? { fps: FPS / 2, durationInFrames: Math.round(DURATION_IN_FRAMES / 2), props }
      : { props }}
/>
```

Caveats: wall-clock timing is preserved only for seconds-based animations
(`frame / fps`); frame-count-hardcoded animations run 2× fast in drafts.
Components can also branch on `props.quality` to skip expensive effects
(blurs, particles). Combined draft speedup is typically ~4-8×. With halved
fps, `frame_range` refers to draft frame numbers.

Typical flow: `sync_assets` → `submit_render_job(quality="draft", assets=…)` →
iterate → same submit with `quality="final"`.

## Python engine (added 2026-07-06)

`engine: "python"` runs a repo script on the worker — for GPU-hungry
non-render work (rembg/BiRefNet matting, Real-ESRGAN upscales, RIFE
interpolation). Params: `script` (repo-relative .py, required), `output`
(repo-relative file or dir the script writes — dir gets zipped; required),
`args` (string list), `requirements` (repo-relative requirements.txt — a
venv is built once per unique requirements content and cached in
`%LOCALAPPDATA%\render-farm\venvs\<hash>`). The script runs with
cwd=<repo>, gets `RENDER_WORK_DIR` in env, and can print `PROGRESS <0-100>`
lines to drive job progress.

GPU tip for onnxruntime jobs on Windows: put `onnxruntime-directml` in the
job's requirements and create sessions with
`providers=["DmlExecutionProvider"]` — DirectML uses the 4080 without any
CUDA/cuDNN install. ffmpeg must be on the worker PATH for jobs that use it.

**Deploy this upgrade on the render PC** (one-time):

```sh
cd <render-farm checkout> && git pull
# then restart the worker: close the pythonw process (Task Manager) or reboot;
# the startup shortcut relaunches it. No new worker deps needed.
```

## hyperframes engine (added 2026-09-03)

`engine: "hyperframes"` renders a [HyperFrames](https://hyperframes.heygen.com)
project — an `index.html` whose clips carry `data-start`/`data-duration` and
whose motion is a paused GSAP timeline — through headless Chrome and the
**system** ffmpeg. Params: `entry` (composition .html, default `index.html`),
`project_dir`, `format` (mp4 default, webm/mov with alpha, gif, png-sequence →
zip), `quality` (draft|standard|final→standard|high), `fps`, `variables`
(object → `--variables-file --strict-variables`), `workers`, `gpu` (NVENC),
`resolution`, `crf`; `output_kind: "still"` + `at: <seconds>` returns one PNG
via `hyperframes snapshot`. `assets` manifests work as for remotion.

Nothing to `npm ci`: the project's package.json pins `npx --yes
hyperframes@X.Y.Z` and the runner reads that pin (else `HYPERFRAMES_VERSION`,
else 0.8.26). The CLI needs **Node >= 22** and ffmpeg on PATH. Chrome headless
shell is downloaded once into `~/.cache\hyperframes\chrome` by `hyperframes
browser ensure`, which the runner calls the first time each worker process
renders; fonts pulled from Google Fonts are cached under
`~/.cache\hyperframes\fonts`, so warm them once online. Telemetry and update
checks are disabled through the environment. On Windows the runner also patches
`@puppeteer/browsers/lib/launch.js` inside the npx cache right after `browser
ensure` (`detached` off for win32): Puppeteer's detached launch makes
CreateProcess ignore CREATE_NO_WINDOW, so chrome-headless-shell and each child
process it forks otherwise open a visible Windows Terminal window on the render
PC (eight per snapshot, measured 2026-09-07). Idempotent, re-applied per
version. Measured on the laptop iGPU: a
645-frame 1080x1920 footage reel with music in 69 s, A/V offset 0 ms (no AAC
priming compensation needed, unlike Remotion's ~+40 ms).

`quality: "draft"` only lowers the bitrate (capture dominates); send `fps: 15`
for a genuinely fast preview.

**Deploy on the render PC** (one-time): `git pull`, confirm `node -v` is 22+,
restart the worker. First hyperframes job downloads the CLI and Chrome (~1-2 min).

## reference_extract engine (added 2026-09-02)

`engine: "reference_extract"` measures a video into a `motion_spec` JSON for
the render-platform's reference library (HANDOFF-reference-extraction.md):
ffprobe, PySceneDetect cuts, librosa tempo/beats, optical-flow camera labels,
k-means grade, OCR captions (once easyocr lands), saliency composition. No
repo is cloned — `repo_url` is a `"-"` placeholder; params:

```json
{ "extract": { "kind": "reference" | "render_output",
               "reference_id" | "render_id": "<uuid>",
               "bucket": "assets" | "renders", "path": "<object path>",
               "stages": ["probe","shots","audio","motion","grade","composition"],
               "ocr_fps": 1, "motion_fps": 10 } }
```

Motion samples at its own rate (default 10fps, `motion_fps` param) decoupled
from the 2fps colour/OCR set; shots whose peak flow exceeds 2x the clip
median re-sample at 15fps and carry a per-frame `flow_curve`, and the camera
vocabulary includes `speed_ramp` (a sustained monotonic >3x magnitude run).
Budget: ~90s typical, may stretch toward ~2 minutes on clips with many
high-motion shots (§9 amendment).

The runner downloads the object, runs `worker/extract/extract.py` in a cached
venv keyed on `worker/extract/requirements.txt` (the worker's own venv stays
supabase-only), uploads keyframes + contact sheet to `renders/refs/<id>/`
(upsert — idempotent per id), writes the spec onto
`rp_references.motion_spec` / `rp_renders.motion_spec` with the service key,
and returns the JSON as the job's single output. Partial stage failures mark
that section `failed` and continue; only a broken probe fails the job.

These jobs insert with `priority: 200` (renders default 100) and the updated
`claim_farm_job()` orders by `(priority, created_at)` — a queued extraction
never delays a render. **Deploy: paste the changed statements from schema.sql
(priority column + index + claim_farm_job) into the Supabase SQL editor, pull
on the render PC, restart the worker** (kill the real python.exe worker child,
never the pythonw stub — see the restart note in project memory). The Phase 2
requirements bump (torch + easyocr) must be pre-built by hand:
`pip install -r worker\extract\requirements.txt` into the hashed venv path, and
run `easyocr.Reader(['en'])` once to pre-download models — a cold build blows
the 15-minute job timeout.

## Notes

- Private repos work if the PC's Git Credential Manager has credentials
  (seed once with a manual `git clone`).
- First Remotion render of a new repo downloads Chromium — takes minutes.
- One worker process only: it serializes the single GPU.

## video_gen engine: MiniMax H3 (added 2026-09-06)

Local AI video with native audio on the 4080, replacing remote Seedance
credits for b-roll. `engine: "video_gen"` needs no repo (`repo_url` is `-`);
everything lives in `params.video_gen` (see `worker/runners/video_gen.py`
docstring): `prompt` (required), `mode` t2v|i2v|r2v, `duration_s` 1-15,
`resolution` 480p|768p, `ratio`, `seed`, `turbo`, `steps`, and
`{bucket, path}` inputs `first_frame` / `last_frame` (i2v) or
`ref_images[]` / `ref_videos[]` / `ref_audios[]` (r2v). Output is one mp4 at
`outputs/<job_id>.mp4` like every other engine.

The generation runs in a headless ComfyUI at `C:\ComfyUI` (own venv, torch
2.8.0+cu128, official Comfy-Org H3 safetensors: pruned fp8 FL2VA/Ref2VA,
NVFP4 Qwen3-VL text encoder, both VAEs, turbo LoRAs). The worker starts it
on demand via `C:\ComfyUI\run-headless.bat` (`--fast-disk --reserve-vram 0.9`,
mandatory on a 31 GB RAM / 16 GB VRAM box), pauses the TTS workers for the
duration (`VIDEO_GEN_PAUSE_TTS=0` to disable), and calls `/free` afterwards.
Expect minutes per clip; a video job blocks the single render worker, so
set `priority` accordingly. 480p is the sane default on this card.

Optional resize pass (`upscale` object, or `mode: "upscale"` + `source` for
an existing clip): `method` `lanczos` (default: plain ffmpeg, instant, no
GPU) or `seedvr2` (3B restoration model), `factor` or `shorter_size`. SeedVR2
extras: `color_correction` (wavelet default), `frames_per_chunk`,
`temporal_overlap`, `seed`, `blend` (share of SeedVR2 vs lanczos, default
0.5: raw SeedVR2 over-etches clean H3 768p footage), `segment_frames`. Long
SeedVR2 clips run in 73-frame segments (RAM ceiling) and are re-joined with
the source audio; the base clip is kept at `outputs/<job_id>-base.mp4`.
SeedVR2 output ceiling is ~2 MP/frame (1080p), so 768p sources use
`shorter_size: 1080`; ~150-165 s per 3 s segment on the 4080.

**H3 Studio** (`worker\start-studio.bat` -> http://127.0.0.1:8790, added
2026-09-06): a local web UI over the same queue. Prompt H3 with every knob
(mode, duration, resolution, ratio, seed, turbo, first/last frame, reference
images/videos/audio, reference size) plus the resize pass; a library of every
result with thumbnails, inline playback, params, upscale (lanczos / SeedVR2 +
blend), 60 fps (RIFE, local), reuse, download, delete, cancel; and the **car
turntable** flow (`worker/studio/turntable.py`): two photos (front, rear) ->
two anchored 180-degree image-to-video halves -> automatic reseed of a half
whose scene drifts off the backdrop (grey floor / overhead camera, seen with
seed 7) and repair of a half that teleports mid-turn -> seam-exact join -> constant-speed time remap ->
RIFE to 60 fps. Library lives in `STUDIO_DIR` (default
`%USERPROFILE%\Videos\H3-Studio`), RIFE is the portable rife-ncnn-vulkan at
`RIFE_DIR` (default `C:\Coding	ools
ife-ncnn-vulkan`). Stdlib server,
binds 127.0.0.1 only; `studio\import_clip.py` adds existing mp4s.

`mode: "turntable"` (added 2026-09-06): the car 360 as one farm job, same
flow the studio uses (`worker/studio/turntable.py`, run in-process against
ComfyUI). Two-anchor: `turntable = {front, rear, car, details,
seconds_per_half, ...}` -> `front_to_rear`, `rear_to_front`. Four-anchor
(same day, later): add `left` and `right` (the vehicle's own sides, both
required) -> `front_to_left`, `left_to_rear`, `rear_to_right`,
`right_to_front`, clockwise from above, `seconds_per_quarter` (5). `segments`
picks a consecutive subset (one quarter to review), `ready_segments`
`{name: {bucket, path}}` reuses approved native 24 fps segment clips instead of
regenerating them (all four ready = assembly only, no GPU). Photos are padded
at one common scale and rejected with numbers when they do not match; every
seam and the loop seam are measured. Sidecars: `outputs/<id>-<segment>.mp4`
(native 24 fps), `-joined24.mp4`, `-manifest.json` (segments, seams, defects,
timing); two-anchor jobs also keep `-piece1..2.mp4`. Inputs may be
extensionless (`assets/sha256/<hex>`): `videogen/media_type.py` sniffs and
verifies them. ~5 min per 5 s quarter, ~9 min per 10 s half at 768p; MCP
default `timeout_minutes` 150 (two) / 180 (four). Tests:
`cd worker && ..\.venv\Scripts\python.exe -m unittest discover -s tests -t .`.
Contract for the platform: `docs/video_gen-platform-brief.md`.

PC-side checks without Supabase: `worker\videogen\smoke.py` (t2v/i2v/r2v
flags, prints VRAM before/after and wall time). Queue path:
`worker\insert_test_job.py --engine video_gen --params "{\"prompt\": \"...\"}"`.

### H3 latent upscale (added 2026-09-10, GPU validation pending)

A third finishing method, `upscale.method: "h3_latent_upscale"`, upsamples in
H3's own latent space instead of post-processing pixels: the F07 workflows of
[einhorn13/mmh3_media](https://github.com/einhorn13/mmh3_media) (MIT) with the
external `MinimaxH3LatentUpscaler3D` node
([LBH-123-AI](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler),
weights `minimax_h3_latent_upscaler_3d_bf16.safetensors` in
`ComfyUI/models/latent_upscale_models/`). Graphs live in
`worker/videogen/graphs_h3.py` (transcribed from the reference JSONs vendored
under `worker/videogen/recipes/reference/`); a `/object_info` preflight fails
the job with install hints when the node pack or the weight is missing. It
never falls back to SeedVR2 or lanczos.

- `save_latent: true` on a generation (t2v/i2v/turntable) also saves the joint
  AV latent as an `.mmh3` packet: `outputs/<id>-latent.mmh3`
  (`outputs/<id>-<segment>-latent.mmh3` per turntable quarter). Same models,
  sampler and seed as the default graph, which stays untouched.
- `upscale: {method: "h3_latent_upscale", variant: "tile"|"full"|"decoded",
  latent: {bucket, path}, shorter_size|factor, denoise (0 = source-aware),
  steps_override, seed, tile_*}`. `tile` is the 16 GB default; `full` refines
  the whole frame (short clips only, guarded); `decoded` re-encodes an mp4
  that has no packet (clips generated before this branch; lower fidelity,
  experimental, 24 fps native clips only).
- The refine runs on the 32-aligned cover of the request (1088x1888 for a
  480x832 -> 1080 job) and the result is centre-cropped. Sidecars:
  `outputs/<id>-upscale.json` (provenance: source, latent, recipe, weight,
  refine settings, ComfyUI/node versions, timing, VRAM), `-fidelity.json`
  (ffmpeg ssim/psnr against a lanczos resize, 4x4 grid, warns only),
  `-compare.mp4` (source | upscaled), `-upscaled-latent.mmh3` with `save_latent`.
- Provisional cost (unmeasured): ~9 min per 5 s 480p -> 1080p clip in tiles;
  MCP default `timeout_minutes` 90 (standalone) / 120 (generation + upscale).
- Deploy on the render PC: `docs/h3-latent-upscale-pc-handoff.md` (install,
  the five option strings to confirm, measurements, failure drills, calibration).
