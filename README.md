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
