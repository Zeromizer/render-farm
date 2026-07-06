# render-farm

Personal GPU render farm: dispatch Remotion/Blender renders and Python
scripts from a laptop (via Claude Code MCP tools) to a home PC with an
RTX 4080 SUPER. Supabase
("Aimotion" project) is the only rendezvous — job queue table + private
Storage bucket. Both ends talk outbound HTTPS only; no inbound connections,
no admin rights needed anywhere.

```
LAPTOP (Claude Code + mcp/)                     HOME PC (worker/)
submit_render_job ──► render_jobs table ◄────── claim_render_job() RPC (poll 3s)
get_job_status    ◄── progress/heartbeat ◄───── heartbeat thread + stdout parsing
download_result   ◄── 'renders' bucket   ◄───── upload + signed URL
```

Transport is **git**: push your project, submit a job referencing repo+ref;
the worker clones/fetches, `npm ci` (Remotion, lockfile-hash cached), renders
with GPU (`--gl=angle` / OPTIX), uploads the result.

## Layout

- `schema.sql` — `render_jobs` table + `claim_render_job()` + `reclaim_stale_jobs()` (apply in Supabase SQL editor)
- `setup_supabase.py` — creates the private `renders` bucket
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

## Notes

- Private repos work if the PC's Git Credential Manager has credentials
  (seed once with a manual `git clone`).
- First Remotion render of a new repo downloads Chromium — takes minutes.
- One worker process only: it serializes the single GPU.
