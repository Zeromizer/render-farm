# render-farm

Personal GPU render farm: dispatch Remotion/Blender renders from a laptop
(via Claude Code MCP tools) to a home PC with an RTX 4080 SUPER. Supabase
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

## Notes

- Private repos work if the PC's Git Credential Manager has credentials
  (seed once with a manual `git clone`).
- First Remotion render of a new repo downloads Chromium — takes minutes.
- One worker process only: it serializes the single GPU.
