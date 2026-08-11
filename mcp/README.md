# render-farm MCP — laptop install

Gives Claude Code tools to dispatch GPU renders to the home PC (RTX 4080).

## Install (work laptop)

```sh
git clone https://github.com/Zeromizer/render-farm.git
cd render-farm/mcp
npm install
```

Register with Claude Code (fill in the two env values — same ones as the
worker's `.env` on the home PC):

```sh
claude mcp add render-farm \
  -e SUPABASE_URL=https://otznmoiakqhtoeannldu.supabase.co \
  -e SUPABASE_SERVICE_KEY=<service key> \
  -- node <absolute path to render-farm>/mcp/index.js
```

Restart Claude Code, then check `/mcp` lists `render-farm`.

## Typical flow inside Claude Code

1. Work on your Remotion project locally, commit and **push** (code only —
   large media goes via `sync_assets`, not git).
2. `sync_assets(paths=["./broll"], dest_prefix="public")` — hashes files,
   uploads only new ones, returns an `assets` manifest.
3. `submit_render_job` with `engine=remotion`, `repo_url`, `ref`,
   `composition`, `quality="draft"`, `assets=<manifest>`.
4. `wait_for_job` / `get_job_status` — phase (cloning → syncing_assets →
   rendering) and progress %.
5. `download_result(job_id, "out/draft.mp4")` — iterate, then resubmit the
   same job with `quality="final"` (manifest reused verbatim).

Blender: `engine=blender`, `blend_file=scenes/shot.blend`, plus either
`single_frame` or `frame_start`/`frame_end`. PNG sequences arrive as a zip.

Python (GPU scripts — matting, upscales, interpolation): `engine=python`,
`script=scripts/job.py`, `output=out/frames` (dir → zip), optional
`args=[...]` and `requirements=scripts/requirements.txt` (venv cached per
content hash on the worker). See "Python engine" in the root README.

## Tools

| Tool | Purpose |
|---|---|
| submit_render_job | queue a render (returns job_id); `quality` draft/final, `assets` manifest |
| sync_assets | hash + upload local media to the assets bucket, returns manifest |
| get_job_status | status + progress % + phase |
| wait_for_job | block up to 600s until done |
| list_jobs | recent jobs |
| cancel_job | cancel pending/processing |
| download_result | download output into the workspace |

## Testing without restarting Claude Code

`node mcp/call.mjs` spawns the server over stdio and loads `../.env` itself, so
new tools/params can be exercised before Claude Code is restarted (its MCP
client only re-reads the tool list on startup):

```sh
node mcp/call.mjs                          # list tools
node mcp/call.mjs sync_assets '{"paths":["C:/clips"]}'
node mcp/call.mjs submit_render_job '{"engine":"remotion", ...}'
```

`smoke.mjs` does the same but expects the credentials already in the environment.
