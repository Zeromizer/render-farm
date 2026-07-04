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

1. Work on your Remotion project locally, commit and **push**.
2. `submit_render_job` with `engine=remotion`, `repo_url`, `ref`, `composition`.
3. `wait_for_job` / `get_job_status` — shows phase (cloning → rendering) and progress %.
4. `download_result(job_id, "out/final.mp4")` — file lands in your workspace.

Blender: `engine=blender`, `blend_file=scenes/shot.blend`, plus either
`single_frame` or `frame_start`/`frame_end`. PNG sequences arrive as a zip.

## Tools

| Tool | Purpose |
|---|---|
| submit_render_job | queue a render (returns job_id) |
| get_job_status | status + progress % + phase |
| wait_for_job | block up to 600s until done |
| list_jobs | recent jobs |
| cancel_job | cancel pending/processing |
| download_result | download output into the workspace |
