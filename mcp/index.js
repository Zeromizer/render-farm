#!/usr/bin/env node
// Render farm MCP server (stdio). Submits GPU render jobs to the Supabase
// queue; a worker on the home PC (RTX 4080) renders and uploads results.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { insertJob, getJob, listJobs, cancelJob, summarize } from "./lib/jobs.js";
import { downloadResult } from "./lib/download.js";

const server = new McpServer({ name: "render-farm", version: "1.0.0" });

const json = (obj) => ({ content: [{ type: "text", text: JSON.stringify(obj, null, 2) }] });
const fail = (e) => ({ content: [{ type: "text", text: `Error: ${e.message}` }], isError: true });

server.tool(
  "submit_render_job",
  "Submit a GPU job (Remotion render, Blender render, or a Python script — e.g. " +
    "rembg matting, upscaling) to the home render farm. " +
    "The repo+ref must be pushed first; the farm clones it and runs on an RTX 4080. " +
    "Returns a job_id — poll with get_job_status, then download_result.",
  {
    engine: z.enum(["remotion", "blender", "python"]),
    repo_url: z.string().describe("Git URL, e.g. https://github.com/user/repo"),
    ref: z.string().default("main").describe("Branch, tag, or commit SHA (must be pushed)"),
    composition: z.string().optional().describe("Remotion: composition id (required for remotion)"),
    project_dir: z.string().optional().describe("Remotion: subdir of the repo containing package.json"),
    entry: z.string().optional().describe("Remotion: entry point if not auto-detected, e.g. src/index.ts"),
    codec: z.string().optional().describe("Remotion: h264 (default), vp9, gif, prores..."),
    frame_range: z.string().optional().describe("Remotion: e.g. '0-120'"),
    props: z.record(z.any()).optional().describe("Remotion: input props object"),
    blend_file: z.string().optional().describe("Blender: repo-relative .blend path (required for blender)"),
    frame_start: z.number().int().optional(),
    frame_end: z.number().int().optional(),
    single_frame: z.number().int().optional().describe("Blender: render just this frame as an image"),
    output_format: z.string().optional().describe("Blender: FFMPEG (default) or PNG (zipped sequence)"),
    script: z.string().optional().describe("Python: repo-relative .py to run (required for python)"),
    args: z.array(z.string()).optional().describe("Python: CLI args for the script"),
    requirements: z.string().optional().describe("Python: repo-relative requirements.txt; venv is cached per content hash"),
    output: z.string().optional().describe("Python: repo-relative output file/dir the script writes (dir → zip; required for python)"),
    timeout_minutes: z.number().int().optional().describe("Kill the job after this long (default 120)"),
  },
  async (args) => {
    try {
      if (args.engine === "remotion" && !args.composition)
        throw new Error("remotion jobs require 'composition'");
      if (args.engine === "blender" && !args.blend_file)
        throw new Error("blender jobs require 'blend_file'");
      if (args.engine === "python" && (!args.script || !args.output))
        throw new Error("python jobs require 'script' and 'output'");
      const params = {};
      for (const k of ["composition", "project_dir", "entry", "codec", "frame_range",
        "props", "blend_file", "frame_start", "frame_end", "single_frame", "output_format",
        "script", "args", "requirements", "output"]) {
        if (args[k] !== undefined) params[k] = args[k];
      }
      const job = await insertJob({
        engine: args.engine, repo_url: args.repo_url, git_ref: args.ref,
        params, timeout_minutes: args.timeout_minutes,
      });
      return json({ job_id: job.id, status: job.status });
    } catch (e) { return fail(e); }
  }
);

server.tool(
  "get_job_status",
  "Get status/progress of a render job.",
  { job_id: z.string().uuid() },
  async ({ job_id }) => {
    try { return json(summarize(await getJob(job_id))); } catch (e) { return fail(e); }
  }
);

server.tool(
  "wait_for_job",
  "Poll a job until it finishes or timeout_s elapses (max 600). Returns final status.",
  { job_id: z.string().uuid(), timeout_s: z.number().int().max(600).default(600) },
  async ({ job_id, timeout_s }) => {
    try {
      const deadline = Date.now() + timeout_s * 1000;
      let job = await getJob(job_id);
      while (["pending", "processing"].includes(job.status) && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10000));
        job = await getJob(job_id);
      }
      return json(summarize(job));
    } catch (e) { return fail(e); }
  }
);

server.tool(
  "list_jobs",
  "List recent render jobs (newest first).",
  { limit: z.number().int().max(50).default(10), status: z.string().optional() },
  async (args) => {
    try { return json(await listJobs(args)); } catch (e) { return fail(e); }
  }
);

server.tool(
  "cancel_job",
  "Cancel a pending or processing render job.",
  { job_id: z.string().uuid() },
  async ({ job_id }) => {
    try { return json(await cancelJob(job_id)); } catch (e) { return fail(e); }
  }
);

server.tool(
  "download_result",
  "Download a finished render into the local workspace. dest_path is relative to the current project (parent dirs are created).",
  { job_id: z.string().uuid(), dest_path: z.string() },
  async ({ job_id, dest_path }) => {
    try {
      const job = await getJob(job_id);
      return json(await downloadResult(job, dest_path));
    } catch (e) { return fail(e); }
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
