#!/usr/bin/env node
// Render farm MCP server (stdio). Submits GPU render jobs to the Supabase
// queue; a worker on the home PC (RTX 4080) renders and uploads results.
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { insertJob, getJob, listJobs, cancelJob, summarize } from "./lib/jobs.js";
import { downloadResult } from "./lib/download.js";
import { syncAssets } from "./lib/assets.js";

const server = new McpServer({ name: "render-farm", version: "1.0.0" });

const json = (obj) => ({ content: [{ type: "text", text: JSON.stringify(obj, null, 2) }] });
const fail = (e) => ({ content: [{ type: "text", text: `Error: ${e.message}` }], isError: true });

server.tool(
  "submit_render_job",
  "Submit a GPU job (Remotion render, HyperFrames HTML render, Blender render, a Python script — e.g. " +
    "rembg matting, upscaling — or video_gen: MiniMax H3 AI video with native audio) to the home render farm. " +
    "For code engines the repo+ref must be pushed first; the farm clones it and runs on an RTX 4080. " +
    "video_gen needs no repo: pass `video_gen` (prompt required); inputs are {bucket, path} storage objects " +
    "(sync_assets puts files in the 'assets' bucket at sha256/<hex>). Expect several minutes per clip. " +
    "Returns a job_id — poll with get_job_status, then download_result.",
  {
    engine: z.enum(["remotion", "blender", "python", "hyperframes", "video_gen"]),
    repo_url: z.string().optional().describe("Git URL, e.g. https://github.com/user/repo (not used by video_gen)"),
    ref: z.string().default("main").describe("Branch, tag, or commit SHA (must be pushed)"),
    composition: z.string().optional().describe("Remotion: composition id (required for remotion)"),
    project_dir: z.string().optional().describe("Remotion: subdir of the repo containing package.json"),
    entry: z.string().optional().describe("Remotion: entry point if not auto-detected, e.g. src/index.ts"),
    codec: z.string().optional().describe("Remotion: h264 (default), vp9, gif, prores... (video only)"),
    output_kind: z.enum(["video", "still"]).optional()
      .describe("Remotion: 'still' renders one image via `remotion still` instead of a video. Named output_kind because `output` already belongs to the python engine."),
    image_format: z.enum(["png", "jpeg", "webp"]).optional()
      .describe("Remotion: image format when output_kind=still (default png)"),
    frame: z.number().int().optional()
      .describe("Remotion: which frame to capture when output_kind=still (zero-based, default 0)"),
    frame_range: z.string().optional().describe("Remotion: e.g. '0-120'"),
    props: z.record(z.any()).optional().describe("Remotion: input props object"),
    quality: z.enum(["draft", "final"]).optional()
      .describe("Remotion: draft = half resolution + fast encode (+half fps if the composition supports the quality prop) for fast previews; default final"),
    assets: z.array(z.object({
      path: z.string(),
      sha256: z.string().length(64),
      size: z.number().int(),
    })).optional().describe("Asset manifest from sync_assets; files are placed into the checkout at these repo-relative paths before rendering"),
    blend_file: z.string().optional().describe("Blender: repo-relative .blend path (required for blender)"),
    frame_start: z.number().int().optional(),
    frame_end: z.number().int().optional(),
    single_frame: z.number().int().optional().describe("Blender: render just this frame as an image"),
    output_format: z.string().optional().describe("Blender: FFMPEG (default) or PNG (zipped sequence)"),
    script: z.string().optional().describe("Python: repo-relative .py to run (required for python)"),
    args: z.array(z.string()).optional().describe("Python: CLI args for the script"),
    requirements: z.string().optional().describe("Python: repo-relative requirements.txt; venv is cached per content hash"),
    output: z.string().optional().describe("Python: repo-relative output file/dir the script writes (dir → zip; required for python)"),
    format: z.enum(["mp4", "webm", "mov", "gif", "png-sequence"]).optional()
      .describe("HyperFrames: output container (default mp4; webm/mov carry alpha; png-sequence → zip). entry (default index.html) names the composition file, project_dir the project subdir; quality accepts draft|standard|final|high; output_kind='still' + at=<seconds> returns one PNG via `hyperframes snapshot`."),
    fps: z.number().int().min(1).max(240).optional().describe("HyperFrames: override the composition fps (fps=15 is the real fast-draft knob)"),
    variables: z.record(z.any()).optional().describe("HyperFrames: values for data-composition-variables, passed as --variables-file --strict-variables"),
    workers: z.number().int().min(1).max(24).optional().describe("HyperFrames: parallel Chrome instances (default auto)"),
    gpu: z.boolean().optional().describe("HyperFrames: NVENC encode (--gpu)"),
    at: z.number().min(0).optional().describe("HyperFrames: seconds into the composition for output_kind=still (default 0)"),
    video_gen: z.object({
      prompt: z.string().optional().describe("Required for t2v/i2v/r2v. Describe the shot, motion and the sound you want (H3 generates synced audio)."),
      mode: z.enum(["t2v", "i2v", "r2v", "upscale", "turntable"]).optional().describe("Default: turntable if turntable given, upscale if source given, i2v if first/last_frame given, r2v if any ref_* given, else t2v"),
      turntable: z.object({
        front: z.object({ bucket: z.string(), path: z.string() }).describe("Straight-on front photo of the car, plain background"),
        rear: z.object({ bucket: z.string(), path: z.string() }).describe("Straight-on rear photo, same scale/framing/background"),
        car: z.string().describe("One line: colour, make, model, body style (e.g. 'A light teal-green metallic Proton e.MAS 7 SUV')"),
        details: z.string().optional().describe("Things to hold, e.g. 'Front badge is the Proton tiger emblem; plates read exactly e.MAS 7.'"),
        seconds_per_half: z.number().min(4).max(10).optional().describe("Seconds per 180-degree half (default 10; 768p cannot exceed 10)"),
        resolution: z.enum(["480p", "768p"]).optional().describe("Default 768p"),
        seed: z.number().int().optional(),
        fps: z.number().int().optional().describe("Output fps after RIFE interpolation (default 60)"),
        shorter_size: z.number().int().optional().describe("Output short edge in px (default 1080)"),
      }).optional().describe("Car/product 360 for background removal (mode turntable): two anchored 180-degree image-to-video halves between the two photos, automatic reseed/repair when a half drifts or teleports, seam-exact join, constant-speed remap, RIFE to fps. Loops. ~25-45 GPU minutes at 768p; use timeout_minutes 120+. Result outputs/<id>.mp4 plus sidecars outputs/<id>-piece1..N.mp4 (24 fps halves) and outputs/<id>-joined24.mp4."),
      source: z.object({ bucket: z.string(), path: z.string() }).optional().describe("upscale mode: existing clip to resize (no generation)"),
      upscale: z.object({
        method: z.enum(["lanczos", "seedvr2"]).optional().describe("Default lanczos: plain ffmpeg resize, instant and faithful. seedvr2 = restoration model, ~55 s per second of video, use for hero shots"),
        factor: z.number().min(1.01).max(4).optional().describe("Scale multiplier (default 2). 832x480 -> 1664x960"),
        shorter_size: z.number().int().optional().describe("Target short edge in px, e.g. 1080; overrides factor"),
        color_correction: z.enum(["wavelet", "lab", "adain", "none"]).optional().describe("Default wavelet (fast). lab is slower by ~3 s/frame"),
        temporal_overlap: z.number().int().min(0).optional().describe("Latent frames cross-faded between chunks (default 1)"),
        frames_per_chunk: z.number().int().optional().describe("4n+1 frames per chunk; default auto from free VRAM"),
        seed: z.number().int().optional(),
        blend: z.number().min(0).max(1).optional().describe("seedvr2 only: share of SeedVR2 in the output vs the lanczos resize (default 0.5; 1 = raw SeedVR2, which over-etches clean 768p sources)"),
      }).optional().describe("Resize pass (lanczos by default, seedvr2 opt-in). On generation modes it runs after the clip is made (base clip kept at outputs/<id>-base.mp4); required intent for mode=upscale (may be {})."),
      duration_s: z.number().min(1).max(15).optional().describe("Seconds, snapped up to H3's frame grid (default 5)"),
      resolution: z.enum(["480p", "768p"]).optional().describe("Short edge; default 480p (16 GB card). 768p is native but much slower"),
      ratio: z.enum(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]).optional().describe("Default 16:9; 9:16 for vertical ads"),
      seed: z.number().int().optional(),
      turbo: z.boolean().optional().describe("Distilled LoRA (default true): 8 steps t2v/i2v, 4 steps r2v"),
      steps: z.number().int().min(1).max(50).optional(),
      first_frame: z.object({ bucket: z.string(), path: z.string() }).optional().describe("i2v: still to animate from"),
      last_frame: z.object({ bucket: z.string(), path: z.string() }).optional().describe("i2v: still to end on"),
      ref_images: z.array(z.object({ bucket: z.string(), path: z.string() })).max(9).optional().describe("r2v: character/style references"),
      ref_videos: z.array(z.object({ bucket: z.string(), path: z.string() })).max(3).optional().describe("r2v: 2-15 s reference clips"),
      ref_audios: z.array(z.object({ bucket: z.string(), path: z.string() })).max(3).optional().describe("r2v: reference audio"),
      ref_image_size: z.enum(["match", "max"]).optional(),
    }).optional().describe("video_gen engine parameters (required for video_gen)"),
    priority: z.number().int().optional().describe("Lower runs first; a video_gen job blocks the farm for minutes"),
    timeout_minutes: z.number().int().optional().describe("Kill the job after this long (default 120; video_gen default 60, turntable 150)"),
  },
  async (args) => {
    try {
      if (args.engine === "video_gen" && !args.video_gen?.prompt && !args.video_gen?.source && !args.video_gen?.turntable)
        throw new Error("video_gen jobs require 'video_gen.prompt' (generation) or 'video_gen.source' (upscale)");
      if (args.engine !== "video_gen" && !args.repo_url)
        throw new Error("repo_url is required for this engine");
      if (args.engine === "remotion" && !args.composition)
        throw new Error("remotion jobs require 'composition'");
      if (args.engine === "blender" && !args.blend_file)
        throw new Error("blender jobs require 'blend_file'");
      if (args.engine === "python" && (!args.script || !args.output))
        throw new Error("python jobs require 'script' and 'output'");
      if (args.engine === "hyperframes" && args.entry && !/\.html?$/i.test(args.entry))
        throw new Error("hyperframes jobs take 'entry' as a composition .html file (default index.html)");
      const params = {};
      for (const k of ["composition", "project_dir", "entry", "codec", "frame_range",
        "props", "quality", "assets", "output_kind", "image_format", "frame",
        "blend_file", "frame_start", "frame_end", "single_frame", "output_format",
        "script", "args", "requirements", "output",
        "format", "fps", "variables", "workers", "gpu", "at", "video_gen"]) {
        if (args[k] !== undefined) params[k] = args[k];
      }
      const job = await insertJob({
        engine: args.engine, repo_url: args.repo_url || "-", git_ref: args.ref,
        params, priority: args.priority,
        timeout_minutes: args.timeout_minutes ?? (args.engine === "video_gen" ? (args.video_gen?.turntable ? 150 : 60) : undefined),
      });
      return json({ job_id: job.id, status: job.status });
    } catch (e) { return fail(e); }
  }
);

server.tool(
  "sync_assets",
  "Upload local media assets (b-roll videos, images, audio) to the farm's " +
    "content-addressed assets bucket. Files are hashed (SHA-256) and only " +
    "missing hashes are uploaded, so re-syncing is cheap. Returns an 'assets' " +
    "manifest — pass it directly as the `assets` param of submit_render_job. " +
    "Assets synced this way no longer need to be committed to git.",
  {
    paths: z.array(z.string()).min(1)
      .describe("Local files and/or directories to sync (dirs are walked recursively)"),
    dest_prefix: z.string().default("public")
      .describe("Repo-relative dir the files are materialized under on the worker, e.g. 'public' (prepend project_dir if the job uses one)"),
  },
  async (args) => {
    try { return json(await syncAssets(args)); } catch (e) { return fail(e); }
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
