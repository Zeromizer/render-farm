import { sb } from "./supabase.js";

export async function insertJob({ engine, repo_url, git_ref, params, timeout_minutes }) {
  const row = { engine, repo_url, git_ref: git_ref || "main", params };
  if (timeout_minutes) row.timeout_minutes = timeout_minutes;
  const { data, error } = await sb.from("render_jobs").insert(row).select().single();
  if (error) throw new Error(error.message);
  return data;
}

export async function getJob(jobId) {
  const { data, error } = await sb.from("render_jobs").select("*").eq("id", jobId).single();
  if (error) throw new Error(error.message);
  return data;
}

export async function listJobs({ limit = 10, status }) {
  let q = sb
    .from("render_jobs")
    .select("id,status,engine,repo_url,git_ref,progress,phase,error,created_at,completed_at")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (status) q = q.eq("status", status);
  const { data, error } = await q;
  if (error) throw new Error(error.message);
  return data;
}

export async function cancelJob(jobId) {
  // pending -> flip directly; processing -> request cancel (worker acts within ~5s)
  const { data: flipped, error: e1 } = await sb
    .from("render_jobs")
    .update({ status: "canceled", phase: "canceled" })
    .eq("id", jobId)
    .eq("status", "pending")
    .select();
  if (e1) throw new Error(e1.message);
  if (flipped && flipped.length) return { result: "canceled (was pending)" };

  const { data: req, error: e2 } = await sb
    .from("render_jobs")
    .update({ cancel_requested: true })
    .eq("id", jobId)
    .eq("status", "processing")
    .select();
  if (e2) throw new Error(e2.message);
  if (req && req.length) return { result: "cancellation requested; worker stops within ~5s" };
  return { result: "job is not pending or processing — nothing to cancel" };
}

export function summarize(job) {
  return {
    job_id: job.id,
    status: job.status,
    engine: job.engine,
    progress: job.progress,
    phase: job.phase,
    error: job.error,
    attempts: job.attempts,
    created_at: job.created_at,
    completed_at: job.completed_at,
    output_ready: job.status === "done",
  };
}
