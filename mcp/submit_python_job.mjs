// One-off: submit a python-engine job (bypasses a stale in-session MCP schema).
// Usage: node submit_python_job.mjs <repo_url> <ref> <script> <output> [requirements]
import { readFileSync } from "fs";

const env = readFileSync(new URL("../.env", import.meta.url), "utf8");
for (const key of ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]) {
  const m = env.match(new RegExp(`${key}=(.+)`));
  if (m && !process.env[key]) process.env[key] = m[1].trim();
}

const { insertJob } = await import("./lib/jobs.js");
const [repo_url, git_ref, script, output, requirements] = process.argv.slice(2);
const params = { script, output };
if (requirements) params.requirements = requirements;
const job = await insertJob({ engine: "python", repo_url, git_ref, params, timeout_minutes: 90 });
console.log(JSON.stringify({ job_id: job.id, status: job.status }));
