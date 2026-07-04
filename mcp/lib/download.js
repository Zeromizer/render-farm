import fs from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { sb, BUCKET } from "./supabase.js";

export async function downloadResult(job, destPath) {
  if (job.status !== "done" || !job.output_path) {
    throw new Error(`job is '${job.status}' — no output to download`);
  }
  // Mint a fresh signed URL so stored-URL expiry never blocks us.
  const { data, error } = await sb.storage.from(BUCKET).createSignedUrl(job.output_path, 3600);
  if (error) throw new Error(`signed url: ${error.message}`);

  const abs = path.resolve(destPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });

  const res = await fetch(data.signedUrl);
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(abs));

  const size = fs.statSync(abs).size;
  return { path: abs, bytes: size };
}
