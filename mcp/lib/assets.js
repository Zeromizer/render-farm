// Content-addressed asset sync: hash local files, upload only hashes missing
// from the private 'assets' bucket (objects keyed sha256/<hex>), and return a
// manifest to pass verbatim as submit_render_job's `assets` param.
import fs from "node:fs";
import path from "node:path";
import { posix } from "node:path";
import crypto from "node:crypto";
import { sb } from "./supabase.js";

export const ASSETS_BUCKET = "assets";
const UPLOAD_CONCURRENCY = 3;

const MIME = {
  ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
  ".mkv": "video/x-matroska", ".gif": "image/gif", ".png": "image/png",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
  ".svg": "image/svg+xml", ".mp3": "audio/mpeg", ".wav": "audio/wav",
  ".m4a": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg",
  ".ttf": "font/ttf", ".otf": "font/otf", ".woff": "font/woff", ".woff2": "font/woff2",
  ".json": "application/json",
};
const mimeFromExt = (p) => MIME[path.extname(p).toLowerCase()] || "application/octet-stream";

function sha256File(abs) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    fs.createReadStream(abs)
      .on("data", (chunk) => hash.update(chunk))
      .on("end", () => resolve(hash.digest("hex")))
      .on("error", reject);
  });
}

function* walkDir(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walkDir(abs);
    else if (entry.isFile()) yield abs;
  }
}

// Expand input paths (files and/or dirs) into [{abs, dest}] with forward-slash
// repo-relative dests under dest_prefix.
function collectFiles(paths, destPrefix) {
  const files = [];
  for (const p of paths) {
    const abs = path.resolve(p);
    const stat = fs.statSync(abs); // throws for nonexistent paths — intended
    if (stat.isDirectory()) {
      for (const f of walkDir(abs)) {
        const rel = path.relative(abs, f).split(path.sep).join("/");
        files.push({ abs: f, dest: posix.join(destPrefix, rel) });
      }
    } else {
      files.push({ abs, dest: posix.join(destPrefix, path.basename(abs)) });
    }
  }
  if (!files.length) throw new Error("no files found under the given paths");
  return files;
}

async function existsInBucket(hash) {
  const { data, error } = await sb.storage
    .from(ASSETS_BUCKET)
    .list("sha256", { limit: 1, search: hash });
  if (error) throw new Error(`assets bucket check failed: ${error.message}`);
  return !!data?.find((o) => o.name === hash);
}

async function uploadOne(abs, hash) {
  const { error } = await sb.storage
    .from(ASSETS_BUCKET)
    .upload(`sha256/${hash}`, fs.createReadStream(abs), {
      contentType: mimeFromExt(abs),
      upsert: true, // content-addressed: same key => same bytes, repeats are idempotent
      duplex: "half",
    });
  if (error) throw new Error(`upload failed for ${abs}: ${error.message}`);
}

export async function syncAssets({ paths, dest_prefix = "public" }) {
  const files = collectFiles(paths, dest_prefix);

  const entries = [];
  for (const f of files) {
    entries.push({
      path: f.dest,
      sha256: await sha256File(f.abs),
      size: fs.statSync(f.abs).size,
      _abs: f.abs,
    });
  }

  let uploaded = 0, skipped = 0, uploaded_bytes = 0;
  const queue = [...entries];
  const seen = new Set(); // dedupe identical content within this call
  const workers = Array.from({ length: UPLOAD_CONCURRENCY }, async () => {
    for (;;) {
      const e = queue.shift();
      if (!e) return;
      if (seen.has(e.sha256) || (await existsInBucket(e.sha256))) {
        skipped++;
      } else {
        await uploadOne(e._abs, e.sha256);
        uploaded++;
        uploaded_bytes += e.size;
      }
      seen.add(e.sha256);
    }
  });
  await Promise.all(workers);

  return {
    assets: entries.map(({ path: p, sha256, size }) => ({ path: p, sha256, size })),
    uploaded,
    skipped,
    total_bytes: entries.reduce((n, e) => n + e.size, 0),
    uploaded_bytes,
  };
}
