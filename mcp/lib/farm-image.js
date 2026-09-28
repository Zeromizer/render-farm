// Shrink video_gen image inputs before the farm sees them.
//
// ComfyUI's LoadImage unpacks every image into a float32 tensor at full size
// (~1.5 GB for a 120 MP camera JPEG) before the H3 node resizes it, and the H3
// node never uses more than a 2048 px short edge (ref_image_size "max"; first/
// last frames go to the generation size, 768p at most). On 2026-09-28 three such
// refs pushed the desktop out of RAM mid-generation. Same recipe as the render
// platform's lib/media/farm-image.ts - keep the two in step.
//
// Only the copy sent to the farm is shrunk; the synced original is untouched.
import crypto from "node:crypto";
import path from "node:path";

export const FARM_IMAGE_SHORT_EDGE = 2048;
export const FARM_IMAGE_LONG_EDGE = 4096; // panorama guard, only bites past 2:1
const MAX_INPUT_PIXELS = 250_000_000;
const RECIPE = `s${FARM_IMAGE_SHORT_EDGE}-l${FARM_IMAGE_LONG_EDGE}-q95`;
const FARM_DIR = "farm-refs";

// Target size, or null when the image already fits (then it is not re-encoded).
export function farmImageTarget(width, height) {
  const short = Math.min(width, height), long = Math.max(width, height);
  const scale = Math.min(FARM_IMAGE_SHORT_EDGE / short, FARM_IMAGE_LONG_EDGE / long);
  if (!(scale < 1)) return null;
  const byShort = FARM_IMAGE_SHORT_EDGE / short <= FARM_IMAGE_LONG_EDGE / long;
  const fit = (edge, limit, bound) => bound ? limit : Math.max(1, Math.round(edge * scale));
  const s = fit(short, FARM_IMAGE_SHORT_EDGE, byShort), l = fit(long, FARM_IMAGE_LONG_EDGE, !byShort);
  return width <= height ? { width: s, height: l } : { width: l, height: s };
}

// EXIF orientation first, then 8-bit sRGB RGB (alpha dropped, CMYK/16-bit
// converted), lanczos3, JPEG q95 4:4:4.
export async function shrinkForFarm(bytes) {
  const { default: sharp } = await import("sharp");
  const meta = await sharp(bytes, { limitInputPixels: MAX_INPUT_PIXELS }).metadata();
  const shown = meta.autoOrient ?? { width: meta.width, height: meta.height };
  if (!shown.width || !shown.height) throw new Error("its dimensions could not be read");
  const target = farmImageTarget(shown.width, shown.height);
  if (!target) return { fits: true, ...shown };
  const out = await sharp(bytes, { limitInputPixels: MAX_INPUT_PIXELS, failOn: "error" })
    .autoOrient()
    .resize(target.width, target.height, { fit: "fill", kernel: "lanczos3", fastShrinkOnLoad: false })
    .removeAlpha()
    .toColourspace("srgb")
    .jpeg({ quality: 95, chromaSubsampling: "4:4:4" })
    .toBuffer();
  return { fits: false, ...target, bytes: out, from: shown };
}

const fitsCache = new Set(); // "bucket/path" already known to fit, this process

// {bucket, path} -> the ref the farm should load.
export async function farmImageRef(sb, ref, log = () => {}) {
  if (!ref?.bucket || !ref?.path) return ref;
  const key = `${ref.bucket}/${ref.path}`;
  if (fitsCache.has(key)) return ref;
  // Content-addressed originals name their copy without a download.
  const hex = /^sha256\/([0-9a-f]{64})$/.exec(ref.path)?.[1];
  const nameFor = (h) => `${h}-${RECIPE}.jpg`;
  const exists = async (name) => {
    const { data } = await sb.storage.from(ref.bucket).list(FARM_DIR, { search: name, limit: 1 });
    return !!data?.some(o => o.name === name);
  };
  if (hex && await exists(nameFor(hex))) return { bucket: ref.bucket, path: `${FARM_DIR}/${nameFor(hex)}` };

  const { data, error } = await sb.storage.from(ref.bucket).download(ref.path);
  if (error || !data) throw new Error(`video_gen image ${ref.path} could not be read: ${error?.message ?? "empty"}`);
  const bytes = Buffer.from(await data.arrayBuffer());
  let result;
  try { result = await shrinkForFarm(bytes); }
  catch (e) { throw new Error(`video_gen image ${ref.path} could not be prepared (${e.message}); re-export it as JPEG/PNG with a short edge of at most ${FARM_IMAGE_SHORT_EDGE} px`); }
  if (result.fits) { fitsCache.add(key); return ref; }

  const name = nameFor(hex ?? crypto.createHash("sha256").update(bytes).digest("hex"));
  const dest = `${FARM_DIR}/${name}`;
  const up = await sb.storage.from(ref.bucket).upload(dest, result.bytes, { contentType: "image/jpeg", upsert: true });
  if (up.error) throw new Error(`resized copy of ${ref.path} could not be stored: ${up.error.message}`);
  log(`video_gen: ${path.posix.basename(ref.path)} ${result.from.width}x${result.from.height} (${(bytes.length / 1e6).toFixed(1)} MB) -> ${result.width}x${result.height} (${(result.bytes.length / 1e6).toFixed(1)} MB)`);
  return { bucket: ref.bucket, path: dest };
}

// Every image input of a video_gen params object, one at a time. Videos, audio
// and turntable ready_segments pass through untouched.
export async function prepareVideoGenImages(sb, vg, log) {
  if (!vg) return vg;
  const out = { ...vg };
  for (const k of ["first_frame", "last_frame"]) if (out[k]) out[k] = await farmImageRef(sb, out[k], log);
  if (Array.isArray(out.ref_images)) {
    const refs = [];
    for (const r of out.ref_images) refs.push(await farmImageRef(sb, r, log));
    out.ref_images = refs;
  }
  if (out.turntable) {
    const t = { ...out.turntable };
    for (const k of ["front", "rear", "left", "right"]) if (t[k]) t[k] = await farmImageRef(sb, t[k], log);
    out.turntable = t;
  }
  return out;
}
