#!/usr/bin/env bash
# Copy the models from the network volume to this pod's local disk, H3 first.
# WHY: ComfyUI's loader reads weights in small pieces, and the volume serves
# those at ~270 MB/s (2026-10-15: the first H3 job spent ~150 s of its 190 s
# loading 40 GB) while one sequential copy runs at ~3.7 GB/s. Local NVMe serves
# the small reads fast.
#
# MODELS (a local dir) starts as a tree of symlinks to the volume files, so
# ComfyUI finds every model at once; each finished copy replaces its symlink by
# an atomic rename, so a job that starts mid-copy reads the volume (slower, but
# correct) and every later load reads local disk.
set -u
SRC=/workspace/ComfyUI/models
MODELS=${1:-/root/local/ComfyUI/models}
T0=$(date +%s)

# 1. The symlink tree (directories real, files linked).
( cd "$SRC" && find . -type d ) | while read -r d; do mkdir -p "$MODELS/$d"; done
( cd "$SRC" && find . -type f ! -name '*.part' ) | while read -r f; do
  [ -e "$MODELS/$f" ] || ln -s "$SRC/${f#./}" "$MODELS/$f"
done
touch "$MODELS/.links-ready"
echo "[stage +$(( $(date +%s) - T0 ))s] symlink tree ready"

# 2. Copy, H3 text-to-video first, then the other H3 family, then the rest.
order() {
  ( cd "$SRC" && find . -type f -name '*.safetensors' ) | awk '
    /minimax_h3_(audio|video)_vae|qwen3vl_32b_minimax_h3|minimax_h3_fl2va|fl2v_turbo/ {print 0, $0; next}
    /minimax_h3/                                                                   {print 1, $0; next}
                                                                                   {print 2, $0}' |
  sort -k1,1n -s | cut -d" " -f2-
}
order | while read -r f; do
  rel=${f#./}
  [ -L "$MODELS/$rel" ] || continue            # already a local copy
  cp "$SRC/$rel" "$MODELS/$rel.copy" && mv -f "$MODELS/$rel.copy" "$MODELS/$rel" \
    || { rm -f "$MODELS/$rel.copy"; echo "copy failed: $rel (stays on the volume)"; }
done
echo "[stage +$(( $(date +%s) - T0 ))s] all models local ($(du -shL "$MODELS" | cut -f1))"
