#!/usr/bin/env bash
# Pack ComfyUI's code + venv and the worker venv into ONE tar on the volume.
# boot.sh unpacks it onto the pod's local disk: Python start-up reads thousands
# of small files, which the network volume serves slowly (~2.5 min for ComfyUI
# to start from it, 2026-10-15) while one big sequential read runs at ~3.7 GB/s.
# Models stay on the volume (symlinked in). Re-run after any change to the
# ComfyUI checkout, its venv, custom nodes or the worker venv:
#   bash /workspace/render-farm/cloud/pack-local.sh
set -euo pipefail
W=/workspace
OUT=$W/cache/local-bundle.tar
mkdir -p $W/cache
t=$(date +%s)
tar -cf "$OUT.tmp" -C $W \
  --exclude=ComfyUI/models --exclude=ComfyUI/input --exclude=ComfyUI/output \
  --exclude=ComfyUI/temp --exclude=ComfyUI/user --exclude='ComfyUI/.venv-*' --exclude='*.log' --exclude='__pycache__' \
  ComfyUI venvs/worker
mv "$OUT.tmp" "$OUT"
echo "packed $(du -h "$OUT" | cut -f1) in $(( $(date +%s) - t )) s -> $OUT"
