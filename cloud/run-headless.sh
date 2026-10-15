#!/usr/bin/env bash
# ComfyUI launcher on the cloud pod; setup-volume.sh copies it to
# /workspace/ComfyUI and worker/videogen/comfy_client.ensure_server starts it.
# Same flags as the PC's run-headless.bat minus --fast-disk / --reserve-vram,
# which are the 16 GB card's workarounds. stays for now so H3
# it every prompt re-built the models: ~200 s per H3 clip on the pod, 2026-10-15).
# input/output/temp live on this pod's local disk: several pods share the
# volume, and ComfyUI empties its temp directory at startup.
cd "$(dirname "$0")"
export HF_HOME=/workspace/cache/hf PYTHONUTF8=1 PYTHONIOENCODING=utf-8
POD=${RUNPOD_POD_ID:-local}
LOCAL=/root/comfy
mkdir -p $LOCAL/input $LOCAL/output $LOCAL/temp /workspace/logs
SAGE=$(.venv/bin/python -c "import sageattention" 2>/dev/null && echo --use-sage-attention)
exec .venv/bin/python -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --input-directory $LOCAL/input --output-directory $LOCAL/output --temp-directory $LOCAL/temp \
  --verbose INFO $SAGE >> /workspace/logs/comfyui-$POD.log 2>&1
