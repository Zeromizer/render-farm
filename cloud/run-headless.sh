#!/usr/bin/env bash
# ComfyUI launcher on the cloud pod; setup-volume.sh copies it to
# /workspace/ComfyUI and worker/videogen/comfy_client.ensure_server starts it.
# No --fast-disk / --reserve-vram: those are the 16 GB card's workarounds.
cd "$(dirname "$0")"
SAGE=$(.venv/bin/python -c "import sageattention" 2>/dev/null && echo --use-sage-attention)
exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch $SAGE \
  >> comfyui-headless.log 2>&1
