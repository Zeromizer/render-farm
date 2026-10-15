#!/usr/bin/env bash
# ComfyUI launcher on the cloud pod; setup-volume.sh copies it to
# /workspace/ComfyUI and worker/videogen/comfy_client.ensure_server starts it.
# Same flags as the PC's run-headless.bat minus --fast-disk / --reserve-vram,
# which are the 16 GB card's workarounds. --cache-none stays for now so H3
# timings compare like for like with the PC; try without it later.
cd "$(dirname "$0")"
export HF_HOME=/workspace/cache/hf PYTHONUTF8=1 PYTHONIOENCODING=utf-8
SAGE=$(.venv/bin/python -c "import sageattention" 2>/dev/null && echo --use-sage-attention)
exec .venv/bin/python -u main.py --listen 127.0.0.1 --port 8188 --cache-none --disable-auto-launch \
  --verbose INFO $SAGE >> comfyui-headless.log 2>&1
