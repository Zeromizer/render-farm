#!/usr/bin/env bash
# One-time install of the ComfyUI render worker, next to this script: ComfyUI 0.42.0 with torch 2.9.1 / CUDA 13.0, the
# repository's node pack linked in, the int8 transformers and the text encoder and VAEs. SKIP_MODELS=1 skips the weights.
set -e
cd "$(dirname "$0")"
REPO=$(realpath ../..)
export UV_HTTP_TIMEOUT=600
[ -d ComfyUI ] || git clone -q --depth 1 --branch v0.42.0 https://github.com/comfyanonymous/ComfyUI.git
[ -x ComfyUI/venv/bin/python ] || uv venv -q --python 3.12 ComfyUI/venv
uv pip install -q --python ComfyUI/venv/bin/python torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install -q --python ComfyUI/venv/bin/python -r ComfyUI/requirements.txt
M=(--include "comfyui/ComfyUI-Viggle-Remake/*")
[ -n "$SKIP_MODELS" ] || M+=(--include "int8/*")
hf download Viggle/Viggle-Remake "${M[@]}" --local-dir "$REPO"
ln -sfn "$REPO/comfyui/ComfyUI-Viggle-Remake" ComfyUI/custom_nodes/ComfyUI-Viggle-Remake
if [ -z "$SKIP_MODELS" ]; then
    hf download Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
        vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors --local-dir ComfyUI/models
    ln -sf "$REPO"/int8/*.safetensors ComfyUI/models/diffusion_models/
fi
echo "done. Start the worker with: bash $PWD/run_worker.sh"
