#!/usr/bin/env bash
# One-time install of the Viggle Remake render stack (engine `remake`) on the network volume, BESIDE the farm's
# ComfyUI 0.37 (/workspace/ComfyUI stays untouched). It is Viggle's mcp/comfy_worker/setup.sh, step for step, with every
# download pinned to the revision studied and reproduced on a 5090 (2026-11-08; worker/remake/vendor/VENDOR.md):
#   ComfyUI v0.42.0, torch 2.9.1 / CUDA 13.0 venv, their node pack, the int8 swap+edit transformers,
#   the nvfp4 text encoder and both VAEs.
# Layout = theirs, so remake_worker.py finds ComfyUI next to itself:
#   $ROOT/repo/mcp/comfy_worker/{remake_worker.py, ComfyUI/}   $ROOT/repo/{int8,comfyui}/
# Run on a pod with the volume mounted:  bash /workspace/render-farm/cloud/setup-remake.sh
set -euo pipefail
ROOT=${REMAKE_ROOT:-/workspace/remake}
REV=7e5be143702dd53b9154482de673b8519ed2d4d6
COMFY_ORG_REV=${COMFY_ORG_REV:-main}
T0=$(date +%s); stamp() { echo "[setup-remake +$(( $(date +%s) - T0 ))s] $*"; }
mkdir -p "$ROOT"
[ -x "$ROOT/tools/bin/hf" ] || { python3 -m venv "$ROOT/tools" && "$ROOT/tools/bin/pip" install -q -U "huggingface_hub>=1.0" uv; }
export PATH="$ROOT/tools/bin:$PATH" UV_HTTP_TIMEOUT=600 HF_HUB_ENABLE_HF_TRANSFER=0
REPO="$ROOT/repo"
stamp "Viggle MCP + node pack + int8 at $REV"
hf download Viggle/Viggle-Remake --revision "$REV" --include "mcp/*" --include "comfyui/ComfyUI-Viggle-Remake/*" \
  --include "int8/*" --local-dir "$REPO"
cd "$REPO/mcp/comfy_worker"
stamp "ComfyUI v0.42.0 + venv (torch 2.9.1 cu130)"
[ -d ComfyUI ] || git clone -q --depth 1 --branch v0.42.0 https://github.com/comfyanonymous/ComfyUI.git
[ -x ComfyUI/venv/bin/python ] || uv venv -q --python 3.12 ComfyUI/venv
uv pip install -q --python ComfyUI/venv/bin/python torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install -q --python ComfyUI/venv/bin/python -r ComfyUI/requirements.txt
ln -sfn "$REPO/comfyui/ComfyUI-Viggle-Remake" ComfyUI/custom_nodes/ComfyUI-Viggle-Remake
stamp "text encoder + VAEs"
hf download Comfy-Org/MiniMax-H3 --revision "$COMFY_ORG_REV" text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors --local-dir ComfyUI/models
ln -sf "$REPO"/int8/*.safetensors ComfyUI/models/diffusion_models/
ComfyUI/venv/bin/python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
echo "$REV" > "$ROOT/REVISION"
stamp "done: comfy_worker_dir=$REPO/mcp/comfy_worker"
