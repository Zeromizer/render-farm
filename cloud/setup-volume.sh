#!/usr/bin/env bash
# One-time (and re-runnable) setup of the RunPod network volume, run over SSH on
# a pod with the volume at /workspace:
#   COMFY_REF=v0.37.0 VLO_REPO=<git url> VLO_REF=8a7092a bash setup-volume.sh
# Installs under /workspace: opt/{blender,ffmpeg}, ComfyUI (+venv, ComfyUI-vlo,
# run-headless.sh), venvs/worker, render-farm checkout. Models come separately:
# fetch_models.py (public files) and the PC's S3 upload (local-only files).
set -euo pipefail
: "${COMFY_REF:?set COMFY_REF to the tag/commit the PC runs}"
BLENDER_VER=${BLENDER_VER:-4.5.9}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
TORCH_VER=${TORCH_VER:-2.8.0}
W=/workspace
mkdir -p $W/opt $W/venvs $W/cache $W/models $W/logs $W/.ssh

echo "== ffmpeg (static, with NVENC)"
if [ ! -x $W/opt/ffmpeg/bin/ffmpeg ]; then
  curl -fsSL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    | tar -xJ -C $W/opt
  mv $W/opt/ffmpeg-master-latest-linux64-gpl $W/opt/ffmpeg
fi

echo "== Blender $BLENDER_VER"
if [ ! -x $W/opt/blender/blender ]; then
  curl -fsSL "https://download.blender.org/release/Blender${BLENDER_VER%.*}/blender-${BLENDER_VER}-linux-x64.tar.xz" \
    | tar -xJ -C $W/opt
  mv "$W/opt/blender-${BLENDER_VER}-linux-x64" $W/opt/blender
fi

echo "== ComfyUI @ $COMFY_REF"
[ -d $W/ComfyUI/.git ] || git clone -q https://github.com/comfyanonymous/ComfyUI $W/ComfyUI
git -C $W/ComfyUI fetch -q --tags origin
git -C $W/ComfyUI checkout -q "$COMFY_REF"
[ -x $W/ComfyUI/.venv/bin/python ] || python3 -m venv $W/ComfyUI/.venv
CPY=$W/ComfyUI/.venv/bin/python
$CPY -m pip install -q -U pip wheel
# torch first, from the CUDA index (Blackwell sm_120 needs cu128+), then
# ComfyUI's requirements without letting them swap torch. No xformers.
$CPY -m pip install -q "torch==$TORCH_VER" torchvision torchaudio --index-url "$TORCH_INDEX"
$CPY -m pip install -q -r $W/ComfyUI/requirements.txt
$CPY -m pip install -q sageattention || echo "sageattention not installed (ComfyUI runs without it)"
if [ -n "${VLO_REPO:-}" ]; then
  [ -d $W/ComfyUI/custom_nodes/ComfyUI-vlo/.git ] || git clone -q "$VLO_REPO" $W/ComfyUI/custom_nodes/ComfyUI-vlo
  git -C $W/ComfyUI/custom_nodes/ComfyUI-vlo checkout -q "${VLO_REF:-HEAD}"
  if [ -f $W/ComfyUI/custom_nodes/ComfyUI-vlo/requirements.txt ]; then
    $CPY -m pip install -q -r $W/ComfyUI/custom_nodes/ComfyUI-vlo/requirements.txt
  fi
fi
$CPY -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

echo "== render-farm checkout + worker venv"
if [ ! -f $W/.ssh/render-farm-deploy ]; then
  ssh-keygen -t ed25519 -N "" -C "runpod-render-farm" -f $W/.ssh/render-farm-deploy -q
  echo "ADD THIS READ-ONLY DEPLOY KEY to github.com/Zeromizer/render-farm:"
  cat $W/.ssh/render-farm-deploy.pub
fi
export GIT_SSH_COMMAND="ssh -i $W/.ssh/render-farm-deploy -o StrictHostKeyChecking=accept-new"
if [ ! -d $W/render-farm/.git ]; then
  git clone -q git@github.com:Zeromizer/render-farm.git $W/render-farm || echo "clone failed: add the deploy key, then re-run"
fi
if [ -f $W/render-farm/cloud/run-headless.sh ]; then
  cp $W/render-farm/cloud/run-headless.sh $W/ComfyUI/run-headless.sh
  chmod +x $W/ComfyUI/run-headless.sh
fi
[ -x $W/venvs/worker/bin/python ] || python3 -m venv $W/venvs/worker
if [ -f $W/render-farm/worker/requirements.txt ]; then
  $W/venvs/worker/bin/python -m pip install -q -r $W/render-farm/worker/requirements.txt httpx
fi

echo "== done. Next: fetch_models.py, then the PC's S3 upload of local-only files."
du -sh $W/* 2>/dev/null | sort -h | tail
