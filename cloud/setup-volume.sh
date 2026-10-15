#!/usr/bin/env bash
# One-time (and re-runnable) setup of the RunPod network volume, run over SSH on
# a pod with the volume at /workspace:
#   bash cloud/setup-volume.sh
# Installs under /workspace: opt/{blender,ffmpeg}, ComfyUI (+venv, the node
# packs in custom-nodes.txt, run-headless.sh), venvs/worker, render-farm
# checkout. Versions match the PC (render-pc report 2026-10-15). Models come
# separately: fetch_models.py (public files) and the PC's S3 upload (local-only).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMFY_REF=${COMFY_REF:-73c9bad4d21e7addbe1d13bc92eee0f1431b017d}  # v0.37.0, as on the PC
BLENDER_VER=${BLENDER_VER:-4.5.9}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
TORCH_VER=${TORCH_VER:-2.8.0}
TORCHVISION_VER=${TORCHVISION_VER:-0.23.0}  # torch-family versions pinned as on the PC (unpinned torchaudio came up 2.11, ABI break)
W=/workspace
mkdir -p $W/opt $W/venvs $W/cache $W/models $W/logs $W/.ssh

echo "== ffmpeg (static, with NVENC)"
if [ ! -x $W/opt/ffmpeg/bin/ffmpeg ]; then
  # A release branch, not master: master's NVENC wants driver 610+, RunPod had 595.
  curl -fsSL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz \
    | tar -xJ --no-same-owner -C $W/opt
  mv $W/opt/ffmpeg-n8.1-latest-linux64-gpl-8.1 $W/opt/ffmpeg
fi

echo "== Blender $BLENDER_VER"
if [ ! -x $W/opt/blender/blender ]; then
  curl -fsSL "https://download.blender.org/release/Blender${BLENDER_VER%.*}/blender-${BLENDER_VER}-linux-x64.tar.xz" \
    | tar -xJ --no-same-owner -C $W/opt
  mv "$W/opt/blender-${BLENDER_VER}-linux-x64" $W/opt/blender
fi

echo "== ComfyUI @ $COMFY_REF"
# init + fetch rather than clone: the PC's S3 upload may already have made
# ComfyUI/models/..., and git clone refuses a non-empty directory.
mkdir -p $W/ComfyUI
if [ ! -d $W/ComfyUI/.git ]; then
  git -C $W/ComfyUI init -q
  git -C $W/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI
fi
git -C $W/ComfyUI fetch -q --tags origin
git -C $W/ComfyUI checkout -q -f "$COMFY_REF"
[ -x $W/ComfyUI/.venv/bin/python ] || python3 -m venv $W/ComfyUI/.venv
CPY=$W/ComfyUI/.venv/bin/python
$CPY -m pip install -q -U pip wheel
# torch first, from the CUDA index (Blackwell sm_120 needs cu128+), then
# ComfyUI's requirements without letting them swap torch. No xformers.
$CPY -m pip install -q "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" "torchaudio==$TORCH_VER" --index-url "$TORCH_INDEX"
$CPY -m pip install -q -r $W/ComfyUI/requirements.txt
$CPY -m pip install -q sageattention || echo "sageattention not installed (ComfyUI runs without it)"
while read -r name url ref; do
  case "$name" in ''|'#'*) continue ;; esac
  d=$W/ComfyUI/custom_nodes/$name
  [ -d "$d/.git" ] || git clone -q "$url" "$d"
  git -C "$d" fetch -q origin
  git -C "$d" checkout -q "$ref"
  if [ -f "$d/requirements.txt" ]; then $CPY -m pip install -q -r "$d/requirements.txt"; fi
  echo "node $name @ $(git -C "$d" rev-parse --short HEAD)"
done < "$HERE/custom-nodes.txt"
# A node pack's requirements can swap torch for a CPU or older build: put it back.
$CPY -m pip install -q "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" "torchaudio==$TORCH_VER" --index-url "$TORCH_INDEX"
$CPY -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

echo "== render-farm checkout + worker venv"
if [ ! -f $W/.ssh/render-farm-deploy ]; then
  ssh-keygen -t ed25519 -N "" -C "runpod-render-farm" -f $W/.ssh/render-farm-deploy -q
  echo "ADD THIS READ-ONLY DEPLOY KEY to github.com/Zeromizer/render-farm:"
  cat $W/.ssh/render-farm-deploy.pub
fi
# The volume makes every file 0666 and ssh refuses such a key: use a local copy.
install -D -m 600 $W/.ssh/render-farm-deploy /root/.ssh/render-farm-deploy
export GIT_SSH_COMMAND="ssh -i /root/.ssh/render-farm-deploy -o StrictHostKeyChecking=accept-new"
if [ ! -d $W/render-farm/.git ]; then
  git clone -q git@github.com:Zeromizer/render-farm.git $W/render-farm || echo "clone failed: add the deploy key, then re-run"
fi
cp "$HERE/run-headless.sh" $W/ComfyUI/run-headless.sh
chmod +x $W/ComfyUI/run-headless.sh
[ -x $W/venvs/worker/bin/python ] || python3 -m venv $W/venvs/worker
if [ -f $W/render-farm/worker/requirements.txt ]; then
  $W/venvs/worker/bin/python -m pip install -q -r $W/render-farm/worker/requirements.txt httpx
fi

echo "== done. Next: fetch_models.py, then the PC's S3 upload of local-only files."
du -sh $W/* 2>/dev/null | sort -h | tail

# Code + venvs onto one tar for fast pod starts (boot.sh unpacks it locally).
bash "$HERE/pack-local.sh"
