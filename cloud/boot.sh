#!/usr/bin/env bash
# Pod start command (worker/cloud_lane.py create_pod): bring the render worker up
# from the network volume, then hand over to the image's own /start.sh (sshd).
# Everything heavy (ComfyUI + models, Blender, ffmpeg, venvs) lives on the volume;
# setup-volume.sh puts it there once. Per boot only apt packages are reinstalled.
set -u
mkdir -p /workspace/logs
LOG=/workspace/logs/boot-$(date -u +%m%d-%H%M%S).log

{
  echo "boot $(date -u) pod=${RUNPOD_POD_ID:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
  # Runtime libraries for Blender (EGL/GL) and headless Chrome; not kept on the volume.
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libegl1 libgl1 libglu1-mesa libxi6 libxkbcommon0 libxrender1 libsm6 libxxf86vm1 libxfixes3 \
    libnss3 libatk-bridge2.0-0 libgbm1 libasound2t64 xvfb >/dev/null || echo "apt failed (continuing)"

  # Deploy key on the volume: the repo is private.
  # The volume makes every file 0666 and ssh refuses such a key: copy it out.
  install -D -m 600 /workspace/.ssh/render-farm-deploy /root/.ssh/render-farm-deploy
  export GIT_SSH_COMMAND="ssh -i /root/.ssh/render-farm-deploy -o StrictHostKeyChecking=accept-new"
  cd /workspace/render-farm
  git fetch -q origin && git checkout -q -f "origin/${RF_REF:-main}" && echo "render-farm $(git log --oneline -1)"

  set -a; . /workspace/render-farm/cloud/pod.env; set +a
  export PATH=/workspace/opt/ffmpeg/bin:/workspace/opt/blender:$PATH

  while true; do
    /workspace/venvs/worker/bin/python -u worker/render_worker.py
    echo "worker exited $? at $(date -u); restarting in 5 s"
    sleep 5
  done
} >>"$LOG" 2>&1 &

exec /start.sh
