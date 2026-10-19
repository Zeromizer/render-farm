#!/usr/bin/env bash
# Pod start command (worker/cloud_lane.py create_pod): bring the render worker up
# from the network volume, then hand over to the image's own /start.sh (sshd).
# Everything heavy (ComfyUI + models, Blender, ffmpeg, venvs) lives on the volume;
# setup-volume.sh puts it there once. Per boot: unpack the code+venv bundle to
# local disk, start ComfyUI, install the apt libraries, run the worker.
set -u
# The pod starts whatever boot.sh the LAST pod left on the volume. Update the
# checkout first and re-run the new script, or a change here takes effect one
# boot late (2026-10-15: a cold test ran the previous version).
if [ -z "${RF_BOOT_UPDATED:-}" ]; then
  install -D -m 600 /workspace/.ssh/render-farm-deploy /root/.ssh/render-farm-deploy
  GIT_SSH_COMMAND="ssh -i /root/.ssh/render-farm-deploy -o StrictHostKeyChecking=accept-new" \
    git -C /workspace/render-farm fetch -q origin \
    && git -C /workspace/render-farm checkout -q -f "origin/${RF_REF:-main}"
  export RF_BOOT_UPDATED=1
  exec bash /workspace/render-farm/cloud/boot.sh
fi
mkdir -p /workspace/logs
LOG=/workspace/logs/boot-$(date -u +%m%d-%H%M%S)-${RUNPOD_POD_ID:-local}.log
T0=$(date +%s)
stamp() { echo "[boot +$(( $(date +%s) - T0 ))s] $*"; }

{
  stamp "pod=${RUNPOD_POD_ID:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
  # Deploy key on the volume: the repo is private.
  # The volume makes every file 0666 and ssh refuses such a key: copy it out.
  install -D -m 600 /workspace/.ssh/render-farm-deploy /root/.ssh/render-farm-deploy
  export GIT_SSH_COMMAND="ssh -i /root/.ssh/render-farm-deploy -o StrictHostKeyChecking=accept-new"
  cd /workspace/render-farm
  git fetch -q origin && git checkout -q -f "origin/${RF_REF:-main}" && stamp "render-farm $(git log --oneline -1)"

  # ComfyUI and the worker venv run from local disk (cloud/pack-local.sh made
  # the bundle); the models stay on the volume behind a symlink.
  COMFY=/workspace/ComfyUI
  WPY=/workspace/venvs/worker/bin/python
  if [ -f /workspace/cache/local-bundle.tar ]; then
    mkdir -p /root/local && tar -xf /workspace/cache/local-bundle.tar -C /root/local \
      && COMFY=/root/local/ComfyUI && WPY=/root/local/venvs/worker/bin/python
    stamp "bundle unpacked to local disk (ComfyUI at $COMFY)"
    # Models: symlinks to the volume at once, replaced by local copies (H3 first)
    # in the background; cloud/stage-models.sh says why.
    bash /workspace/render-farm/cloud/stage-models.sh "$COMFY/models" 2>&1 | while read -r l; do stamp "$l"; done &
    for _ in $(seq 120); do [ -e "$COMFY/models/.links-ready" ] && break; sleep 0.5; done
  else
    stamp "no local bundle; running from the volume (slow start)"
  fi

  # ComfyUI first, beside the apt installs, so it is up before the first job
  # needs it. The launcher follows the checked-out branch.
  cp /workspace/render-farm/cloud/run-headless.sh $COMFY/run-headless.sh
  setsid bash $COMFY/run-headless.sh < /dev/null > /dev/null 2>&1 &

  # Runtime libraries for Blender (EGL/GL) and headless Chrome; not kept on the volume.
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libegl1 libgl1 libglu1-mesa libxi6 libxkbcommon0 libxrender1 libsm6 libxxf86vm1 libxfixes3 \
    libnss3 libatk-bridge2.0-0 libgbm1 libasound2t64 xvfb libvulkan1 >/dev/null || echo "apt failed (continuing)"
  stamp "apt done"

  set -a; . /workspace/render-farm/cloud/pod.env; set +a
  export COMFYUI_DIR=$COMFY WORKER_LOG=/workspace/logs/worker-${RUNPOD_POD_ID:-local}.log
  export PATH=/workspace/opt/ffmpeg/bin:/workspace/opt/blender:$PATH
  # Wait for it (the worker would start it too, but this marks the timing);
  # give up after 5 min and let the worker try.
  for _ in $(seq 300); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 1; done
  stamp "comfyui up (or gave up waiting)"

  while true; do
    $WPY -u worker/render_worker.py
    echo "worker exited $? at $(date -u); restarting in 5 s"
    sleep 5
  done
} >>"$LOG" 2>&1 &

exec /start.sh
