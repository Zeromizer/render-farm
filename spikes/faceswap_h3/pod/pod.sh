#!/usr/bin/env bash
# One-shot H3 render pod for spikes/faceswap_h3 (NOT the farm lane: no worker; name fsh3-*).
# usage (pod start command): bash /workspace/fsh3/pod.sh <job>
# Reads /workspace/fsh3/jobs/<job>/{in,graphs}, writes /workspace/fsh3/jobs/<job>/out, then removes the pod.
# Uses only core ComfyUI nodes + H3 weights already on the farm volume, so no installs or downloads.
set -u
JOB=$1
J=/workspace/fsh3/jobs/$JOB
OUT=$J/out
mkdir -p $OUT
exec > >(tee -a $OUT/pod.log) 2>&1
T0=$(date +%s); stamp() { echo "[fsh3 +$(( $(date +%s) - T0 ))s] $*"; }
finish() {
  stamp "finishing: $1"
  cp -f /root/comfy/comfyui.log $OUT/ 2>/dev/null; sync
  runpodctl remove pod "$RUNPOD_POD_ID" || curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
  sleep infinity
}
( sleep 7200; echo "[fsh3] 2 h cap hit"; runpodctl remove pod "$RUNPOD_POD_ID" ) &
stamp "job=$JOB pod=${RUNPOD_POD_ID:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ComfyUI code + venv from the farm bundle onto local disk (fast start)
mkdir -p /root/local && tar -xf /workspace/cache/local-bundle.tar -C /root/local || finish "bundle unpack failed"
C=/root/local/ComfyUI; PY=$C/.venv/bin/python; M=$C/models
for f in diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors \
         text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
         vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors \
         loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors; do
  mkdir -p $(dirname $M/$f); cp /workspace/ComfyUI/models/$f $M/$f || finish "copy failed: $f"
done
stamp "models copied"

mkdir -p /root/comfy/input /root/comfy/output /root/comfy/temp
cp $J/in/* /root/comfy/input/
cd $C
SAGE=$($PY -c "import sageattention" 2>/dev/null && echo --use-sage-attention)
setsid $PY -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
  --input-directory /root/comfy/input --output-directory /root/comfy/output --temp-directory /root/comfy/temp \
  $SAGE > /root/comfy/comfyui.log 2>&1 < /dev/null &
for _ in $(seq 600); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8188/system_stats >/dev/null || finish "comfyui did not come up"
stamp "comfyui up"

for g in $J/graphs/*.json; do
  $PY /workspace/fsh3/run_graphs.py http://127.0.0.1:8188 $OUT/stats.jsonl "$g"
  cp -rf /root/comfy/output/fsh3/. $OUT/ 2>/dev/null
  stamp "done $(basename $g)"
done
finish "all graphs done"
