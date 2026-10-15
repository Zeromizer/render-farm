#!/usr/bin/env bash
# Benchmark ComfyUI settings on a pod: one case = restart ComfyUI with a given
# python + flags, then warm H3 480p 3 s (x2 after a load run), H3 768p 5 s
# (x2) and YuE2 30 s (x2). Writes timings to /workspace/logs/bench/<case>.txt
# and keeps the last output of each kind for eyeballing.
#   bash cloud/bench.sh <case> <python> "<extra comfyui flags>"
set -u
CASE=$1 PY=$2 FLAGS=${3:-}
COMFY=/root/local/ComfyUI
OUT=/workspace/logs/bench; mkdir -p $OUT
R=$OUT/$CASE.txt; : > $R
LOG=$OUT/comfy-$CASE.log
cd /workspace/render-farm
set -a; . cloud/pod.env; set +a
export COMFYUI_DIR=$COMFY PATH=/workspace/opt/ffmpeg/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor TRITON_CACHE_DIR=/workspace/cache/triton HF_HOME=/workspace/cache/hf

pkill -f "main.py --liste[n]"; sleep 3
SAGE=$($PY -c "import sageattention" 2>/dev/null && echo --use-sage-attention)
( cd $COMFY && setsid $PY -u main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch \
    --input-directory /root/comfy/input --output-directory /root/comfy/output --temp-directory /root/comfy/temp \
    --verbose INFO $SAGE $FLAGS > $LOG 2>&1 < /dev/null & )
for _ in $(seq 180); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 1; done
echo "case=$CASE python=$PY flags='$SAGE $FLAGS' torch=$($PY -c 'import torch;print(torch.__version__)') sage=$($PY -c 'import importlib.metadata as m;print(m.version("sageattention"))' 2>/dev/null)" >> $R

WPY=/root/local/venvs/worker/bin/python
run() {  # label engine json
  local s=$(date +%s.%N)
  local res=$($WPY -u cloud/smoke_engine.py "$2" "$3" 2>&1 | grep -E "^OK|rror|Traceback" | tail -1)
  local secs=$(python3 -c "print(round($(date +%s.%N)-$s,1))")
  local itl=$(sed 's/\x1b\[[0-9;]*m//g' $LOG | grep -oE "[0-9.]+(s/it|it/s)" | tail -1)
  echo "$1: ${secs}s (last step rate $itl) ${res:0:120}" | tee -a $R
}
P='A white electric SUV drives along a coastal road at golden hour, cinematic tracking shot'
run h3_480_load video_gen "{\"prompt\": \"$P\", \"duration_s\": 3, \"resolution\": \"480p\", \"seed\": 1}"
run h3_480_a    video_gen "{\"prompt\": \"$P\", \"duration_s\": 3, \"resolution\": \"480p\", \"seed\": 2}"
run h3_480_b    video_gen "{\"prompt\": \"$P\", \"duration_s\": 3, \"resolution\": \"480p\", \"seed\": 3}"
cp /root/rf-cache/work/smoke-video_gen/video_gen.mp4 $OUT/$CASE-h3_480.mp4 2>/dev/null
run h3_768_a    video_gen "{\"prompt\": \"$P\", \"duration_s\": 5, \"resolution\": \"768p\", \"seed\": 4}"
run h3_768_b    video_gen "{\"prompt\": \"$P\", \"duration_s\": 5, \"resolution\": \"768p\", \"seed\": 5}"
cp /root/rf-cache/work/smoke-video_gen/video_gen.mp4 $OUT/$CASE-h3_768.mp4 2>/dev/null
T='[intro 0:00-0:05]\n[verse 0:05-0:20]\n[outro 0:20-0:30]'
run yue2_a audio_gen "{\"prompt\": \"upbeat electronic pop, 120 bpm, synth bass, bright pads\", \"tags\": \"$T\", \"duration_s\": 30, \"seed\": 1}"
run yue2_b audio_gen "{\"prompt\": \"upbeat electronic pop, 120 bpm, synth bass, bright pads\", \"tags\": \"$T\", \"duration_s\": 30, \"seed\": 2}"
cp /root/rf-cache/work/smoke-audio_gen/*.wav $OUT/$CASE-yue2.wav 2>/dev/null
grep -iE "emulat|fallback|not supported|warn.*(fp8|fp4|int8|convrot)" $LOG | sort | uniq -c | head -8 >> $R
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $R
echo "CASE_DONE $CASE" >> $R
