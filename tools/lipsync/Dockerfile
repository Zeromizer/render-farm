# LatentSync 1.5 exactly as the authors set it up (setup_env.sh at 7b380d6, the last commit
# before the 1.6 release): python 3.10, ffmpeg, libgl1, their requirements.txt unmodified.
# CUDA 12.4 + cuDNN 9 base because onnxruntime-gpu 1.21 (InsightFace) needs cuDNN 9;
# torch brings its own cu121 libraries.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git ffmpeg libgl1 libglib2.0-0 build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python && rm -rf /var/lib/apt/lists/*
ARG LATENTSYNC_REF=7b380d6
RUN git clone https://github.com/bytedance/LatentSync.git /opt/LatentSync \
    && cd /opt/LatentSync && git checkout ${LATENTSYNC_REF}
WORKDIR /opt/LatentSync
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
