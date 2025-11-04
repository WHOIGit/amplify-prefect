# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

FROM docker.io/pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Security updates
RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /ultralytics

# Clone the specific branch from GitHub
RUN git clone --branch v8.3.148 --depth 1 https://github.com/ultralytics/ultralytics.git . && \
    git config --global --add safe.directory /ultralytics

# Install pip packages
RUN pip install uv
RUN uv pip install --system ".[export]" albumentations pycocotools "onnxruntime-gpu" tensorrt wandb

# Clean up
RUN rm -rf /root/.config/Ultralytics/persistent_cache.json

# Copy your custom inference script
COPY docker/amplify-ultralytics/yolo_inference.sh /ultralytics/yolo_inference.sh

