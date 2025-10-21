FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Qwen-Image pipeline (DISABLED - download at runtime to save memory during build)
# Models will be downloaded on first container start (~5-10 min one-time download)
# RUN python3.11 -c "from diffusers import DiffusionPipeline; \
#     import torch; \
#     print('Downloading Qwen-Image pipeline...'); \
#     pipe = DiffusionPipeline.from_pretrained( \
#         'Qwen/Qwen-Image', \
#         torch_dtype=torch.bfloat16, \
#         cache_dir='/app/.cache/huggingface' \
#     ); \
#     print('Pipeline downloaded successfully!')"

# Copy handler and cloud storage module
COPY handler.py .
COPY cloud_storage.py .

# Run handler
CMD ["python3.11", "handler.py"]
