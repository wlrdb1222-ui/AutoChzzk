#!/bin/bash

set -e

echo "Installing system packages..."
apt-get update -qq
apt-get install -y aria2 ffmpeg -qq

echo "Installing Python packages..."
pip install -q requests faster-whisper tqdm google-genai google-api-python-client google-auth
pip install -q "av<16"
pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12

echo "Registering CUDA 12 libraries..."
CUBLAS_DIR=$(python3 -c "import importlib.util; s = importlib.util.find_spec('nvidia.cublas'); print(list(s.submodule_search_locations)[0] + '/lib')")
echo "$CUBLAS_DIR" > /etc/ld.so.conf.d/cublas12.conf
ldconfig

echo "Setup complete!"
