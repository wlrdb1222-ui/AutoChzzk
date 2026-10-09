#!/bin/bash

set -e

echo "Installing system packages..."
apt-get update -qq
apt-get install -y aria2 ffmpeg -qq

echo "Installing Python packages..."
pip install requests faster-whisper tqdm google-genai google-api-python-client google-auth
pip install "av<16"
echo "Setup complete!"
