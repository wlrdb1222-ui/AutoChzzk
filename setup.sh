#!/bin/bash

set -e

echo "Installing system packages..."
apt-get update -qq
apt-get install -y aria2 ffmpeg -qq

echo "Installing Python packages..."
pip install -r requirements.txt -q

echo "Setup complete!"
