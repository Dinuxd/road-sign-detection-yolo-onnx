#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv python3-picamera2 libopenblas0 libgomp1

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_pi.txt

python -c "import cv2, ncnn, onnxruntime, ultralytics; print('Install check OK')"
