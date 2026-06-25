#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

python run_pi_inference.py \
  --picamera \
  --detector models/detector_ncnn_416/best_ncnn_model \
  --det-imgsz 416 \
  --width 1280 \
  --height 720 \
  --frame-skip 2 \
  --display \
  "$@"
