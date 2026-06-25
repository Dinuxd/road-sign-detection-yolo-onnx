#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_image.sh /path/to/image_or_folder"
  exit 1
fi

source_path="$1"
shift

python run_pi_inference.py --source "$source_path" "$@"
