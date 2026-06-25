"""
Train the binary YOLO sign detector candidates for the two-stage pipeline.

Run after prepare_twostage_dataset.py:
  python train_twostage_detector.py --candidate pi
  python train_twostage_detector.py --candidate reference
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "road_sign_twostage_dataset" / "detector_yolo" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "twostage_detector"

CANDIDATES = {
    "pi": {
        "model": "yolo26n.pt",
        "name": "yolo26n_sign_640",
        "imgsz": 640,
        "batch": 16,
    },
    "reference": {
        "model": "yolo26s.pt",
        "name": "yolo26s_sign_768",
        "imgsz": 768,
        "batch": 16,
    },
}


def train_candidate(candidate_name: str, args: argparse.Namespace):
    cfg = CANDIDATES[candidate_name]
    model_ref = args.model if args.model else cfg["model"]
    if not args.data.exists():
        raise FileNotFoundError(f"Detector data.yaml not found: {args.data}")

    model = YOLO(str(model_ref))
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch if args.batch else cfg["batch"],
        imgsz=args.imgsz if args.imgsz else cfg["imgsz"],
        device=args.device,
        project=str(args.project),
        name=args.name if args.name else cfg["name"],
        exist_ok=args.exist_ok,
        pretrained=True,
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        close_mosaic=25,
        workers=args.workers,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        degrees=3.0,
        translate=0.08,
        scale=0.30,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        cutmix=0.0,
        erasing=0.0,
        plots=True,
        val=True,
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["pi", "reference", "all"], default="pi")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--model", default="", help="Optional override for YOLO weights.")
    parser.add_argument("--name", default="", help="Optional run-name override. Ignored with --candidate all.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidates = ["pi", "reference"] if args.candidate == "all" else [args.candidate]
    for candidate in candidates:
        print(f"Training detector candidate: {candidate}")
        train_candidate(candidate, args)


if __name__ == "__main__":
    main()
