"""
Run two-stage road-sign inference for training/evaluation checkpoints.

For the final Raspberry Pi 4B ONNX deployment package, use:
  ../raspberry_pi_twostage_deploy/run_pi_inference.py

Example:
  python twostage_infer.py ^
    --detector runs/twostage_detector/yolo26n_known_640/weights/best.pt ^
    --classifier runs/twostage_classifier/<run_id>/best_classifier.pt ^
    --source path/to/image_or_video.jpg
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from train_twostage_classifier import IMAGENET_MEAN, IMAGENET_STD, build_model, make_transform


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "twostage_infer"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DEFAULT_REJECT_CLASS = "other_sign"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def expanded_square_xyxy(
    xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = xyxy
    bw = xmax - xmin
    bh = ymax - ymin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    left = cx - side / 2.0
    top = cy - side / 2.0
    right = cx + side / 2.0
    bottom = cy + side / 2.0

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = int(clamp(round(left), 0, image_width - 1))
    top = int(clamp(round(top), 0, image_height - 1))
    right = int(clamp(round(right), left + 1, image_width))
    bottom = int(clamp(round(bottom), top + 1, image_height))
    return left, top, right, bottom


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = list(checkpoint["classes"])
    reject_class = str(checkpoint.get("reject_class") or DEFAULT_REJECT_CLASS)
    model = build_model(checkpoint["arch"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    input_size = int(checkpoint.get("input_size", 224))
    threshold = float(checkpoint.get("reject_threshold", 0.65))
    return model, classes, input_size, threshold, reject_class


@torch.no_grad()
def classify_crop(model, transform, classes: Sequence[str], crop: Image.Image, device: torch.device) -> Dict[str, object]:
    tensor = transform(crop.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0]
    confidence, class_idx = torch.max(probs, dim=0)
    return {
        "class_name": classes[int(class_idx.item())],
        "class_conf": float(confidence.item()),
        "class_index": int(class_idx.item()),
    }


def draw_prediction(draw: ImageDraw.ImageDraw, box: Sequence[float], label: str, color: Tuple[int, int, int]) -> None:
    xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((xmin, ymin), label, font=font)
    text_h = text_bbox[3] - text_bbox[1]
    text_w = text_bbox[2] - text_bbox[0]
    bg_top = max(0, ymin - text_h - 6)
    draw.rectangle([xmin, bg_top, xmin + text_w + 8, bg_top + text_h + 6], fill=color)
    draw.text((xmin + 4, bg_top + 3), label, fill=(255, 255, 255), font=font)


def process_pil_image(
    image: Image.Image,
    detector: YOLO,
    classifier,
    classifier_transform,
    classes: Sequence[str],
    device: torch.device,
    det_imgsz: int,
    det_conf: float,
    det_iou: float,
    classifier_threshold: float,
    reject_class: str,
    crop_margin: float,
    draw_rejected: bool,
) -> Tuple[Image.Image, List[Dict[str, object]]]:
    image = image.convert("RGB")
    width, height = image.size
    result = detector.predict(
        source=np.array(image),
        imgsz=det_imgsz,
        conf=det_conf,
        iou=det_iou,
        verbose=False,
    )[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    predictions: List[Dict[str, object]] = []
    if result.boxes is None:
        return annotated, predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    det_scores = result.boxes.conf.cpu().numpy()
    for det_idx, (xyxy, det_score) in enumerate(zip(boxes, det_scores)):
        crop_box = expanded_square_xyxy(xyxy, width, height, crop_margin)
        crop = image.crop(crop_box)
        cls = classify_crop(classifier, classifier_transform, classes, crop, device)
        if cls["class_name"] == reject_class:
            accepted = False
            reject_reason = "reject_class"
        elif cls["class_conf"] < classifier_threshold:
            accepted = False
            reject_reason = "low_confidence"
        else:
            accepted = True
            reject_reason = ""
        row = {
            "det_index": det_idx,
            "det_conf": float(det_score),
            "xmin": float(xyxy[0]),
            "ymin": float(xyxy[1]),
            "xmax": float(xyxy[2]),
            "ymax": float(xyxy[3]),
            "crop_left": crop_box[0],
            "crop_top": crop_box[1],
            "crop_right": crop_box[2],
            "crop_bottom": crop_box[3],
            "class_name": cls["class_name"],
            "class_conf": cls["class_conf"],
            "accepted": accepted,
            "reject_reason": reject_reason,
        }
        predictions.append(row)
        if accepted:
            label = f"{cls['class_name']} {cls['class_conf']:.2f}"
            draw_prediction(draw, xyxy, label, (0, 160, 70))
        elif draw_rejected:
            label = f"reject {cls['class_conf']:.2f}"
            draw_prediction(draw, xyxy, label, (120, 120, 120))
    return annotated, predictions


def iter_image_sources(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def run_images(args, detector, classifier, classes, classifier_transform, threshold, device, run_dir: Path) -> None:
    images = iter_image_sources(args.source_path)
    if not images:
        raise FileNotFoundError(f"No image files found at {args.source}")
    output_images_dir = run_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for image_path in images:
        with Image.open(image_path) as img:
            annotated, predictions = process_pil_image(
                img,
                detector,
                classifier,
                classifier_transform,
                classes,
                device,
                args.det_imgsz,
                args.det_conf,
                args.det_iou,
                threshold,
                args.reject_class,
                args.crop_margin,
                args.draw_rejected,
            )
        out_path = output_images_dir / image_path.name
        annotated.save(out_path, quality=95)
        for pred in predictions:
            rows.append({"source": str(image_path), "output": str(out_path), **pred})
        print(f"{image_path.name}: {sum(1 for p in predictions if p['accepted'])} accepted / {len(predictions)} detected")

    write_predictions_csv(run_dir / "predictions.csv", rows)


def run_video(args, detector, classifier, classes, classifier_transform, threshold, device, run_dir: Path) -> None:
    import cv2

    source_value = int(args.source) if str(args.source).isdigit() else str(args.source_path)
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/webcam source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = run_dir / "annotated_video.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rows: List[Dict[str, object]] = []

    frame_index = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated, predictions = process_pil_image(
            pil,
            detector,
            classifier,
            classifier_transform,
            classes,
            device,
            args.det_imgsz,
            args.det_conf,
            args.det_iou,
            threshold,
            args.reject_class,
            args.crop_margin,
            args.draw_rejected,
        )
        writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))
        for pred in predictions:
            rows.append({"frame": frame_index, **pred})
        if args.max_frames and frame_index >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = max(1e-6, time.time() - start)
    write_predictions_csv(run_dir / "predictions.csv", rows)
    print(f"Wrote {out_path}")
    print(f"Processed {frame_index} frames at {frame_index / elapsed:.2f} FPS")


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True, help="Binary YOLO sign detector weights.")
    parser.add_argument("--classifier", type=Path, required=True, help="Classifier checkpoint from train_twostage_classifier.py.")
    parser.add_argument("--source", required=True, help="Image path, image folder, video path, or webcam index like 0.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="", help="Example: cuda, cuda:0, or cpu. Auto-selects when omitted.")
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--classifier-threshold", type=float, default=None, help="Override calibrated reject threshold.")
    parser.add_argument("--reject-class", default="", help="Internal classifier class to reject. Defaults to checkpoint value.")
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--draw-rejected", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for video/webcam runs.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.source_path = Path(args.source)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(args.detector))
    classifier, classes, input_size, checkpoint_threshold, checkpoint_reject_class = load_classifier(args.classifier, device)
    threshold = args.classifier_threshold if args.classifier_threshold is not None else checkpoint_threshold
    args.reject_class = args.reject_class or checkpoint_reject_class
    classifier_transform = make_transform(input_size)
    print(f"Using classifier threshold: {threshold:.2f}")
    print(f"Rejecting internal class: {args.reject_class}")

    suffix = args.source_path.suffix.lower()
    if str(args.source).isdigit() or suffix in VIDEO_EXTENSIONS:
        run_video(args, detector, classifier, classes, classifier_transform, threshold, device, run_dir)
    else:
        run_images(args, detector, classifier, classes, classifier_transform, threshold, device, run_dir)


if __name__ == "__main__":
    main()
