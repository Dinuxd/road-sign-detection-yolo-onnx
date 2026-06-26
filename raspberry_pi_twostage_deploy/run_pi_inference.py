#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def import_cv2():
    import cv2

    return cv2


def import_onnxruntime():
    import onnxruntime as ort

    return ort


def import_yolo():
    from ultralytics import YOLO

    return YOLO


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_detector(config: Dict[str, Any], override: str | None = None) -> Path:
    if override:
        detector = resolve_path(override)
        if not detector.exists():
            raise FileNotFoundError(f"Detector not found: {detector}")
        return detector

    ncnn_detector = resolve_path(config["detector_ncnn"])
    if ncnn_detector.exists():
        return ncnn_detector

    pt_detector = resolve_path(config["detector_pt"])
    if pt_detector.exists():
        print("NCNN detector missing, falling back to PyTorch .pt detector.")
        return pt_detector

    raise FileNotFoundError(f"No detector found at {ncnn_detector} or {pt_detector}")


def expanded_square_xyxy(
    xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = xyxy
    box_w = xmax - xmin
    box_h = ymax - ymin
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    side = max(box_w, box_h) * (1.0 + 2.0 * margin)

    left = center_x - side / 2.0
    top = center_y - side / 2.0
    right = center_x + side / 2.0
    bottom = center_y + side / 2.0

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

    left = int(max(0, min(round(left), image_width - 1)))
    top = int(max(0, min(round(top), image_height - 1)))
    right = int(max(left + 1, min(round(right), image_width)))
    bottom = int(max(top + 1, min(round(bottom), image_height)))
    return left, top, right, bottom


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


class OnnxCropClassifier:
    def __init__(self, model_path: Path, config: Dict[str, Any], threads: int) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier ONNX not found: {model_path}")

        ort = import_onnxruntime()
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, threads)
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes = list(config["classes"])
        self.input_size = int(config["classifier_input_size"])
        self.mean = np.array(config["imagenet_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(config["imagenet_std"], dtype=np.float32).reshape(1, 1, 3)

    def predict(self, crop_rgb: np.ndarray) -> Dict[str, Any]:
        cv2 = import_cv2()
        resized = cv2.resize(crop_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]

        logits = self.session.run(None, {self.input_name: tensor})[0]
        probabilities = softmax(np.asarray(logits).reshape(-1))
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        return {
            "class_index": class_index,
            "class_name": self.classes[class_index],
            "class_conf": confidence,
        }


def draw_label(frame_bgr: np.ndarray, xyxy: Sequence[float], label: str, color: Tuple[int, int, int]) -> None:
    cv2 = import_cv2()
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_top = max(0, y1 - text_h - baseline - 6)
    cv2.rectangle(frame_bgr, (x1, label_top), (x1 + text_w + 8, label_top + text_h + baseline + 6), color, -1)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 4, label_top + text_h + 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def process_frame(
    frame_bgr: np.ndarray,
    detector: YOLO,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    cv2 = import_cv2()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_bgr.shape[:2]
    predict_kwargs: Dict[str, Any] = {
        "source": frame_rgb,
        "imgsz": args.det_imgsz,
        "conf": args.det_conf,
        "iou": args.det_iou,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    result = detector.predict(**predict_kwargs)[0]
    annotated = frame_bgr.copy()
    predictions: List[Dict[str, Any]] = []

    if result.boxes is None:
        return annotated, predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    det_scores = result.boxes.conf.cpu().numpy()
    reject_class = str(config["reject_class"])

    for det_index, (xyxy, det_score) in enumerate(zip(boxes, det_scores)):
        crop_left, crop_top, crop_right, crop_bottom = expanded_square_xyxy(
            xyxy,
            width,
            height,
            args.crop_margin,
        )
        crop_rgb = frame_rgb[crop_top:crop_bottom, crop_left:crop_right]
        class_result = classifier.predict(crop_rgb)

        reject_reason = ""
        if class_result["class_name"] == reject_class:
            accepted = False
            reject_reason = "reject_class"
        elif class_result["class_conf"] < args.classifier_threshold:
            accepted = False
            reject_reason = "low_confidence"
        else:
            accepted = True

        row = {
            "det_index": det_index,
            "det_conf": float(det_score),
            "xmin": float(xyxy[0]),
            "ymin": float(xyxy[1]),
            "xmax": float(xyxy[2]),
            "ymax": float(xyxy[3]),
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_right": crop_right,
            "crop_bottom": crop_bottom,
            "class_name": class_result["class_name"],
            "class_conf": class_result["class_conf"],
            "accepted": accepted,
            "reject_reason": reject_reason,
        }
        predictions.append(row)

        if accepted:
            label = f"{class_result['class_name']} {class_result['class_conf']:.2f}"
            draw_label(annotated, xyxy, label, (20, 160, 60))
        elif args.draw_rejected:
            label = f"reject {class_result['class_conf']:.2f}"
            draw_label(annotated, xyxy, label, (120, 120, 120))

    return annotated, predictions


def iter_image_paths(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_images(
    source: Path,
    detector: YOLO,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cv2 = import_cv2()
    image_paths = iter_image_paths(source)
    if not image_paths:
        raise FileNotFoundError(f"No images found at {source}")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for image_path in image_paths:
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        annotated, predictions = process_frame(frame_bgr, detector, classifier, config, args)
        output_path = images_dir / image_path.name
        cv2.imwrite(str(output_path), annotated)
        for prediction in predictions:
            rows.append({"source": str(image_path), "output": str(output_path), **prediction})
        accepted_count = sum(1 for prediction in predictions if prediction["accepted"])
        print(f"{image_path.name}: {accepted_count} accepted / {len(predictions)} detected")

    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    print(f"Saved outputs to {args.out_dir}")


def open_video_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    cv2 = import_cv2()
    source_value: int | str
    if str(args.source).isdigit():
        source_value = int(args.source)
    else:
        source_value = str(args.source)

    capture = cv2.VideoCapture(source_value)
    if str(args.source).isdigit():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    return capture


def run_video_capture(
    detector: YOLO,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cv2 = import_cv2()
    capture = open_video_capture(args)
    writer = None
    rows: List[Dict[str, Any]] = []
    frame_index = 0
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % args.frame_skip == 0:
                annotated, predictions = process_frame(frame_bgr, detector, classifier, config, args)
                processed_count += 1
                for prediction in predictions:
                    rows.append({"frame": frame_index, **prediction})
            else:
                annotated = frame_bgr
                predictions = []

            writer = maybe_write_video_frame(writer, annotated, args)

            if args.display:
                cv2.imshow("Road sign detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if processed_count and processed_count % 30 == 0:
                elapsed = max(1e-6, time.time() - start_time)
                labels = [p["class_name"] for p in predictions if p["accepted"]]
                print(f"frames={frame_index} processed={processed_count} fps={processed_count / elapsed:.2f} accepted={labels}")

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    elapsed = max(1e-6, time.time() - start_time)
    print(f"Processed {processed_count} inference frames from {frame_index} camera/video frames.")
    print(f"Inference FPS: {processed_count / elapsed:.2f}")
    print(f"Saved outputs to {args.out_dir}")


def run_picamera(
    detector: YOLO,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cv2 = import_cv2()
    from picamera2 import Picamera2

    camera = Picamera2()
    camera_config = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)}
    )
    camera.configure(camera_config)
    camera.start()
    time.sleep(1.0)

    writer = None
    rows: List[Dict[str, Any]] = []
    frame_index = 0
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            frame_rgb = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_index += 1

            if frame_index % args.frame_skip == 0:
                annotated, predictions = process_frame(frame_bgr, detector, classifier, config, args)
                processed_count += 1
                for prediction in predictions:
                    rows.append({"frame": frame_index, **prediction})
            else:
                annotated = frame_bgr
                predictions = []

            writer = maybe_write_video_frame(writer, annotated, args)

            if args.display:
                cv2.imshow("Road sign detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if processed_count and processed_count % 30 == 0:
                elapsed = max(1e-6, time.time() - start_time)
                labels = [p["class_name"] for p in predictions if p["accepted"]]
                print(f"frames={frame_index} processed={processed_count} fps={processed_count / elapsed:.2f} accepted={labels}")

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        camera.stop()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    elapsed = max(1e-6, time.time() - start_time)
    print(f"Processed {processed_count} inference frames from {frame_index} camera frames.")
    print(f"Inference FPS: {processed_count / elapsed:.2f}")
    print(f"Saved outputs to {args.out_dir}")


def maybe_write_video_frame(writer, frame_bgr: np.ndarray, args: argparse.Namespace):
    cv2 = import_cv2()
    if not args.save_video:
        return writer

    save_path = Path(args.save_video)
    if not save_path.is_absolute():
        save_path = args.out_dir / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if writer is None:
        height, width = frame_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.output_fps,
            (width, height),
        )
    writer.write(frame_bgr)
    return writer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-stage road-sign model on Raspberry Pi.")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="Path to config.json.")
    parser.add_argument("--detector", default="", help="Optional detector override. Defaults to NCNN, then .pt fallback.")
    parser.add_argument("--classifier", default="", help="Optional classifier ONNX override.")
    parser.add_argument("--source", default="0", help="Image, folder, video path, or USB webcam index. Default: 0.")
    parser.add_argument("--picamera", action="store_true", help="Use Raspberry Pi Camera through Picamera2.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"), help="Folder for annotated images/video and CSV.")
    parser.add_argument("--display", action="store_true", help="Show live window. Press q to quit.")
    parser.add_argument("--save-video", default="", help="Optional output video filename, for example road_test.mp4.")
    parser.add_argument("--output-fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frame-skip", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for testing.")
    parser.add_argument("--device", default="", help="Leave empty on Raspberry Pi. Use cpu only if needed.")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime CPU threads.")
    parser.add_argument("--det-imgsz", type=int, default=0)
    parser.add_argument("--det-conf", type=float, default=-1.0)
    parser.add_argument("--det-iou", type=float, default=-1.0)
    parser.add_argument("--classifier-threshold", type=float, default=-1.0)
    parser.add_argument("--crop-margin", type=float, default=-1.0)
    parser.add_argument("--draw-rejected", action="store_true")
    return parser


def apply_config_defaults(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    args.config = Path(args.config)
    args.out_dir = Path(args.out_dir)
    args.source_path = Path(args.source)
    args.frame_skip = max(1, args.frame_skip)
    args.det_imgsz = args.det_imgsz or int(config["det_imgsz"])
    args.det_conf = float(config["det_conf"]) if args.det_conf < 0 else args.det_conf
    args.det_iou = float(config["det_iou"]) if args.det_iou < 0 else args.det_iou
    args.classifier_threshold = (
        float(config["classifier_threshold"]) if args.classifier_threshold < 0 else args.classifier_threshold
    )
    args.crop_margin = float(config["crop_margin"]) if args.crop_margin < 0 else args.crop_margin
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(Path(args.config))
    args = apply_config_defaults(args, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    detector_path = choose_detector(config, args.detector or None)
    classifier_path = resolve_path(args.classifier) if args.classifier else resolve_path(config["classifier_onnx"])

    print(f"Detector: {detector_path}")
    print(f"Classifier: {classifier_path}")
    print(
        "Thresholds: "
        f"det_conf={args.det_conf:.2f}, det_iou={args.det_iou:.2f}, "
        f"classifier={args.classifier_threshold:.2f}, crop_margin={args.crop_margin:.2f}"
    )

    YOLO = import_yolo()
    detector = YOLO(str(detector_path), task="detect")
    classifier = OnnxCropClassifier(classifier_path, config, args.threads)

    source_suffix = args.source_path.suffix.lower()
    if args.picamera:
        run_picamera(detector, classifier, config, args)
    elif args.source_path.exists() and (source_suffix in IMAGE_EXTENSIONS or args.source_path.is_dir()):
        run_images(args.source_path, detector, classifier, config, args)
    elif str(args.source).isdigit() or source_suffix in VIDEO_EXTENSIONS:
        run_video_capture(detector, classifier, config, args)
    else:
        raise FileNotFoundError(f"Source not found or unsupported: {args.source}")


if __name__ == "__main__":
    main()
