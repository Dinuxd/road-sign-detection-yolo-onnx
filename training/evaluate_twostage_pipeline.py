"""
Evaluate the final two-stage road-sign pipeline against Pascal VOC ground truth.

Examples:
  python evaluate_twostage_pipeline.py ^
    --predictions-csv runs/twostage_infer/<run_id>/predictions.csv

  python evaluate_twostage_pipeline.py ^
    --detector runs/twostage_detector/yolo26n_known_640/weights/best.pt ^
    --classifier runs/twostage_classifier/<run_id>/best_classifier.pt ^
    --source road_sign_twostage_dataset/detector_known_yolo/images/test ^
    --run-grid-sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from prepare_twostage_dataset import DEFAULT_RAW_DIR, KNOWN_CLASSES, OTHER_CLASS, parse_record
from twostage_infer import IMAGE_EXTENSIONS, load_classifier, make_transform, process_pil_image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "road_sign_twostage_dataset"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "twostage_eval"
DEFAULT_DET_SWEEP = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
DEFAULT_NMS_SWEEP = [0.50, 0.60, 0.70]
DEFAULT_CROP_SWEEP = [0.15, 0.25, 0.30, 0.40]


@dataclass(frozen=True)
class GroundTruth:
    gt_id: str
    image: str
    class_name: str
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    source: str
    image: str
    det_index: int
    det_conf: float
    class_name: str
    class_conf: float
    bbox: Tuple[float, float, float, float]
    accepted: bool
    nms_iou: str = ""
    crop_margin: str = ""

    @property
    def score(self) -> float:
        return self.det_conf * self.class_conf


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def parse_float_list(value: str, default: Sequence[float]) -> List[float]:
    if not value:
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def class_thresholds() -> List[float]:
    return [round(0.20 + i * 0.01, 2) for i in range(80)]


def load_ground_truth(dataset_dir: Path, raw_dir: Path, split: str) -> Dict[str, List[GroundTruth]]:
    splits_path = dataset_dir / "reports" / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits report: {splits_path}")
    split_data = json.loads(splits_path.read_text(encoding="utf-8"))
    if split == "all":
        split_items = [
            item
            for split_name in ("train", "val", "test")
            for item in split_data.get(split_name, [])
        ]
    elif split in split_data:
        split_items = split_data[split]
    else:
        raise ValueError(f"Split {split!r} not found in {splits_path}")

    warnings: List[str] = []
    counters: Counter = Counter()
    by_image: Dict[str, List[GroundTruth]] = defaultdict(list)
    for item in split_items:
        record = parse_record(raw_dir / item["xml"], raw_dir, counters, warnings)
        if record is None:
            continue
        image_key = record.image_path.name.lower()
        for idx, box in enumerate(record.boxes):
            by_image[image_key].append(
                GroundTruth(
                    gt_id=f"{image_key}:{idx}",
                    image=image_key,
                    class_name=box.class_name,
                    bbox=(box.xmin, box.ymin, box.xmax, box.ymax),
                )
            )
    return by_image


def load_predictions_csv(path: Path) -> List[Prediction]:
    predictions: List[Prediction] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get("source") or row.get("image") or ""
            image = Path(source).name.lower()
            predictions.append(
                Prediction(
                    source=source,
                    image=image,
                    det_index=int(float(row.get("det_index", 0) or 0)),
                    det_conf=float(row.get("det_conf", 0.0) or 0.0),
                    class_name=row.get("class_name", ""),
                    class_conf=float(row.get("class_conf", 0.0) or 0.0),
                    bbox=tuple(float(row[key]) for key in ["xmin", "ymin", "xmax", "ymax"]),
                    accepted=str(row.get("accepted", "")).strip().lower() == "true",
                    nms_iou=row.get("nms_iou", ""),
                    crop_margin=row.get("crop_margin", ""),
                )
            )
    return predictions


def accept_existing(prediction: Prediction) -> bool:
    return prediction.accepted and prediction.class_name in KNOWN_CLASSES


def threshold_acceptor(det_conf: float, class_threshold: float, reject_class: str) -> Callable[[Prediction], bool]:
    known = set(KNOWN_CLASSES)

    def accept(prediction: Prediction) -> bool:
        if prediction.det_conf < det_conf:
            return False
        if prediction.class_name == reject_class:
            return False
        if prediction.class_name not in known:
            return False
        return prediction.class_conf >= class_threshold

    return accept


def best_gt_overlap(prediction: Prediction, ground_truth: Dict[str, List[GroundTruth]]) -> Tuple[Optional[GroundTruth], float]:
    best_gt = None
    best_iou = 0.0
    for gt in ground_truth.get(prediction.image, []):
        overlap = iou(prediction.bbox, gt.bbox)
        if overlap > best_iou:
            best_gt = gt
            best_iou = overlap
    return best_gt, best_iou


def compute_metrics(
    predictions: Sequence[Prediction],
    ground_truth: Dict[str, List[GroundTruth]],
    accept_fn: Callable[[Prediction], bool],
    iou_threshold: float,
) -> Dict[str, object]:
    gt_known = {
        image: [gt for gt in items if gt.class_name in KNOWN_CLASSES]
        for image, items in ground_truth.items()
    }
    gt_other = {
        image: [gt for gt in items if gt.class_name == OTHER_CLASS]
        for image, items in ground_truth.items()
    }
    accepted = [prediction for prediction in predictions if accept_fn(prediction)]
    accepted_sorted = sorted(accepted, key=lambda prediction: prediction.score, reverse=True)

    matched_gt = set()
    class_stats = {class_name: Counter() for class_name in KNOWN_CLASSES}
    failure_rows: List[Dict[str, object]] = []
    failure_breakdown: Counter = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)

    for prediction in accepted_sorted:
        best_same_class = None
        best_same_iou = 0.0
        for gt in gt_known.get(prediction.image, []):
            if gt.gt_id in matched_gt or gt.class_name != prediction.class_name:
                continue
            overlap = iou(prediction.bbox, gt.bbox)
            if overlap > best_same_iou:
                best_same_class = gt
                best_same_iou = overlap

        if best_same_class is not None and best_same_iou >= iou_threshold:
            matched_gt.add(best_same_class.gt_id)
            class_stats[prediction.class_name]["tp"] += 1
            confusion[best_same_class.class_name][prediction.class_name] += 1
            continue

        class_stats[prediction.class_name]["fp"] += 1
        best_any, best_any_iou = best_gt_overlap(prediction, ground_truth)
        if best_any is None or best_any_iou < iou_threshold:
            reason = "fp_no_gt_iou50"
            actual_class = ""
        elif best_any.class_name == OTHER_CLASS:
            reason = "fp_other_sign_overlap"
            actual_class = OTHER_CLASS
        elif best_any.class_name != prediction.class_name:
            reason = "fp_wrong_class"
            actual_class = best_any.class_name
            confusion[best_any.class_name][prediction.class_name] += 1
        else:
            reason = "fp_duplicate_or_unmatched"
            actual_class = best_any.class_name
        failure_breakdown[reason] += 1
        failure_rows.append(
            {
                "type": "false_positive",
                "reason": reason,
                "image": prediction.image,
                "actual_class": actual_class,
                "predicted_class": prediction.class_name,
                "iou": round(best_any_iou, 4),
                "det_conf": round(prediction.det_conf, 6),
                "class_conf": round(prediction.class_conf, 6),
                "xmin": round(prediction.bbox[0], 2),
                "ymin": round(prediction.bbox[1], 2),
                "xmax": round(prediction.bbox[2], 2),
                "ymax": round(prediction.bbox[3], 2),
            }
        )

    for image, gt_items in gt_known.items():
        for gt in gt_items:
            if gt.gt_id in matched_gt:
                continue
            class_stats[gt.class_name]["fn"] += 1
            failure_breakdown["fn_missed_known"] += 1
            failure_rows.append(
                {
                    "type": "false_negative",
                    "reason": "fn_missed_known",
                    "image": image,
                    "actual_class": gt.class_name,
                    "predicted_class": "",
                    "iou": "",
                    "det_conf": "",
                    "class_conf": "",
                    "xmin": round(gt.bbox[0], 2),
                    "ymin": round(gt.bbox[1], 2),
                    "xmax": round(gt.bbox[2], 2),
                    "ymax": round(gt.bbox[3], 2),
                }
            )

    tp = sum(stats["tp"] for stats in class_stats.values())
    fp = sum(stats["fp"] for stats in class_stats.values())
    fn = sum(stats["fn"] for stats in class_stats.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    detected_known_ids = set()
    for image, gt_items in gt_known.items():
        for gt in gt_items:
            if any(iou(prediction.bbox, gt.bbox) >= iou_threshold for prediction in accepted if prediction.image == image):
                detected_known_ids.add(gt.gt_id)
    known_gt_total = sum(len(items) for items in gt_known.values())
    other_gt_total = sum(len(items) for items in gt_other.values())

    other_false_accept_ids = set()
    other_false_accept_predictions = 0
    for prediction in accepted:
        best_any, best_any_iou = best_gt_overlap(prediction, ground_truth)
        if best_any is not None and best_any.class_name == OTHER_CLASS and best_any_iou >= iou_threshold:
            other_false_accept_predictions += 1
    for image, gt_items in gt_other.items():
        for gt in gt_items:
            if any(iou(prediction.bbox, gt.bbox) >= iou_threshold for prediction in accepted if prediction.image == image):
                other_false_accept_ids.add(gt.gt_id)

    per_class_rows = []
    per_class = {}
    f1_values = []
    for class_name in KNOWN_CLASSES:
        stats = class_stats[class_name]
        class_precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
        class_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        support = stats["tp"] + stats["fn"]
        f1_values.append(class_f1)
        row = {
            "class_name": class_name,
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "support": support,
        }
        per_class[class_name] = row
        per_class_rows.append(row)

    return {
        "test_images": len(ground_truth),
        "raw_predictions": len(predictions),
        "accepted_predictions": len(accepted),
        "known_gt": known_gt_total,
        "other_gt": other_gt_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "end_to_end_precision": precision,
        "end_to_end_recall": recall,
        "end_to_end_f1": f1,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "known_detection_coverage": len(detected_known_ids) / known_gt_total if known_gt_total else 0.0,
        "other_sign_false_accept_rate": len(other_false_accept_ids) / other_gt_total if other_gt_total else 0.0,
        "other_sign_reject_rate": 1.0 - (len(other_false_accept_ids) / other_gt_total) if other_gt_total else 0.0,
        "other_sign_false_accept_predictions": other_false_accept_predictions,
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
        "confusion_by_class": {actual: dict(preds) for actual, preds in sorted(confusion.items())},
        "per_class": per_class,
        "per_class_rows": per_class_rows,
        "failure_rows": failure_rows,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No image files found at {source}")


def generate_predictions_for_grid(args: argparse.Namespace, run_dir: Path) -> List[Prediction]:
    from ultralytics import YOLO

    if not args.detector or not args.classifier or not args.source:
        raise ValueError("--detector, --classifier, and --source are required for --run-grid-sweep")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector = YOLO(str(args.detector))
    classifier, classes, input_size, _, checkpoint_reject_class = load_classifier(args.classifier, device)
    reject_class = args.reject_class or checkpoint_reject_class
    transform = make_transform(input_size)
    source_images = iter_images(args.source)
    nms_values = parse_float_list(args.nms_sweep, DEFAULT_NMS_SWEEP)
    crop_values = parse_float_list(args.crop_margin_sweep, DEFAULT_CROP_SWEEP)
    min_det_conf = min(parse_float_list(args.det_conf_sweep, DEFAULT_DET_SWEEP))
    generated: List[Prediction] = []
    raw_rows: List[Dict[str, object]] = []

    for nms_iou in nms_values:
        for crop_margin in crop_values:
            for image_path in source_images:
                with Image.open(image_path) as image:
                    _, rows = process_pil_image(
                        image,
                        detector,
                        classifier,
                        transform,
                        classes,
                        device,
                        args.det_imgsz,
                        min_det_conf,
                        nms_iou,
                        0.0,
                        "__evaluate_all_classes__",
                        crop_margin,
                        False,
                    )
                for row in rows:
                    raw = {
                        "source": str(image_path),
                        "nms_iou": nms_iou,
                        "crop_margin": crop_margin,
                        **row,
                    }
                    raw_rows.append(raw)
                    generated.append(
                        Prediction(
                            source=str(image_path),
                            image=image_path.name.lower(),
                            det_index=int(row["det_index"]),
                            det_conf=float(row["det_conf"]),
                            class_name=str(row["class_name"]),
                            class_conf=float(row["class_conf"]),
                            bbox=(float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])),
                            accepted=False,
                            nms_iou=str(nms_iou),
                            crop_margin=str(crop_margin),
                        )
                    )
            print(f"Generated predictions for nms={nms_iou:.2f}, crop_margin={crop_margin:.2f}")

    write_csv(run_dir / "raw_grid_predictions.csv", raw_rows)
    print(f"Internal reject class for generated predictions: {reject_class}")
    return generated


def grouped_prediction_sets(predictions: Sequence[Prediction]) -> Dict[Tuple[str, str], List[Prediction]]:
    groups: Dict[Tuple[str, str], List[Prediction]] = defaultdict(list)
    for prediction in predictions:
        groups[(prediction.nms_iou, prediction.crop_margin)].append(prediction)
    if not groups:
        groups[("", "")] = list(predictions)
    return groups


def build_threshold_sweep(
    predictions: Sequence[Prediction],
    ground_truth: Dict[str, List[GroundTruth]],
    det_values: Sequence[float],
    class_values: Sequence[float],
    reject_class: str,
    iou_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (nms_iou, crop_margin), group_predictions in grouped_prediction_sets(predictions).items():
        for det_conf in det_values:
            for class_threshold in class_values:
                metrics = compute_metrics(
                    group_predictions,
                    ground_truth,
                    threshold_acceptor(det_conf, class_threshold, reject_class),
                    iou_threshold,
                )
                rows.append(
                    {
                        "nms_iou": nms_iou,
                        "crop_margin": crop_margin,
                        "det_conf": det_conf,
                        "classifier_threshold": class_threshold,
                        "precision": metrics["end_to_end_precision"],
                        "recall": metrics["end_to_end_recall"],
                        "f1": metrics["end_to_end_f1"],
                        "macro_f1": metrics["macro_f1"],
                        "accepted_predictions": metrics["accepted_predictions"],
                        "other_sign_false_accept_rate": metrics["other_sign_false_accept_rate"],
                        "known_detection_coverage": metrics["known_detection_coverage"],
                    }
                )
    rows.sort(key=lambda row: row["f1"], reverse=True)
    return rows


def serializable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in metrics.items() if key not in {"per_class_rows", "failure_rows"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--det-conf-threshold", type=float, default=None)
    parser.add_argument("--classifier-threshold", type=float, default=None)
    parser.add_argument("--reject-class", default=OTHER_CLASS)
    parser.add_argument("--det-conf-sweep", default="")
    parser.add_argument("--classifier-threshold-sweep", default="")
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--run-grid-sweep", action="store_true")
    parser.add_argument("--nms-sweep", default="")
    parser.add_argument("--crop-margin-sweep", default="")
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    run_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth(args.dataset_dir, args.raw_dir, args.split)

    if args.run_grid_sweep:
        predictions = generate_predictions_for_grid(args, run_dir)
        accept_fn = threshold_acceptor(
            args.det_conf_threshold if args.det_conf_threshold is not None else min(parse_float_list(args.det_conf_sweep, DEFAULT_DET_SWEEP)),
            args.classifier_threshold if args.classifier_threshold is not None else 0.65,
            args.reject_class,
        )
    else:
        if args.predictions_csv is None:
            raise ValueError("Use --predictions-csv, or provide --run-grid-sweep with detector/classifier/source.")
        predictions = load_predictions_csv(args.predictions_csv)
        if args.det_conf_threshold is None and args.classifier_threshold is None:
            accept_fn = accept_existing
        else:
            accept_fn = threshold_acceptor(
                args.det_conf_threshold if args.det_conf_threshold is not None else 0.0,
                args.classifier_threshold if args.classifier_threshold is not None else 0.0,
                args.reject_class,
            )

    metrics = compute_metrics(predictions, ground_truth, accept_fn, args.iou_threshold)
    sweep_rows = build_threshold_sweep(
        predictions,
        ground_truth,
        parse_float_list(args.det_conf_sweep, DEFAULT_DET_SWEEP),
        parse_float_list(args.classifier_threshold_sweep, class_thresholds()),
        args.reject_class,
        args.iou_threshold,
    )

    (run_dir / "metrics.json").write_text(json.dumps(serializable_metrics(metrics), indent=2), encoding="utf-8")
    write_csv(run_dir / "threshold_sweep.csv", sweep_rows)
    write_csv(run_dir / "per_class_metrics.csv", metrics["per_class_rows"])
    write_csv(run_dir / "failure_cases.csv", metrics["failure_rows"])

    print(f"Saved evaluation to {run_dir}")
    print(
        "End-to-end: "
        f"P={metrics['end_to_end_precision']:.4f} "
        f"R={metrics['end_to_end_recall']:.4f} "
        f"F1={metrics['end_to_end_f1']:.4f} "
        f"accepted={metrics['accepted_predictions']}"
    )
    if sweep_rows:
        best = sweep_rows[0]
        print(
            "Best threshold sweep: "
            f"det={best['det_conf']} cls={best['classifier_threshold']} "
            f"F1={best['f1']:.4f} P={best['precision']:.4f} R={best['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
