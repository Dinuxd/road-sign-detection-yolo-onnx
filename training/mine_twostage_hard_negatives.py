"""
Mine hard classifier crops from a two-stage predictions.csv file.

Examples:
  python mine_twostage_hard_negatives.py ^
    --predictions-csv runs/twostage_infer/<train_val_run>/predictions.csv ^
    --splits train val --recreate

Then include the mined metadata during classifier training:
  python train_twostage_classifier.py ^
    --classifier-subdir classifier_crops_14class ^
    --extra-metadata-csv hard_negative_crops/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from prepare_twostage_dataset import (
    CLASS_TO_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_OUT_DIR,
    KNOWN_CLASSES,
    OTHER_CLASS,
    parse_record,
    safe_rmtree,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HARD_NEGATIVE_DIR = PROJECT_ROOT / "hard_negative_crops"


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


def load_ground_truth(dataset_dir: Path, raw_dir: Path, splits: Sequence[str]) -> Dict[str, list]:
    split_path = dataset_dir / "reports" / "splits.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    warnings: List[str] = []
    counters: Counter = Counter()
    by_image: Dict[str, list] = defaultdict(list)
    for split in splits:
        for item in split_data[split]:
            record = parse_record(raw_dir / item["xml"], raw_dir, counters, warnings)
            if record is None:
                continue
            key = record.image_path.name.lower()
            for idx, box in enumerate(record.boxes):
                by_image[key].append(
                    {
                        "id": f"{key}:{idx}",
                        "class_name": box.class_name,
                        "bbox": (box.xmin, box.ymin, box.xmax, box.ymax),
                    }
                )
    return by_image


def best_gt(pred_box: Sequence[float], gt_items: Sequence[dict]) -> Tuple[Optional[dict], float]:
    best = None
    best_iou = 0.0
    for gt in gt_items:
        overlap = iou(pred_box, gt["bbox"])
        if overlap > best_iou:
            best = gt
            best_iou = overlap
    return best, best_iou


def crop_box_from_row(row: dict) -> Tuple[int, int, int, int]:
    if all(row.get(key, "") not in ("", None) for key in ["crop_left", "crop_top", "crop_right", "crop_bottom"]):
        return tuple(int(round(float(row[key]))) for key in ["crop_left", "crop_top", "crop_right", "crop_bottom"])
    return tuple(int(round(float(row[key]))) for key in ["xmin", "ymin", "xmax", "ymax"])


def source_path_from_row(row: dict, dataset_dir: Path) -> Path:
    source = Path(row["source"])
    if source.exists():
        return source
    candidate = dataset_dir.parent / source
    if candidate.exists():
        return candidate
    return source


def mine(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    if args.recreate:
        safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for bucket in ["other_sign", "no_match", "wrong_class"]:
        for class_name in [OTHER_CLASS] + KNOWN_CLASSES:
            (out_dir / bucket / CLASS_TO_DIR[class_name]).mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(args.dataset_dir, args.raw_dir, args.splits)
    rows: List[dict] = []
    counts: Counter = Counter()

    with args.predictions_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for pred_idx, row in enumerate(reader):
            accepted = str(row.get("accepted", "")).strip().lower() == "true"
            if not accepted and not args.include_rejected:
                continue
            predicted_class = row.get("class_name", "")
            pred_box = tuple(float(row[key]) for key in ["xmin", "ymin", "xmax", "ymax"])
            image_key = Path(row["source"]).name.lower()
            matched, matched_iou = best_gt(pred_box, ground_truth.get(image_key, []))

            if matched is None or matched_iou < args.match_iou:
                bucket = "no_match"
                label = OTHER_CLASS
                matched_class = ""
            elif matched["class_name"] == OTHER_CLASS:
                bucket = "other_sign"
                label = OTHER_CLASS
                matched_class = OTHER_CLASS
            elif matched["class_name"] != predicted_class:
                bucket = "wrong_class"
                label = matched["class_name"]
                matched_class = matched["class_name"]
            else:
                continue

            source_path = source_path_from_row(row, args.dataset_dir)
            if not source_path.exists():
                continue
            crop_box = crop_box_from_row(row)
            with Image.open(source_path) as image:
                crop = image.convert("RGB").crop(crop_box).resize((args.crop_size, args.crop_size), Image.Resampling.LANCZOS)
            rel_path = Path(bucket) / CLASS_TO_DIR[label] / f"{source_path.stem}_{pred_idx:06d}.jpg"
            dst = out_dir / rel_path
            crop.save(dst, format="JPEG", quality=95, optimize=True)

            rows.append(
                {
                    "split": "train",
                    "class_name": label,
                    "class_dir": CLASS_TO_DIR[label],
                    "relative_path": rel_path.as_posix(),
                    "source_image": source_path.name,
                    "source_xml": "",
                    "object_index": "",
                    "original_xmin": round(float(row["xmin"]), 2),
                    "original_ymin": round(float(row["ymin"]), 2),
                    "original_xmax": round(float(row["xmax"]), 2),
                    "original_ymax": round(float(row["ymax"]), 2),
                    "crop_left": crop_box[0],
                    "crop_top": crop_box[1],
                    "crop_right": crop_box[2],
                    "crop_bottom": crop_box[3],
                    "augmentation": f"hard_negative:{bucket}",
                    "crop_source": "hard_negative",
                    "det_conf": row.get("det_conf", ""),
                    "matched_class": matched_class,
                    "matched_iou": round(matched_iou, 4),
                    "predicted_class": predicted_class,
                }
            )
            counts[(bucket, label)] += 1

    fieldnames = [
        "split",
        "class_name",
        "class_dir",
        "relative_path",
        "source_image",
        "source_xml",
        "object_index",
        "original_xmin",
        "original_ymin",
        "original_xmax",
        "original_ymax",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "augmentation",
        "crop_source",
        "det_conf",
        "matched_class",
        "matched_iou",
        "predicted_class",
    ]
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "predictions_csv": str(args.predictions_csv),
        "output_dir": str(out_dir),
        "mined_crops": len(rows),
        "counts": {f"{bucket}/{label}": count for (bucket, label), count in sorted(counts.items())},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote hard negatives: {out_dir}")
    print(f"Mined crops: {len(rows)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_HARD_NEGATIVE_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    return parser


if __name__ == "__main__":
    mine(build_arg_parser().parse_args())
