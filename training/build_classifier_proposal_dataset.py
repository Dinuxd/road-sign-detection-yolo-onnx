"""
Build a 14-class classifier dataset that mixes GT crops with detector proposal crops.

Run after preparing classifier_crops_14class and training a detector candidate:
  python build_classifier_proposal_dataset.py ^
    --detector runs/twostage_detector/yolo26n_known_640/weights/best.pt ^
    --det-imgsz 640 --det-conf 0.01 --recreate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from ultralytics import YOLO

from prepare_twostage_dataset import (
    CLASS_TO_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_OUT_DIR,
    INTERNAL_CLASSES,
    KNOWN_CLASSES,
    OTHER_CLASS,
    expanded_square_crop,
    parse_record,
    safe_rmtree,
    save_crop,
)


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Proposal:
    split: str
    source_image: Path
    source_xml: Path
    det_index: int
    det_conf: float
    bbox: Tuple[float, float, float, float]
    label: str
    matched_class: str
    matched_iou: float


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


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def load_split_records(dataset_dir: Path, raw_dir: Path) -> Dict[str, list]:
    split_path = dataset_dir / "reports" / "splits.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    warnings: List[str] = []
    counters: Counter = Counter()
    splits: Dict[str, list] = {}
    for split, rows in split_data.items():
        splits[split] = []
        for row in rows:
            record = parse_record(raw_dir / row["xml"], raw_dir, counters, warnings)
            if record is not None:
                splits[split].append(record)
    return splits


def assign_label(record, bbox: Tuple[float, float, float, float], match_iou: float) -> Tuple[str, str, float]:
    best_known = None
    best_known_iou = 0.0
    best_other = None
    best_other_iou = 0.0
    for box in record.boxes:
        gt_bbox = (box.xmin, box.ymin, box.xmax, box.ymax)
        overlap = iou(bbox, gt_bbox)
        if box.class_name in KNOWN_CLASSES and overlap > best_known_iou:
            best_known = box
            best_known_iou = overlap
        elif box.class_name == OTHER_CLASS and overlap > best_other_iou:
            best_other = box
            best_other_iou = overlap

    if best_known is not None and best_known_iou >= match_iou:
        return best_known.class_name, best_known.class_name, best_known_iou
    if best_other is not None and best_other_iou >= match_iou:
        return OTHER_CLASS, OTHER_CLASS, best_other_iou
    return OTHER_CLASS, "", max(best_known_iou, best_other_iou)


def copy_gt_dataset(source_dir: Path, out_dir: Path) -> Tuple[List[Dict[str, object]], Counter]:
    metadata_path = source_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing 14-class metadata: {metadata_path}")
    rows: List[Dict[str, object]] = []
    counts: Counter = Counter()
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["class_name"]
            if class_name not in INTERNAL_CLASSES:
                continue
            src = source_dir / row["relative_path"]
            dst_rel = Path(row["relative_path"])
            dst = out_dir / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(src, dst)
            new_row = dict(row)
            new_row["crop_source"] = "gt"
            new_row["det_conf"] = ""
            new_row["matched_class"] = class_name
            new_row["matched_iou"] = "1.0"
            rows.append(new_row)
            counts[(row["split"], class_name)] += 1
    return rows, counts


def collect_proposals(args: argparse.Namespace, splits: Dict[str, list]) -> List[Proposal]:
    detector = YOLO(str(args.detector))
    proposals: List[Proposal] = []
    for split in args.splits:
        for record in splits[split]:
            result = detector.predict(
                source=str(record.image_path),
                imgsz=args.det_imgsz,
                conf=args.det_conf,
                iou=args.det_iou,
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for det_index, (xyxy, score) in enumerate(zip(boxes, scores)):
                bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                label, matched_class, matched_iou = assign_label(record, bbox, args.match_iou)
                proposals.append(
                    Proposal(
                        split=split,
                        source_image=record.image_path,
                        source_xml=record.xml_path,
                        det_index=det_index,
                        det_conf=float(score),
                        bbox=bbox,
                        label=label,
                        matched_class=matched_class,
                        matched_iou=matched_iou,
                    )
                )
        print(f"Collected detector proposals for {split}: {sum(1 for p in proposals if p.split == split)}")
    return proposals


def limit_proposals(
    proposals: Sequence[Proposal],
    gt_counts: Counter,
    proposal_fraction: float,
    seed: int,
) -> List[Proposal]:
    rng = random.Random(seed)
    by_split: Dict[str, List[Proposal]] = defaultdict(list)
    for proposal in proposals:
        by_split[proposal.split].append(proposal)

    selected: List[Proposal] = []
    for split, split_proposals in by_split.items():
        gt_total = sum(count for (count_split, _), count in gt_counts.items() if count_split == split)
        if proposal_fraction <= 0 or proposal_fraction >= 1:
            target = len(split_proposals)
        else:
            target = int(round(gt_total * proposal_fraction / (1.0 - proposal_fraction)))
        rng.shuffle(split_proposals)
        selected.extend(split_proposals[: min(target, len(split_proposals))])
    return selected


def write_proposal_crops(
    proposals: Sequence[Proposal],
    out_dir: Path,
    crop_size: int,
    crop_margin: float,
) -> Tuple[List[Dict[str, object]], Counter]:
    rows: List[Dict[str, object]] = []
    counts: Counter = Counter()
    for proposal in proposals:
        class_dir = CLASS_TO_DIR[proposal.label]
        filename = f"{proposal.source_image.stem}_det{proposal.det_index:03d}_{len(rows):06d}.jpg"
        rel_path = Path(proposal.split) / class_dir / filename
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(proposal.source_image) as image:
            crop_box = expanded_square_crop(
                type(
                    "BoxLike",
                    (),
                    {
                        "xmin": proposal.bbox[0],
                        "ymin": proposal.bbox[1],
                        "xmax": proposal.bbox[2],
                        "ymax": proposal.bbox[3],
                    },
                )(),
                image.width,
                image.height,
                crop_margin,
            )
            save_crop(image.convert("RGB"), crop_box, dst, crop_size)
        rows.append(
            {
                "split": proposal.split,
                "class_name": proposal.label,
                "class_dir": class_dir,
                "relative_path": rel_path.as_posix(),
                "source_image": proposal.source_image.name,
                "source_xml": proposal.source_xml.name,
                "object_index": "",
                "original_xmin": round(proposal.bbox[0], 2),
                "original_ymin": round(proposal.bbox[1], 2),
                "original_xmax": round(proposal.bbox[2], 2),
                "original_ymax": round(proposal.bbox[3], 2),
                "crop_left": crop_box[0],
                "crop_top": crop_box[1],
                "crop_right": crop_box[2],
                "crop_bottom": crop_box[3],
                "augmentation": "detector_proposal",
                "crop_source": "proposal",
                "det_conf": round(proposal.det_conf, 6),
                "matched_class": proposal.matched_class,
                "matched_iou": round(proposal.matched_iou, 4),
            }
        )
        counts[(proposal.split, proposal.label)] += 1
    return rows, counts


def write_metadata(out_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
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
    ]
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_counts(out_dir: Path) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        summary[split] = {}
        for class_name in INTERNAL_CLASSES:
            summary[split][class_name] = len(list((out_dir / split / CLASS_TO_DIR[class_name]).glob("*.jpg")))
    return summary


def build(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    source_dir = dataset_dir / "classifier_crops_14class"
    out_dir = dataset_dir / args.out_subdir
    if args.recreate:
        safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for class_name in INTERNAL_CLASSES:
            (out_dir / split / CLASS_TO_DIR[class_name]).mkdir(parents=True, exist_ok=True)

    gt_rows, gt_counts = copy_gt_dataset(source_dir, out_dir)
    splits = load_split_records(dataset_dir, args.raw_dir.resolve())
    proposals = collect_proposals(args, splits)
    selected = limit_proposals(proposals, gt_counts, args.proposal_fraction, args.seed)
    proposal_rows, proposal_counts = write_proposal_crops(selected, out_dir, args.crop_size, args.crop_margin)
    rows = gt_rows + proposal_rows
    write_metadata(out_dir, rows)
    (out_dir / "class_map.json").write_text(
        json.dumps(
            {
                "classes": INTERNAL_CLASSES,
                "class_to_dir": {name: CLASS_TO_DIR[name] for name in INTERNAL_CLASSES},
                "public_classes": KNOWN_CLASSES,
                "reject_class": OTHER_CLASS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "class_counts.json").write_text(json.dumps(summarize_counts(out_dir), indent=2), encoding="utf-8")
    report = {
        "source_gt_dir": str(source_dir),
        "output_dir": str(out_dir),
        "proposal_fraction": args.proposal_fraction,
        "proposal_candidates": len(proposals),
        "proposal_selected": len(selected),
        "gt_counts": {f"{split}/{cls}": count for (split, cls), count in sorted(gt_counts.items())},
        "proposal_counts": {f"{split}/{cls}": count for (split, cls), count in sorted(proposal_counts.items())},
    }
    (out_dir / "proposal_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote proposal classifier dataset: {out_dir}")
    print(f"GT crops: {len(gt_rows)}")
    print(f"Proposal crops: {len(proposal_rows)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-subdir", default="classifier_proposal_crops_14class")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.01)
    parser.add_argument("--det-iou", type=float, default=0.60)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--proposal-fraction", type=float, default=0.30)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate", action="store_true")
    return parser


if __name__ == "__main__":
    build(build_arg_parser().parse_args())
