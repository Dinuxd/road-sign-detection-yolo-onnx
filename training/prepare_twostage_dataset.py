"""
Prepare a two-stage road-sign dataset from Pascal VOC annotations.

Outputs:
  road_sign_twostage_dataset/
    detector_yolo/          # binary YOLO dataset: every annotation is class 0 "sign"
    detector_known_yolo/    # binary YOLO dataset: only useful signs are class 0 "known_sign"
    classifier_crops/       # legacy 13 wanted classes only
    classifier_crops_14class/ # 13 wanted classes + internal other_sign reject class
    reject_calibration/     # legacy validation other_sign crops for reject-threshold calibration
    reports/

Run:
  python prepare_twostage_dataset.py --raw-dir path/to/road_signboard_detection_dataset --recreate
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "road_signboard_detection_dataset"
DEFAULT_OUT_DIR = PROJECT_ROOT / "road_sign_twostage_dataset"

OTHER_CLASS = "other_sign"
KNOWN_CLASSES = [
    "tls-g",
    "sls-40",
    "tls-e",
    "tls-y",
    "sls-50",
    "sls-100",
    "sls-80",
    "no honking",
    "tls-r",
    "sls-60",
    "sls-70",
    "sls-15",
    "tls-c",
]
ALL_CLASSES = [OTHER_CLASS] + KNOWN_CLASSES
INTERNAL_CLASSES = KNOWN_CLASSES + [OTHER_CLASS]
CLASS_TO_DIR = {name: name.replace(" ", "_") for name in ALL_CLASSES}
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


@dataclass(frozen=True)
class Box:
    raw_name: str
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class ImageRecord:
    xml_path: Path
    image_path: Path
    width: int
    height: int
    xml_width: int
    xml_height: int
    boxes: Tuple[Box, ...]


def normalize_class_name(raw_name: str) -> str:
    cleaned = " ".join(raw_name.strip().split()).lower().replace("_", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned == "other sign":
        return OTHER_CLASS
    return cleaned


def related_group_key(path: Path) -> str:
    return re.sub(r"[A-Za-z]+$", "", path.stem).lower()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside project root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def find_image_for_xml(xml_path: Path, raw_dir: Path) -> Optional[Path]:
    direct = xml_path.with_suffix(".jpg")
    if direct.exists():
        return direct
    for ext in IMAGE_EXTENSIONS:
        candidate = raw_dir / f"{xml_path.stem}{ext}"
        if candidate.exists():
            return candidate
    lower_stem = xml_path.stem.lower()
    for candidate in raw_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            if candidate.stem.lower() == lower_stem:
                return candidate
    return None


def parse_int_text(node: Optional[ET.Element], default: int) -> int:
    if node is None or node.text is None:
        return default
    try:
        return int(float(node.text.strip()))
    except ValueError:
        return default


def parse_float_text(node: Optional[ET.Element], default: float = 0.0) -> float:
    if node is None or node.text is None:
        return default
    try:
        return float(node.text.strip())
    except ValueError:
        return default


def parse_record(xml_path: Path, raw_dir: Path, counters: Counter, warnings: List[str]) -> Optional[ImageRecord]:
    image_path = find_image_for_xml(xml_path, raw_dir)
    if image_path is None:
        counters["missing_image"] += 1
        warnings.append(f"No supported image found for {xml_path.name}")
        return None

    try:
        with Image.open(image_path) as img:
            actual_width, actual_height = img.size
    except Exception as exc:
        counters["bad_image"] += 1
        warnings.append(f"Could not open {image_path.name}: {exc}")
        return None

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        counters["bad_xml"] += 1
        warnings.append(f"Could not parse {xml_path.name}: {exc}")
        return None

    size_node = root.find("size")
    xml_width = parse_int_text(size_node.find("width") if size_node is not None else None, actual_width)
    xml_height = parse_int_text(size_node.find("height") if size_node is not None else None, actual_height)
    scale_x = actual_width / xml_width if xml_width else 1.0
    scale_y = actual_height / xml_height if xml_height else 1.0
    if xml_width != actual_width or xml_height != actual_height:
        counters["image_size_mismatch"] += 1

    boxes: List[Box] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        class_name = normalize_class_name(raw_name)
        if class_name not in ALL_CLASSES:
            counters[f"unknown_class:{raw_name}"] += 1
            warnings.append(f"Unknown class in {xml_path.name}: {raw_name!r}")
            continue
        if raw_name and raw_name != class_name:
            counters["normalized_class_name"] += 1

        bbox = obj.find("bndbox")
        if bbox is None:
            counters["missing_bbox"] += 1
            continue

        xmin = parse_float_text(bbox.find("xmin")) * scale_x
        ymin = parse_float_text(bbox.find("ymin")) * scale_y
        xmax = parse_float_text(bbox.find("xmax")) * scale_x
        ymax = parse_float_text(bbox.find("ymax")) * scale_y
        xmin = clamp(xmin, 0, actual_width - 1)
        ymin = clamp(ymin, 0, actual_height - 1)
        xmax = clamp(xmax, 0, actual_width - 1)
        ymax = clamp(ymax, 0, actual_height - 1)
        if xmax <= xmin or ymax <= ymin:
            counters["invalid_bbox"] += 1
            continue
        boxes.append(Box(raw_name, class_name, xmin, ymin, xmax, ymax))

    if not boxes:
        counters["records_without_boxes"] += 1
    return ImageRecord(xml_path, image_path, actual_width, actual_height, xml_width, xml_height, tuple(boxes))


def load_records(raw_dir: Path) -> Tuple[List[ImageRecord], Counter, List[str]]:
    counters: Counter = Counter()
    warnings: List[str] = []
    records: List[ImageRecord] = []
    for xml_path in sorted(raw_dir.glob("*.xml")):
        record = parse_record(xml_path, raw_dir, counters, warnings)
        if record is not None:
            records.append(record)
    return records, counters, warnings


def choose_stratify_key(group: Sequence[ImageRecord], global_counts: Counter) -> str:
    group_classes = {
        box.class_name
        for record in group
        for box in record.boxes
        if box.class_name != OTHER_CLASS
    }
    if group_classes:
        return min(group_classes, key=lambda name: (global_counts[name], name))
    return OTHER_CLASS


def split_records(records: Sequence[ImageRecord], seed: int) -> Dict[str, List[ImageRecord]]:
    rng = random.Random(seed)
    object_counts = Counter(box.class_name for record in records for box in record.boxes)

    groups_by_key: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        groups_by_key[related_group_key(record.image_path)].append(record)

    stratified_groups: Dict[str, List[List[ImageRecord]]] = defaultdict(list)
    for group in groups_by_key.values():
        stratified_groups[choose_stratify_key(group, object_counts)].append(group)

    splits: Dict[str, List[ImageRecord]] = {"train": [], "val": [], "test": []}
    for _, groups in sorted(stratified_groups.items()):
        rng.shuffle(groups)
        n = len(groups)
        if n == 1:
            split_sizes = (1, 0, 0)
        elif n == 2:
            split_sizes = (1, 1, 0)
        else:
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(1, int(round(n * 0.20)))
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
            n_test = max(0, n - n_train - n_val)
            split_sizes = (n_train, n_val, n_test)

        n_train, n_val, _ = split_sizes
        for group in groups[:n_train]:
            splits["train"].extend(group)
        for group in groups[n_train : n_train + n_val]:
            splits["val"].extend(group)
        for group in groups[n_train + n_val :]:
            splits["test"].extend(group)

    for records_in_split in splits.values():
        rng.shuffle(records_in_split)
    return splits


def count_group_leakage(splits: Dict[str, Sequence[ImageRecord]]) -> int:
    seen: Dict[str, set] = defaultdict(set)
    for split, records in splits.items():
        for record in records:
            seen[related_group_key(record.image_path)].add(split)
    return sum(1 for split_names in seen.values() if len(split_names) > 1)


def yolo_line(box: Box, width: int, height: int, class_id: int = 0) -> str:
    cx = ((box.xmin + box.xmax) / 2.0) / width
    cy = ((box.ymin + box.ymax) / 2.0) / height
    bw = (box.xmax - box.xmin) / width
    bh = (box.ymax - box.ymin) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_detector_dataset(
    splits: Dict[str, Sequence[ImageRecord]],
    detector_dir: Path,
    link_mode: str,
    *,
    known_only: bool,
    label_name: str,
) -> Dict[str, object]:
    for split in splits:
        (detector_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (detector_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    transfer_counts = Counter()
    label_counts = Counter()
    empty_label_files = 0
    for split, records in splits.items():
        for record in records:
            dst_image = detector_dir / "images" / split / record.image_path.name
            transfer_counts[link_or_copy(record.image_path, dst_image, link_mode)] += 1
            label_path = detector_dir / "labels" / split / f"{record.image_path.stem}.txt"
            boxes = [box for box in record.boxes if not known_only or box.class_name in KNOWN_CLASSES]
            lines = [yolo_line(box, record.width, record.height) for box in boxes]
            if not lines:
                empty_label_files += 1
            for box in boxes:
                label_counts[box.class_name] += 1
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = detector_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {detector_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                f"  0: {label_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "data_yaml": str(data_yaml),
        "image_transfers": dict(transfer_counts),
        "label_name": label_name,
        "known_only": known_only,
        "label_counts": dict(sorted(label_counts.items())),
        "empty_label_files": empty_label_files,
    }


def expanded_square_crop(box: Box, image_width: int, image_height: int, margin: float) -> Tuple[int, int, int, int]:
    bw = box.xmax - box.xmin
    bh = box.ymax - box.ymin
    cx = (box.xmin + box.xmax) / 2.0
    cy = (box.ymin + box.ymax) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    side = max(side, 1.0)
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

    left = int(math.floor(clamp(left, 0, image_width - 1)))
    top = int(math.floor(clamp(top, 0, image_height - 1)))
    right = int(math.ceil(clamp(right, left + 1, image_width)))
    bottom = int(math.ceil(clamp(bottom, top + 1, image_height)))
    return left, top, right, bottom


def save_crop(image: Image.Image, crop_box: Tuple[int, int, int, int], dst: Path, size: int, quality: int = 95) -> None:
    crop = image.crop(crop_box).resize((size, size), Image.Resampling.LANCZOS)
    crop.save(dst, format="JPEG", quality=quality, optimize=True)


def jitter_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    angle = rng.uniform(-5.0, 5.0)
    fill = tuple(int(x) for x in img.resize((1, 1)).getpixel((0, 0)))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)

    scale = rng.uniform(0.94, 1.08)
    shift_x = rng.uniform(-0.04, 0.04) * img.width
    shift_y = rng.uniform(-0.04, 0.04) * img.height
    new_w = max(2, int(round(img.width * scale)))
    new_h = max(2, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", img.size, fill)
    left = int(round((img.width - new_w) / 2 + shift_x))
    top = int(round((img.height - new_h) / 2 + shift_y))
    canvas.paste(resized, (left, top))
    img = canvas

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.90, 1.10))
    if rng.random() < 0.20:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    if rng.random() < 0.25:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=rng.randint(75, 92))
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")
    return img


def metadata_row(
    *,
    split: str,
    class_name: str,
    relative_path: Path,
    source_image: Path,
    source_xml: Path,
    object_index: int,
    original_box: Tuple[float, float, float, float],
    crop_box: Tuple[int, int, int, int],
    augmentation: str,
) -> Dict[str, object]:
    return {
        "split": split,
        "class_name": class_name,
        "class_dir": CLASS_TO_DIR.get(class_name, ""),
        "relative_path": relative_path.as_posix(),
        "source_image": source_image.name,
        "source_xml": source_xml.name,
        "object_index": object_index,
        "original_xmin": round(original_box[0], 2),
        "original_ymin": round(original_box[1], 2),
        "original_xmax": round(original_box[2], 2),
        "original_ymax": round(original_box[3], 2),
        "crop_left": crop_box[0],
        "crop_top": crop_box[1],
        "crop_right": crop_box[2],
        "crop_bottom": crop_box[3],
        "augmentation": augmentation,
    }


def write_classifier_dataset(
    splits: Dict[str, Sequence[ImageRecord]],
    classifier_dir: Path,
    reject_dir: Optional[Path],
    classes: Sequence[str],
    crop_size: int,
    crop_margin: float,
    target_per_class: int,
    seed: int,
    augment: bool,
    *,
    augment_other: bool = False,
    save_reject_val_other: bool = False,
) -> Dict[str, object]:
    rng = random.Random(seed)
    for split in splits:
        for class_name in classes:
            (classifier_dir / split / CLASS_TO_DIR[class_name]).mkdir(parents=True, exist_ok=True)
    reject_other_dir: Optional[Path] = None
    if reject_dir is not None and save_reject_val_other:
        reject_other_dir = reject_dir / "other_sign_val_crops"
        reject_other_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    original_train_paths: Dict[str, List[Path]] = defaultdict(list)
    original_counts = Counter()
    reject_count = 0
    class_set = set(classes)

    for split, records in splits.items():
        for record in records:
            with Image.open(record.image_path) as raw_img:
                image = raw_img.convert("RGB")
                for object_index, box in enumerate(record.boxes):
                    crop_box = expanded_square_crop(box, record.width, record.height, crop_margin)
                    original_box = (box.xmin, box.ymin, box.xmax, box.ymax)
                    if box.class_name in class_set:
                        class_dir = CLASS_TO_DIR[box.class_name]
                        filename = f"{record.image_path.stem}_{object_index:02d}.jpg"
                        rel_path = Path(split) / class_dir / filename
                        dst = classifier_dir / rel_path
                        save_crop(image, crop_box, dst, crop_size)
                        rows.append(
                            metadata_row(
                                split=split,
                                class_name=box.class_name,
                                relative_path=rel_path,
                                source_image=record.image_path,
                                source_xml=record.xml_path,
                                object_index=object_index,
                                original_box=original_box,
                                crop_box=crop_box,
                                augmentation="original",
                            )
                        )
                        original_counts[(split, box.class_name)] += 1
                        if split == "train":
                            original_train_paths[box.class_name].append(dst)
                    elif (
                        save_reject_val_other
                        and reject_other_dir is not None
                        and split == "val"
                        and box.class_name == OTHER_CLASS
                    ):
                        filename = f"{record.image_path.stem}_{object_index:02d}.jpg"
                        dst = reject_other_dir / filename
                        save_crop(image, crop_box, dst, crop_size)
                        reject_count += 1

    augmented_counts = Counter()
    if augment:
        for class_name in classes:
            if class_name == OTHER_CLASS and not augment_other:
                continue
            originals = list(original_train_paths[class_name])
            if not originals:
                continue
            needed = max(0, target_per_class - len(originals))
            for aug_idx in range(needed):
                src = originals[aug_idx % len(originals)]
                with Image.open(src) as img:
                    augmented = jitter_crop(img, rng)
                class_dir = CLASS_TO_DIR[class_name]
                filename = f"{src.stem}_aug{aug_idx:03d}.jpg"
                rel_path = Path("train") / class_dir / filename
                dst = classifier_dir / rel_path
                augmented.save(dst, format="JPEG", quality=92, optimize=True)
                rows.append(
                    {
                        "split": "train",
                        "class_name": class_name,
                        "class_dir": class_dir,
                        "relative_path": rel_path.as_posix(),
                        "source_image": "",
                        "source_xml": "",
                        "object_index": "",
                        "original_xmin": "",
                        "original_ymin": "",
                        "original_xmax": "",
                        "original_ymax": "",
                        "crop_left": "",
                        "crop_top": "",
                        "crop_right": "",
                        "crop_bottom": "",
                        "augmentation": f"mild_aug_from:{src.relative_to(classifier_dir).as_posix()}",
                    }
                )
                augmented_counts[class_name] += 1

    metadata_path = classifier_dir / "metadata.csv"
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
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = summarize_classifier_counts(classifier_dir, classes)
    (classifier_dir / "class_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    (classifier_dir / "class_map.json").write_text(
        json.dumps(
            {
                "classes": list(classes),
                "class_to_dir": {name: CLASS_TO_DIR[name] for name in classes},
                "public_classes": KNOWN_CLASSES,
                "reject_class": OTHER_CLASS if OTHER_CLASS in classes else "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "metadata_csv": str(metadata_path),
        "original_counts": {f"{split}/{cls}": count for (split, cls), count in sorted(original_counts.items())},
        "augmented_train_counts": dict(sorted(augmented_counts.items())),
        "other_sign_val_reject_crops": reject_count,
    }


def summarize_classifier_counts(classifier_dir: Path, classes: Sequence[str] = KNOWN_CLASSES) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        summary[split] = {}
        for class_name in classes:
            class_dir = classifier_dir / split / CLASS_TO_DIR[class_name]
            summary[split][class_name] = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
    return summary


def summarize_splits(splits: Dict[str, Sequence[ImageRecord]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for split, records in splits.items():
        object_counts = Counter(box.class_name for record in records for box in record.boxes)
        image_counts = Counter()
        for record in records:
            for class_name in {box.class_name for box in record.boxes}:
                image_counts[class_name] += 1
        summary[split] = {
            "images": len(records),
            "objects": sum(object_counts.values()),
            "object_counts": dict(sorted(object_counts.items())),
            "image_counts": dict(sorted(image_counts.items())),
        }
    return summary


def write_report(
    *,
    out_dir: Path,
    raw_dir: Path,
    records: Sequence[ImageRecord],
    counters: Counter,
    warnings: Sequence[str],
    splits: Dict[str, Sequence[ImageRecord]],
    detector_info: Dict[str, object],
    known_detector_info: Dict[str, object],
    classifier_info: Dict[str, object],
    classifier_14_info: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_object_counts = Counter(box.class_name for record in records for box in record.boxes)
    raw_image_counts = Counter()
    for record in records:
        for class_name in {box.class_name for box in record.boxes}:
            raw_image_counts[class_name] += 1

    report = {
        "raw_dataset_dir": str(raw_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "xml_files": len(list(raw_dir.glob("*.xml"))),
        "image_files": sum(1 for path in raw_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        "records_used": len(records),
        "raw_object_counts": dict(sorted(raw_object_counts.items())),
        "raw_image_counts": dict(sorted(raw_image_counts.items())),
        "split_summary": summarize_splits(splits),
        "related_group_split_leakage": count_group_leakage(splits),
        "parser_counters": dict(sorted(counters.items())),
        "warnings_sample": list(warnings[:200]),
        "detector": detector_info,
        "detector_known": known_detector_info,
        "classifier": classifier_info,
        "classifier_14class": classifier_14_info,
        "config": {
            "seed": args.seed,
            "crop_size": args.crop_size,
            "crop_margin": args.crop_margin,
            "target_per_class": args.target_per_class,
            "target_per_class_14": args.target_per_class_14,
            "augment": not args.no_augment,
            "image_link_mode": args.image_link_mode,
        },
    }
    report_path = reports_dir / "prep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    splits_json = {
        split: [
            {
                "image": record.image_path.name,
                "xml": record.xml_path.name,
                "group": related_group_key(record.image_path),
            }
            for record in records_in_split
        ]
        for split, records_in_split in splits.items()
    }
    (reports_dir / "splits.json").write_text(json.dumps(splits_json, indent=2), encoding="utf-8")
    return report


def prepare(args: argparse.Namespace) -> Dict[str, object]:
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    if args.recreate:
        safe_rmtree(out_dir)

    detector_dir = out_dir / "detector_yolo"
    known_detector_dir = out_dir / "detector_known_yolo"
    classifier_dir = out_dir / "classifier_crops"
    classifier_14_dir = out_dir / "classifier_crops_14class"
    reject_dir = out_dir / "reject_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, counters, warnings = load_records(raw_dir)
    if not records:
        raise RuntimeError("No usable records were parsed.")
    splits = split_records(records, seed=args.seed)
    leakage = count_group_leakage(splits)
    if leakage:
        raise RuntimeError(f"Related image groups crossed splits: {leakage}")

    detector_info = write_detector_dataset(
        splits,
        detector_dir,
        args.image_link_mode,
        known_only=False,
        label_name="sign",
    )
    known_detector_info = write_detector_dataset(
        splits,
        known_detector_dir,
        args.image_link_mode,
        known_only=True,
        label_name="known_sign",
    )
    classifier_info = write_classifier_dataset(
        splits,
        classifier_dir,
        reject_dir,
        KNOWN_CLASSES,
        crop_size=args.crop_size,
        crop_margin=args.crop_margin,
        target_per_class=args.target_per_class,
        seed=args.seed,
        augment=not args.no_augment,
        save_reject_val_other=True,
    )
    classifier_14_info = write_classifier_dataset(
        splits,
        classifier_14_dir,
        None,
        INTERNAL_CLASSES,
        crop_size=args.crop_size,
        crop_margin=args.crop_margin,
        target_per_class=args.target_per_class_14,
        seed=args.seed + 17,
        augment=not args.no_augment,
        augment_other=False,
    )
    report = write_report(
        out_dir=out_dir,
        raw_dir=raw_dir,
        records=records,
        counters=counters,
        warnings=warnings,
        splits=splits,
        detector_info=detector_info,
        known_detector_info=known_detector_info,
        classifier_info=classifier_info,
        classifier_14_info=classifier_14_info,
        args=args,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the output dataset folder.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-margin", type=float, default=0.30)
    parser.add_argument("--target-per-class", type=int, default=300)
    parser.add_argument("--target-per-class-14", type=int, default=800)
    parser.add_argument("--image-link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--no-augment", action="store_true", help="Disable offline classifier crop augmentation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = prepare(args)
    print("Prepared two-stage dataset")
    print(f"Output: {args.out_dir.resolve()}")
    print(f"Detector data.yaml: {report['detector']['data_yaml']}")
    print(f"Known detector data.yaml: {report['detector_known']['data_yaml']}")
    print("Split images:", {split: data["images"] for split, data in report["split_summary"].items()})
    print("Classifier train counts:", report["classifier"]["original_counts"])
    print("Augmented train counts:", report["classifier"]["augmented_train_counts"])
    print("14-class classifier train counts:", report["classifier_14class"]["original_counts"])
    print("14-class augmented train counts:", report["classifier_14class"]["augmented_train_counts"])
    print("Reject calibration other_sign crops:", report["classifier"]["other_sign_val_reject_crops"])


if __name__ == "__main__":
    main()
