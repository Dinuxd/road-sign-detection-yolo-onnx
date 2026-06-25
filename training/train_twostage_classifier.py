"""
Train and calibrate the crop classifier for the two-stage road-sign pipeline.

Run after prepare_twostage_dataset.py:
  python train_twostage_classifier.py --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "road_sign_twostage_dataset"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "twostage_classifier"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_REJECT_CLASS = "other_sign"


@dataclass(frozen=True)
class CropItem:
    image_path: Path
    label: int
    class_name: str
    source: str = "gt"


class CropDataset(Dataset):
    def __init__(self, items: Sequence[CropItem], transform) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item.image_path) as img:
            image = img.convert("RGB")
        return self.transform(image), item.label


class UnlabeledImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), 0


def read_class_map(classifier_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    map_path = classifier_dir / "class_map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing class map: {map_path}")
    data = json.loads(map_path.read_text(encoding="utf-8"))
    return list(data["classes"]), dict(data["class_to_dir"])


def read_items_from_metadata(
    metadata_path: Path,
    split: str,
    classes: Sequence[str],
    *,
    base_dir: Optional[Path] = None,
) -> List[CropItem]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    root_dir = base_dir if base_dir is not None else metadata_path.parent
    items: List[CropItem] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            class_name = row.get("class_name", "")
            if class_name not in class_to_idx:
                continue
            raw_path = Path(row.get("relative_path", ""))
            path = raw_path if raw_path.is_absolute() else root_dir / raw_path
            if path.exists():
                items.append(CropItem(path, class_to_idx[class_name], class_name, row.get("crop_source", "gt")))
    return items


def read_metadata_items(
    classifier_dir: Path,
    split: str,
    classes: Sequence[str],
    extra_metadata: Sequence[Path] = (),
) -> List[CropItem]:
    items = read_items_from_metadata(classifier_dir / "metadata.csv", split, classes, base_dir=classifier_dir)
    if split == "train":
        for metadata_path in extra_metadata:
            items.extend(read_items_from_metadata(metadata_path, split, classes, base_dir=metadata_path.parent))
    return items


def make_transform(input_size: int, *, train: bool = False, augment: bool = False) -> transforms.Compose:
    steps = [transforms.Resize((input_size, input_size))]
    if train and augment:
        steps.extend(
            [
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.04, 0.04),
                    scale=(0.94, 1.08),
                    shear=0,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.15),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def class_weights(items: Sequence[CropItem], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = Counter(item.label for item in items)
    weights = []
    for idx in range(num_classes):
        count = max(1, counts[idx])
        weights.append(math.sqrt(len(items) / (num_classes * count)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_train_loader(
    items: Sequence[CropItem],
    transform,
    batch_size: int,
    workers: int,
    sampler_name: str,
    samples_per_class: int,
    num_classes: int,
) -> DataLoader:
    dataset = CropDataset(items, transform)
    if sampler_name == "shuffle":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    counts = Counter(item.label for item in items)
    weights = [1.0 / max(1, counts[item.label]) for item in items]
    if samples_per_class > 0:
        num_samples = samples_per_class * num_classes
    else:
        num_samples = len(items)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)


def confusion_from_predictions(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    return matrix


def metrics_from_confusion(matrix: List[List[int]], classes: Sequence[str]) -> Dict[str, object]:
    num_classes = len(classes)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(num_classes))
    per_class = {}
    f1_values = []
    for idx, class_name in enumerate(classes):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(num_classes)) - tp
        fn = sum(matrix[idx][col] for col in range(num_classes)) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[idx]),
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / num_classes if num_classes else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def add_public_metrics(metrics: Dict[str, object], classes: Sequence[str], reject_class: str) -> None:
    matrix = metrics["confusion_matrix"]
    known_indices = [idx for idx, name in enumerate(classes) if name != reject_class]
    reject_idx = classes.index(reject_class) if reject_class in classes else None
    known_support = sum(sum(matrix[idx]) for idx in known_indices)
    known_correct = sum(matrix[idx][idx] for idx in known_indices)
    known_f1 = [metrics["per_class"][classes[idx]]["f1"] for idx in known_indices]
    if reject_idx is None:
        other_support = 0
        other_reject_rate = 0.0
    else:
        other_support = sum(matrix[reject_idx])
        other_reject_rate = matrix[reject_idx][reject_idx] / other_support if other_support else 0.0
    metrics["public"] = {
        "known_accuracy": known_correct / known_support if known_support else 0.0,
        "known_macro_f1": sum(known_f1) / len(known_f1) if known_f1 else 0.0,
        "known_support": known_support,
        "other_sign_reject_rate": other_reject_rate,
        "other_sign_support": other_support,
        "reject_class": reject_class if reject_idx is not None else "",
    }


def select_monitor_metric(metrics: Dict[str, object], monitor: str) -> float:
    if monitor == "macro_f1":
        return float(metrics["macro_f1"])
    if monitor == "accuracy":
        return float(metrics["accuracy"])
    public = metrics.get("public", {})
    if monitor == "known_macro_f1":
        return float(public.get("known_macro_f1", metrics["macro_f1"]))
    if monitor == "known_accuracy":
        return float(public.get("known_accuracy", metrics["accuracy"]))
    raise ValueError(f"Unsupported monitor metric: {monitor}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: Sequence[str]) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    losses: List[float] = []
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        losses.append(float(criterion(logits, labels).detach().cpu()))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = metrics_from_confusion(confusion_from_predictions(y_true, y_pred, len(classes)), classes)
    metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
    add_public_metrics(metrics, classes, DEFAULT_REJECT_CLASS)
    return metrics


@torch.no_grad()
def collect_top_confidences(model: nn.Module, loader: DataLoader, device: torch.device) -> List[float]:
    model.eval()
    confidences: List[float] = []
    for images, _ in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
    return confidences


@torch.no_grad()
def collect_prediction_pairs(model: nn.Module, loader: DataLoader, device: torch.device) -> List[Tuple[int, float]]:
    model.eval()
    predictions: List[Tuple[int, float]] = []
    for images, _ in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        confidence, class_idx = probs.max(dim=1)
        predictions.extend((int(idx), float(conf)) for idx, conf in zip(class_idx.cpu().tolist(), confidence.cpu().tolist()))
    return predictions


def calibrate_reject_threshold(
    model: nn.Module,
    classifier_dir: Path,
    reject_dir: Path,
    input_size: int,
    device: torch.device,
    val_items: Sequence[CropItem],
    classes: Sequence[str],
    min_known_keep: float,
    default_threshold: float,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    transform = make_transform(input_size)
    reject_idx = classes.index(DEFAULT_REJECT_CLASS) if DEFAULT_REJECT_CLASS in classes else None
    known_val_items = [item for item in val_items if item.class_name != DEFAULT_REJECT_CLASS]
    known_loader = DataLoader(CropDataset(known_val_items, transform), batch_size=batch_size, shuffle=False, num_workers=workers)

    if reject_idx is None:
        other_paths = sorted((reject_dir / "other_sign_val_crops").glob("*.jpg"))
        other_loader = DataLoader(UnlabeledImageDataset(other_paths, transform), batch_size=batch_size, shuffle=False, num_workers=workers)
        known_conf = collect_top_confidences(model, known_loader, device)
        other_conf = collect_top_confidences(model, other_loader, device) if other_paths else []
        known_pred_pairs = [(0, conf) for conf in known_conf]
        other_pred_pairs = [(0, conf) for conf in other_conf]
    else:
        other_val_items = [item for item in val_items if item.class_name == DEFAULT_REJECT_CLASS]
        other_loader = DataLoader(CropDataset(other_val_items, transform), batch_size=batch_size, shuffle=False, num_workers=workers)
        known_pred_pairs = collect_prediction_pairs(model, known_loader, device)
        other_pred_pairs = collect_prediction_pairs(model, other_loader, device) if other_val_items else []
        known_conf = [conf for _, conf in known_pred_pairs]
        other_conf = [conf for _, conf in other_pred_pairs]
    thresholds = [round(0.20 + i * 0.01, 2) for i in range(76)]

    def known_keep_at(threshold: float) -> float:
        if not known_pred_pairs:
            return 0.0
        kept = 0
        for pred_idx, conf in known_pred_pairs:
            if reject_idx is not None and pred_idx == reject_idx:
                continue
            if conf >= threshold:
                kept += 1
        return kept / len(known_pred_pairs)

    def other_reject_at(threshold: float) -> float:
        if not other_pred_pairs:
            return 0.0
        rejected = 0
        for pred_idx, conf in other_pred_pairs:
            if reject_idx is not None and pred_idx == reject_idx:
                rejected += 1
            elif conf < threshold:
                rejected += 1
        return rejected / len(other_pred_pairs)

    best = {
        "threshold": default_threshold,
        "known_keep_rate": known_keep_at(default_threshold),
        "other_reject_rate": other_reject_at(default_threshold),
    }
    for threshold in thresholds:
        known_keep = known_keep_at(threshold)
        other_reject = other_reject_at(threshold)
        if known_keep < min_known_keep:
            continue
        score = other_reject + 0.05 * known_keep
        current_score = best["other_reject_rate"] + 0.05 * best["known_keep_rate"]
        if score > current_score:
            best = {
                "threshold": threshold,
                "known_keep_rate": known_keep,
                "other_reject_rate": other_reject,
            }

    return {
        **best,
        "default_threshold": default_threshold,
        "min_known_keep": min_known_keep,
        "known_val_count": len(known_conf),
        "other_sign_val_count": len(other_conf),
        "reject_class": DEFAULT_REJECT_CLASS if reject_idx is not None else "",
        "known_confidence_summary": summarize_confidences(known_conf),
        "other_confidence_summary": summarize_confidences(other_conf),
    }


def summarize_confidences(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    def quantile(q: float) -> float:
        index = int(round((len(ordered) - 1) * q))
        return float(ordered[index])
    return {
        "min": float(ordered[0]),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "max": float(ordered[-1]),
    }


def train(args: argparse.Namespace) -> Dict[str, object]:
    classifier_dir = args.dataset_dir / args.classifier_subdir
    reject_dir = args.dataset_dir / "reject_calibration"
    classes, class_to_dir = read_class_map(classifier_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    extra_metadata = [path.resolve() for path in args.extra_metadata_csv]
    train_items = read_metadata_items(classifier_dir, "train", classes, extra_metadata)
    val_items = read_metadata_items(classifier_dir, "val", classes)
    test_items = read_metadata_items(classifier_dir, "test", classes)
    if not train_items or not val_items:
        raise RuntimeError("Training and validation crops are required. Run prepare_twostage_dataset.py first.")

    train_transform = make_transform(args.input_size, train=True, augment=args.train_augment)
    eval_transform = make_transform(args.input_size)
    train_loader = make_train_loader(
        train_items,
        train_transform,
        args.batch_size,
        args.workers,
        args.sampler,
        args.samples_per_class,
        len(classes),
    )
    val_loader = DataLoader(CropDataset(val_items, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(CropDataset(test_items, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_items, len(classes), device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_dir = args.run_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_path = run_dir / "best_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, classes)
        train_loss = running_loss / len(train_loader) if train_loader else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"known_f1={val_metrics['public']['known_macro_f1']:.4f} "
            f"other_reject={val_metrics['public']['other_sign_reject_rate']:.4f}"
        )

        monitor_value = select_monitor_metric(val_metrics, args.monitor)
        if monitor_value > best_metric:
            best_metric = float(monitor_value)
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "input_size": args.input_size,
                    "classes": classes,
                    "class_to_dir": class_to_dir,
                    "public_classes": [name for name in classes if name != DEFAULT_REJECT_CLASS],
                    "reject_class": DEFAULT_REJECT_CLASS if DEFAULT_REJECT_CLASS in classes else "",
                    "val_metrics": val_metrics,
                    "reject_threshold": args.default_reject_threshold,
                    "preprocess": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                    "classifier_subdir": args.classifier_subdir,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epochs.")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model = build_model(checkpoint["arch"], len(checkpoint["classes"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    calibration = calibrate_reject_threshold(
        model,
        classifier_dir,
        reject_dir,
        args.input_size,
        device,
        val_items,
        classes,
        args.min_known_keep,
        args.default_reject_threshold,
        args.batch_size,
        args.workers,
    )
    test_metrics = evaluate(model, test_loader, device, classes) if test_items else {}

    checkpoint["reject_threshold"] = calibration["threshold"]
    checkpoint["calibration"] = calibration
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, best_path)

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_val_macro_f1": best_metric,
        "monitor": args.monitor,
        "history": history,
        "calibration": calibration,
        "test_metrics": test_metrics,
        "classes": classes,
        "public_classes": [name for name in classes if name != DEFAULT_REJECT_CLASS],
        "reject_class": DEFAULT_REJECT_CLASS if DEFAULT_REJECT_CLASS in classes else "",
        "classifier_subdir": args.classifier_subdir,
        "sampler": args.sampler,
        "samples_per_class": args.samples_per_class,
        "label_smoothing": args.label_smoothing,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.export_onnx:
        try:
            export_onnx(model, best_path.with_suffix(".onnx"), args.input_size, device)
        except ModuleNotFoundError as exc:
            print(f"Skipping ONNX export because an optional dependency is missing: {exc.name}")
            print("Install onnxscript and onnx, or rerun without --export-onnx.")
        except Exception as exc:
            print(f"Skipping ONNX export because export failed: {exc}")
    return summary


def export_onnx(model: nn.Module, out_path: Path, input_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )
    print(f"Exported ONNX classifier: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--classifier-subdir", default="classifier_crops_14class")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--arch",
        choices=["mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0", "efficientnet_v2_s"],
        default="mobilenet_v3_large",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0, help="Keep 0 on Windows/Jupyter unless you know it is safe.")
    parser.add_argument("--device", default="", help="Example: cuda, cuda:0, or cpu. Auto-selects when omitted.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--sampler", choices=["balanced", "shuffle"], default="balanced")
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--train-augment", action="store_true", help="Add mild online crop augmentation during training.")
    parser.add_argument("--monitor", choices=["known_macro_f1", "known_accuracy", "macro_f1", "accuracy"], default="known_macro_f1")
    parser.add_argument("--extra-metadata-csv", type=Path, action="append", default=[])
    parser.add_argument("--default-reject-threshold", type=float, default=0.65)
    parser.add_argument("--min-known-keep", type=float, default=0.85)
    parser.add_argument("--export-onnx", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = train(args)
    print("Classifier training complete")
    print(f"Best checkpoint: {summary['best_checkpoint']}")
    print(f"Best monitored val metric: {summary['best_val_metric']:.4f}")
    print(f"Reject threshold: {summary['calibration']['threshold']:.2f}")


if __name__ == "__main__":
    main()
