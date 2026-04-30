from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

PathLike = Union[str, Path]

from src.utils.config import save_yaml
from src.utils.images import iter_images


def check_image_label_pairs(image_dir: PathLike, label_dir: PathLike) -> dict:
    """
    Check whether image files and YOLO label files match by stem.

    Returns missing labels, missing images, and matched sample count.
    """
    images = iter_images(image_dir)
    labels_root = Path(label_dir)
    if not labels_root.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_root}")

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels_root.glob("*.txt")}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    return {
        "image_count": len(image_stems),
        "label_count": len(label_stems),
        "matched_count": len(image_stems & label_stems),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
    }


def validate_yolo_label(label_path: PathLike, num_classes: int, allow_empty: bool = True) -> bool:
    """Validate one YOLO detection label file."""
    return not collect_label_errors(label_path, num_classes, allow_empty=allow_empty)


def collect_label_errors(
    label_path: PathLike,
    num_classes: int,
    allow_empty: bool = True,
) -> list[str]:
    """Return line-level validation errors for a YOLO label file."""
    path = Path(label_path)
    if not path.exists():
        return [f"missing label file: {path}"]

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [] if allow_empty else ["empty label file is not allowed"]

    errors: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_no}: expected 5 fields, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"line {line_no}: class_id must be an integer")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(f"line {line_no}: class_id {class_id} outside [0, {num_classes - 1}]")

        for idx, raw_value in enumerate(parts[1:], start=1):
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(f"line {line_no}: coordinate {idx} is not a number")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"line {line_no}: coordinate {idx}={value} outside [0, 1]")
    return errors


def split_dataset(
    image_dir: PathLike,
    label_dir: PathLike,
    output_dir: PathLike,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    require_labels: bool = True,
) -> dict:
    """
    Split image/label pairs into YOLO train/val/test folders.

    Missing labels are skipped by default. If require_labels is false, empty label
    files are created for images without labels, which is useful for negative samples.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    labels_root = Path(label_dir)
    output_root = Path(output_dir)
    images = iter_images(image_dir)

    samples: list[tuple[Path, Optional[Path]]] = []
    skipped_missing_labels: list[str] = []
    for image in images:
        label = labels_root / f"{image.stem}.txt"
        if label.exists():
            samples.append((image, label))
        elif require_labels:
            skipped_missing_labels.append(image.name)
        else:
            samples.append((image, None))

    rng = random.Random(seed)
    rng.shuffle(samples)

    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_samples in splits.items():
        image_output = output_root / "images" / split_name
        label_output = output_root / "labels" / split_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        for image, label in split_samples:
            shutil.copy2(image, image_output / image.name)
            target_label = label_output / f"{image.stem}.txt"
            if label is None:
                target_label.write_text("", encoding="utf-8")
            else:
                shutil.copy2(label, target_label)

    return {
        "total": total,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "skipped_missing_labels": skipped_missing_labels,
    }


def write_data_yaml(
    output_path: PathLike,
    dataset_path: PathLike,
    class_names: Iterable[str],
    train: str = "images/train",
    val: str = "images/val",
    test: str = "images/test",
) -> None:
    """Generate a YOLOv8 data.yaml file."""
    names = {index: name for index, name in enumerate(class_names)}
    save_yaml(
        {
            "path": str(dataset_path),
            "train": train,
            "val": val,
            "test": test,
            "names": names,
        },
        output_path,
    )


def class_distribution(label_dir: PathLike, num_classes: int) -> Dict[int, int]:
    """Count object annotations per class under a label directory."""
    counts: Counter[int] = Counter({index: 0 for index in range(num_classes)})
    for label_path in Path(label_dir).rglob("*.txt"):
        errors = collect_label_errors(label_path, num_classes)
        if errors:
            raise ValueError(f"Invalid label {label_path}: {'; '.join(errors)}")
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts[int(line.split()[0])] += 1
    return dict(sorted(counts.items()))


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8 defect dataset.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-negative", action="store_true")
    args = parser.parse_args()

    pair_report = check_image_label_pairs(args.image_dir, args.label_dir)
    split_report = split_dataset(
        args.image_dir,
        args.label_dir,
        args.output_dir,
        seed=args.seed,
        require_labels=not args.allow_negative,
    )
    write_data_yaml("data/data.yaml", args.output_dir, args.classes)

    print("Pair report:", pair_report)
    print("Split report:", split_report)
    print("data.yaml written to data/data.yaml")


if __name__ == "__main__":
    main()
