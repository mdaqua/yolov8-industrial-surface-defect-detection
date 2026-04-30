from pathlib import Path

from src.dataset.prepare import (
    check_image_label_pairs,
    class_distribution,
    split_dataset,
    validate_yolo_label,
)


def test_validate_yolo_label_accepts_valid_file(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.3 0.4\n", encoding="utf-8")

    assert validate_yolo_label(label, num_classes=2)


def test_validate_yolo_label_rejects_bad_coordinate(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 1.2 0.5 0.2 0.3\n", encoding="utf-8")

    assert not validate_yolo_label(label, num_classes=2)


def test_pair_check_and_split_dataset(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "processed"
    image_dir.mkdir()
    label_dir.mkdir()

    for index in range(10):
        (image_dir / f"img_{index}.jpg").write_bytes(b"fake image bytes")
        (label_dir / f"img_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (label_dir / "orphan.txt").write_text("", encoding="utf-8")

    report = check_image_label_pairs(image_dir, label_dir)
    assert report["matched_count"] == 10
    assert report["missing_images"] == ["orphan"]

    split_report = split_dataset(image_dir, label_dir, output_dir, seed=7)
    assert split_report["total"] == 10
    assert split_report["train"] == 7
    assert split_report["val"] == 2
    assert split_report["test"] == 1
    assert len(list((output_dir / "images" / "train").glob("*.jpg"))) == 7


def test_class_distribution(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "b.txt").write_text("1 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    assert class_distribution(labels, num_classes=3) == {0: 1, 1: 2, 2: 0}
