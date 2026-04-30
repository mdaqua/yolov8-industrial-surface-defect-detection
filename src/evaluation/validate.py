from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.logging import setup_logging

LOGGER = setup_logging("defect_detection.evaluation", "runs/logs/val.log")


def validate_model(
    model_path: Union[str, Path] = "weights/best.pt",
    data_path: Union[str, Path] = "data/data.yaml",
    imgsz: int = 640,
    device: Optional[Union[str, int]] = "cpu",
    project: Union[str, Path] = "runs/val",
    name: str = "defect_eval",
    split: str = "val",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Validate a trained YOLOv8 model and return key detection metrics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model weight not found: {model_file}")

    LOGGER.info("Validating model=%s data=%s split=%s", model_file, data_path, split)
    model = YOLO(str(model_file))
    metrics = model.val(
        data=str(data_path),
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        split=split,
        **kwargs,
    )

    class_maps = metrics.box.maps
    if hasattr(class_maps, "tolist"):
        class_maps = class_maps.tolist()

    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "class_map50_95": [float(value) for value in class_maps],
    }
    LOGGER.info("Validation metrics: %s", summary)
    return summary


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 defect detector.")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default="runs/val/metrics.json")
    args = parser.parse_args()

    metrics = validate_model(args.model, args.data, args.imgsz, args.device, split=args.split)
    save_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
