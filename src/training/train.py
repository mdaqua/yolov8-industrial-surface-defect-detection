from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Union

from src.utils.config import load_yaml
from src.utils.logging import setup_logging

LOGGER = setup_logging("defect_detection.training", "runs/logs/train.log")


def train_model(config_path: Union[str, Path] = "configs/train.yaml", **overrides: Any) -> Path:
    """Train a YOLOv8 detector from a YAML config and return the expected best weight path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    config = load_yaml(config_path)
    config.update({key: value for key, value in overrides.items() if value is not None})

    model_name = config.pop("model", "yolov8n.pt")
    LOGGER.info("Starting YOLOv8 training with model=%s config=%s", model_name, config)

    model = YOLO(model_name)
    result = model.train(**config)
    LOGGER.info("Training finished: %s", result)

    project = Path(str(config.get("project", "runs/train")))
    name = str(config.get("name", "exp"))
    return project / name / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 industrial defect detector.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None, help="Override model, e.g. yolov8n.pt or yolov8s.pt")
    parser.add_argument("--data", default=None, help="Override data.yaml path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    best_path = train_model(
        args.config,
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(f"Expected best weight path: {best_path}")


if __name__ == "__main__":
    main()
