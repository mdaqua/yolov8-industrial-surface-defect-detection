from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from src.utils.logging import setup_logging

LOGGER = setup_logging("defect_detection.inference", "runs/logs/inference.log")


PathLike = Union[str, Path]


class DefectDetector:
    """Small wrapper around Ultralytics YOLO prediction results."""

    def __init__(
        self,
        model_path: PathLike,
        class_names: Optional[Iterable[str]] = None,
        imgsz: int = 640,
        device: Optional[Union[str, int]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weight not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.class_names = list(class_names) if class_names is not None else None
        self.imgsz = imgsz
        self.device = device

    def predict(
        self,
        image_path: PathLike,
        conf: float = 0.25,
        iou: float = 0.7,
        save_visualization: bool = False,
        result_dir: PathLike = "outputs/results",
    ) -> List[Dict[str, Any]]:
        """Run detection on a single image and return JSON-serializable detections."""
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")

        results = self.model.predict(
            source=str(image),
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            device=self.device,
            save=False,
            verbose=False,
        )
        detections = self.parse_results(results)

        if save_visualization:
            self.save_visualization(results, Path(result_dir), image.stem)

        LOGGER.info("Predicted %s detections for %s", len(detections), image)
        return detections

    def parse_results(self, results: Any) -> List[Dict[str, Any]]:
        """Convert Ultralytics result objects into API-friendly dictionaries."""
        detections: List[Dict[str, Any]] = []
        names = self._names()

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = _scalar_to_int(box.cls[0])
                confidence = _scalar_to_float(box.conf[0])
                xyxy = [_scalar_to_float(value) for value in box.xyxy[0]]
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": names.get(class_id, str(class_id)),
                        "confidence": round(confidence, 6),
                        "bbox": {
                            "x1": round(xyxy[0], 3),
                            "y1": round(xyxy[1], 3),
                            "x2": round(xyxy[2], 3),
                            "y2": round(xyxy[3], 3),
                        },
                    }
                )
        return detections

    def save_visualization(self, results: Any, result_dir: Path, stem: str) -> Optional[Path]:
        """Save the first plotted prediction image, if available."""
        result_dir.mkdir(parents=True, exist_ok=True)
        for index, result in enumerate(results):
            plotted = result.plot()
            output_path = result_dir / f"{stem}_prediction_{index}.jpg"
            try:
                import cv2

                cv2.imwrite(str(output_path), plotted)
            except ImportError:
                from PIL import Image

                Image.fromarray(plotted).save(output_path)
            return output_path
        return None

    def _names(self) -> Dict[int, str]:
        if self.class_names is not None:
            return {index: name for index, name in enumerate(self.class_names)}

        model_names = getattr(self.model, "names", {}) or {}
        return {int(key): str(value) for key, value in model_names.items()}


def predict_to_json(
    model_path: PathLike,
    image_path: PathLike,
    output_path: PathLike,
    conf: float = 0.25,
    class_names: Optional[Iterable[str]] = None,
) -> None:
    detector = DefectDetector(model_path=model_path, class_names=class_names)
    detections = detector.predict(image_path=image_path, conf=conf, save_visualization=True)
    payload = {
        "image": str(image_path),
        "defect_count": len(detections),
        "detections": detections,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _scalar_to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _scalar_to_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)
