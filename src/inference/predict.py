from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.detector import DefectDetector
from src.utils.config import load_yaml
from src.utils.images import iter_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 defect detection on an image.")
    parser.add_argument("--image", required=True, help="Image file or directory of images.")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--save", action="store_true", help="Save visualization image.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_path = args.model or config["model_path"]
    conf = args.conf if args.conf is not None else float(config.get("confidence_threshold", 0.25))

    detector = DefectDetector(
        model_path=model_path,
        class_names=config.get("class_names"),
        imgsz=int(config.get("image_size", 640)),
    )
    source = Path(args.image)
    image_paths = iter_images(source) if source.is_dir() else [source]

    results = []
    for image_path in image_paths:
        detections = detector.predict(
            image_path,
            conf=conf,
            iou=float(config.get("iou_threshold", 0.7)),
            save_visualization=args.save,
            result_dir=config.get("result_dir", "outputs/results"),
        )
        results.append(
            {
                "image": str(image_path),
                "defect_count": len(detections),
                "detections": detections,
            }
        )

    payload = results[0] if len(results) == 1 else {"image_count": len(results), "results": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
