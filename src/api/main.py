from __future__ import annotations

import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.detector import DefectDetector
from src.utils.config import load_yaml
from src.utils.logging import setup_logging

CONFIG_PATH = Path("configs/app.yaml")
CONFIG = load_yaml(CONFIG_PATH)
LOGGER = setup_logging("defect_detection.api", "runs/logs/api.log")

app = FastAPI(
    title="Industrial Surface Defect Detection API",
    description="YOLOv8-based industrial surface defect detection service.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_detector() -> DefectDetector:
    return DefectDetector(
        model_path=CONFIG["model_path"],
        class_names=CONFIG.get("class_names"),
        imgsz=int(CONFIG.get("image_size", 640)),
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    model_path = Path(str(CONFIG["model_path"]))
    return {
        "status": "ok",
        "model_path": str(model_path),
        "model_available": model_path.exists(),
        "model_loaded": get_detector.cache_info().currsize > 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    max_upload_bytes = int(CONFIG.get("max_upload_mb", 20)) * 1024 * 1024
    image_bytes = await file.read()
    if len(image_bytes) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    upload_dir = Path(str(CONFIG.get("upload_dir", "outputs/uploads")))
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        _verify_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    saved_path.write_bytes(image_bytes)

    try:
        detector = get_detector()
        started = time.perf_counter()
        detections = detector.predict(
            saved_path,
            conf=float(CONFIG.get("confidence_threshold", 0.25)),
            iou=float(CONFIG.get("iou_threshold", 0.7)),
            save_visualization=True,
            result_dir=CONFIG.get("result_dir", "outputs/results"),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    except FileNotFoundError as exc:
        LOGGER.exception("Model or image file is missing.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Model runtime error.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    LOGGER.info("Predicted filename=%s count=%s elapsed_ms=%s", file.filename, len(detections), elapsed_ms)
    return {
        "filename": file.filename,
        "defect_count": len(detections),
        "elapsed_ms": elapsed_ms,
        "detections": detections,
    }


def _verify_image(image_bytes: bytes) -> None:
    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file.") from exc
