from __future__ import annotations

from pathlib import Path
from typing import Union

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def iter_images(directory: Union[str, Path]) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and is_image_file(path))
