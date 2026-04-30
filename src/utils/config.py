from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is not installed. Run: pip install -r requirements.txt") from exc

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {yaml_path}")
    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a dictionary as YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is not installed. Run: pip install -r requirements.txt") from exc

    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
