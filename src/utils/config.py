"""Load YAML config into a plain dict."""
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)
