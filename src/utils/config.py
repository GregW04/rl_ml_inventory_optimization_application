"""Load and resolve project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return repository root (parent of `src/`)."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config; paths in `data` section are resolved relative to project root."""
    root = get_project_root()
    path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    if not path.is_absolute():
        path = root / path

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = root / cfg["data"]["data_dir"]
    cfg["data"]["data_dir"] = str(data_dir)
    for key in ("sales_file", "instock_file", "master_file", "initial_state_file"):
        if key in cfg["data"]:
            cfg["data"][key] = str(data_dir / cfg["data"][key])

    cfg["_project_root"] = str(root)
    return cfg
