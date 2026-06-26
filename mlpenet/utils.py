from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device_from_config(name: str | None = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "git_commit": git_commit(),
    }


def flatten_ranges(parameters: Mapping[str, list[float]]) -> dict[str, float]:
    ranges = {}
    for name, bounds in parameters.items():
        if len(bounds) != 2:
            raise ValueError(f"Parameter {name} must have [low, high] bounds")
        low, high = float(bounds[0]), float(bounds[1])
        if high <= low:
            raise ValueError(f"Parameter {name} has invalid bounds {bounds}")
        ranges[name] = high - low
    return ranges


def parameter_weights(parameters: Mapping[str, list[float]], explicit: Mapping[str, float] | None) -> list[float]:
    if explicit:
        return [float(explicit[name]) for name in parameters]
    ranges = flatten_ranges(parameters)
    return [1.0 / max(ranges[name], 1e-12) for name in parameters]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
