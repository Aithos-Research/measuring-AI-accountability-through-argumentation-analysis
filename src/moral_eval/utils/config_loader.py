from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_DIR = Path(os.environ.get("MORAL_EVAL_CONFIG_DIR", _DEFAULT_CONFIG_DIR))
MODELS_FILE = Path(os.environ.get("MORAL_EVAL_MODELS_FILE", CONFIG_DIR / "models.json"))
EXPERIMENTS_FILE = Path(os.environ.get("MORAL_EVAL_EXPERIMENTS_FILE", CONFIG_DIR / "experiments.yaml"))
LOGS_DIR = Path(os.environ.get("MORAL_EVAL_LOGS_DIR", "logs"))


def _fallback_alias(name: str) -> str:
    """Create a predictable alias from a model name if none is provided."""
    return name.lower().replace(" ", "_").replace("-", "_")


@lru_cache(maxsize=1)
def load_models() -> Dict[str, str]:
    """Return alias -> model_id mapping from config/models.json."""
    if not MODELS_FILE.exists():
        raise FileNotFoundError(f"Missing models config at {MODELS_FILE}")

    data = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}

    for family in data.get("models", []):
        for entry in family.get("entries", []):
            alias = entry.get("alias") or _fallback_alias(entry.get("name", ""))
            model_id = entry.get("model_id")
            if not alias or not model_id:
                continue
            mapping[alias] = model_id
    return mapping


@lru_cache(maxsize=1)
def load_experiments() -> Dict[str, Any]:
    """Load experiment configurations from config/experiments.yaml."""
    if not EXPERIMENTS_FILE.exists():
        return {}

    data = yaml.safe_load(EXPERIMENTS_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root of {EXPERIMENTS_FILE}")
    return data


def resolve_models(model_keys: Iterable[str]) -> List[str]:
    """
    Convert a list of model aliases into concrete model IDs.
    Raises if an alias is unknown to enforce centralized configuration.
    """
    mapping = load_models()
    missing = [key for key in model_keys if key not in mapping]
    if missing:
        raise KeyError(f"Unknown model alias(es) in config: {missing}. Please add them to {MODELS_FILE}.")
    return [mapping[key] for key in model_keys]


def get_task_config(task_key: str, variant: str | None = None) -> Dict[str, Any]:
    """
    Fetch configuration for a task (and optional variant) from experiments.yaml.

    If variant is provided, returns experiments[task_key].get(variant, {}).
    Otherwise returns experiments.get(task_key, {}).
    """
    root = load_experiments().get(task_key, {})
    if variant is None:
        return root if isinstance(root, dict) else {}
    if isinstance(root, dict):
        return root.get(variant, {}) if isinstance(root.get(variant, {}), dict) else {}
    return {}


def require_task_config(
    task_key: str, variant: str | None = None, required_keys: Iterable[str] | None = None
) -> Dict[str, Any]:
    """
    Fetch a task config and assert required keys are present.

    Raises KeyError if the task/variant is missing or required keys are absent.
    """
    cfg = get_task_config(task_key, variant)
    if not cfg:
        v = f"/{variant}" if variant is not None else ""
        raise KeyError(f"Missing configuration for task '{task_key}{v}' in {EXPERIMENTS_FILE}")

    if required_keys:
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            v = f"/{variant}" if variant is not None else ""
            raise KeyError(f"Configuration for task '{task_key}{v}' is missing required keys: {missing}")
    return cfg


__all__ = [
    "CONFIG_DIR",
    "MODELS_FILE",
    "EXPERIMENTS_FILE",
    "LOGS_DIR",
    "load_models",
    "load_experiments",
    "resolve_models",
    "get_task_config",
    "require_task_config",
]
