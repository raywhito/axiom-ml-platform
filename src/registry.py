"""Minimal model registry: persist / load the model and its metrics, and
decide whether a challenger should replace the champion (lower MAE wins)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from config import METRICS_PATH, MODEL_PATH, MODELS_DIR


def save_model(model: Any, metrics: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(Path(METRICS_PATH).read_text())


def is_better(new: dict, current: dict, metric: str = "mae") -> bool:
    """Lower MAE is better. A missing current model is always beaten."""
    if not current:
        return True
    return new.get(metric, float("inf")) < current.get(metric, float("inf"))
