"""Inference helper used by the serving API."""
from __future__ import annotations

import pandas as pd

import config
import registry

_model = None


def get_model():
    global _model
    if _model is None:
        _model = registry.load_model()
    return _model


def predict_one(features: dict) -> float:
    """Predict biological age from a dict of biomarker -> value."""
    row = {f: float(features.get(f, config.FEATURES[f])) for f in config.FEATURE_NAMES}
    X = pd.DataFrame([row], columns=config.FEATURE_NAMES)
    return round(float(get_model().predict(X)[0]), 2)
