"""Tests for the training pipeline and model quality."""
from __future__ import annotations

from sklearn.model_selection import train_test_split

import data
from config import FEATURE_NAMES, TARGET
from train import build_pipeline, evaluate


def test_pipeline_trains_and_predicts():
    df = data.generate(n=1500)
    X, y = df[FEATURE_NAMES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    pipe = build_pipeline().fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(y_te)


def test_model_quality_threshold():
    df = data.generate(n=2500)
    X, y = df[FEATURE_NAMES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    pipe = build_pipeline().fit(X_tr, y_tr)
    metrics = evaluate(pipe, X_te, y_te)
    # the signal is strong by construction — MAE should be comfortably low
    assert metrics["mae"] < 4.0
    assert metrics["r2"] > 0.6
