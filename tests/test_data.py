"""Tests for the synthetic data generator."""
from __future__ import annotations

import data
from config import FEATURE_NAMES, TARGET


def test_shape_and_columns():
    df = data.generate(n=500)
    assert len(df) == 500
    for col in FEATURE_NAMES + [TARGET]:
        assert col in df.columns


def test_target_in_plausible_range():
    df = data.generate(n=1000)
    assert df[TARGET].between(18, 90).all()


def test_no_missing_values():
    df = data.generate(n=300)
    assert df.notna().all().all()


def test_drift_shifts_distribution():
    ref = data.generate(n=2000, drift=False)
    cur = data.generate(n=2000, drift=True, seed=7)
    # glucose is configured to creep up under drift
    assert cur["fasting_glucose"].mean() > ref["fasting_glucose"].mean()
