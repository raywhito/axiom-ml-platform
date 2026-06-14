"""
Synthetic data generation for the AXIOM biological-age model.

We simulate a cohort of biomarker panels with a known (but noisy) relationship
to biological age, so the pipeline is fully reproducible and offline. The
`drift` flag shifts the input distributions to emulate a production data shift
(e.g. a new lab or population), which the monitoring layer must detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_NAMES, FEATURES, RANDOM_STATE, TARGET

# Per-biomarker standard deviation for the healthy cohort
_SPREAD = {
    "vitamin_d": 18.0, "ferritin": 45.0, "fasting_glucose": 10.0, "hba1c": 0.4,
    "tsh": 0.8, "hscrp": 0.7, "hdl": 15.0, "ldl": 25.0,
    "triglycerides": 30.0, "vitamin_b12": 180.0,
}

# How strongly a deviation from the optimal midpoint ages the user (years per SD)
_WEIGHT = {
    "vitamin_d": -0.8, "ferritin": -0.5, "fasting_glucose": 1.6, "hba1c": 2.2,
    "tsh": 0.6, "hscrp": 1.8, "hdl": -0.9, "ldl": 1.1,
    "triglycerides": 1.0, "vitamin_b12": -0.4,
}


def generate(n: int = 4000, drift: bool = False, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for f in FEATURE_NAMES:
        mid, sd = FEATURES[f], _SPREAD[f]
        shift = 0.0
        if drift:
            # production shift: glucose/HbA1c/CRP creep up, vitamin D drops
            shift = {"fasting_glucose": 1.4, "hba1c": 1.2, "hscrp": 1.5,
                     "vitamin_d": -1.3}.get(f, 0.3) * sd
        cols[f] = rng.normal(mid + shift, sd, n).clip(min=0.0)
    df = pd.DataFrame(cols)

    # Biological age is driven by the biomarker panel (so it is learnable from
    # the features) plus a small irreducible noise term.
    delta = np.zeros(n)
    for f in FEATURE_NAMES:
        z = (df[f] - FEATURES[f]) / _SPREAD[f]
        delta += _WEIGHT[f] * z
    df[TARGET] = (45.0 + 2.2 * delta + rng.normal(0, 2.5, n)).clip(18, 90).round(2)
    return df


def train_reference(n: int = 4000) -> pd.DataFrame:
    """The reference (training) distribution, also used as the drift baseline."""
    return generate(n=n, drift=False, seed=RANDOM_STATE)


def production_sample(n: int = 1000, drift: bool = True, seed: int = 7) -> pd.DataFrame:
    """A simulated batch of live production data (drifted by default)."""
    return generate(n=n, drift=drift, seed=seed)
