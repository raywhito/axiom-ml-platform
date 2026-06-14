"""Central configuration for the AXIOM biological-age model."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
REFERENCE_DATA_PATH = DATA_DIR / "reference.csv"

TARGET = "biological_age"

# Biomarker features used by the model (name -> optimal midpoint, used by the
# synthetic data generator and as serving defaults).
FEATURES: dict[str, float] = {
    "vitamin_d": 65.0,        # ng/mL
    "ferritin": 110.0,        # ng/mL
    "fasting_glucose": 82.0,  # mg/dL
    "hba1c": 5.1,             # %
    "tsh": 1.8,               # mUI/L
    "hscrp": 0.6,             # mg/L
    "hdl": 70.0,              # mg/dL
    "ldl": 85.0,              # mg/dL
    "triglycerides": 75.0,    # mg/dL
    "vitamin_b12": 650.0,     # pg/mL
}

FEATURE_NAMES = list(FEATURES.keys())

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "axiom-biological-age")
RANDOM_STATE = 42
