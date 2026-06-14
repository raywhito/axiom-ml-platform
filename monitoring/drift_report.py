"""
Production monitoring with Evidently.

Compares the training (reference) distribution against a live (drifted)
production batch, produces an HTML data-drift report, and prints a summary
including the production MAE vs the training MAE (performance monitoring).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
import data  # noqa: E402
import registry  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"


def load_reference() -> pd.DataFrame:
    if config.REFERENCE_DATA_PATH.exists():
        return pd.read_csv(config.REFERENCE_DATA_PATH)
    return data.train_reference()


def main() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    cur = data.production_sample(n=1000, drift=True)

    # ── Evidently data-drift report (HTML artifact) ──────────────────────────
    drift_share, dataset_drift = None, None
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref[config.FEATURE_NAMES],
                   current_data=cur[config.FEATURE_NAMES])
        report.save_html(str(REPORTS / "drift_report.html"))
        res = report.as_dict()["metrics"][0]["result"]
        drift_share = res.get("share_of_drifted_columns")
        dataset_drift = res.get("dataset_drift")
        print(f"Evidently report -> {REPORTS / 'drift_report.html'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[evidently] report skipped ({exc})")

    # ── Performance monitoring: production MAE vs training MAE ────────────────
    prod_mae = None
    try:
        model = registry.load_model()
        pred = model.predict(cur[config.FEATURE_NAMES])
        prod_mae = round(float(mean_absolute_error(cur[config.TARGET], pred)), 4)
    except Exception as exc:  # noqa: BLE001
        print(f"[performance] skipped ({exc})")

    summary = {
        "train_mae": registry.load_metrics().get("mae"),
        "production_mae": prod_mae,
        "share_of_drifted_columns": drift_share,
        "dataset_drift": dataset_drift,
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Monitoring summary:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
