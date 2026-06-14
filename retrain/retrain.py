"""
Automated retraining with a champion/challenger gate.

Trains a challenger on accumulated + drifted data and promotes it only if it
beats the current champion (lower MAE), unless FORCE_PROMOTE=true.
Intended to run on a schedule (GitHub Actions cron) or on a drift alert.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
import data  # noqa: E402
import registry  # noqa: E402
from train import build_pipeline, evaluate  # noqa: E402

FORCE = os.getenv("FORCE_PROMOTE", "false").lower() == "true"


def main() -> dict:
    # accumulate the original cohort with a fresh (drifted) production batch
    df = pd.concat(
        [data.train_reference(), data.production_sample(n=1500, drift=True)],
        ignore_index=True,
    )
    X, y = df[config.FEATURE_NAMES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE)

    challenger = build_pipeline().fit(X_tr, y_tr)
    new = evaluate(challenger, X_te, y_te)
    new.update({
        "model_type": "GradientBoostingRegressor",
        "n_train": int(len(X_tr)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "lineage": "retrain",
    })

    champion = registry.load_metrics()
    promote = FORCE or registry.is_better(new, champion)

    print(f"champion MAE = {champion.get('mae', 'none')}  |  challenger MAE = {new['mae']}")
    if promote:
        registry.save_model(challenger, new)
        print(f"PROMOTED challenger ({'forced' if FORCE and not registry.is_better(new, champion) else 'better MAE'})")
    else:
        print("KEPT champion (challenger did not improve MAE)")
    return {"promoted": promote, "challenger": new, "champion": champion}


if __name__ == "__main__":
    main()
