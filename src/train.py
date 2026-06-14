"""
Train the AXIOM biological-age regressor.

Pipeline: StandardScaler -> GradientBoostingRegressor.
Logs params/metrics/model to MLflow (best-effort) and persists the model,
metrics and the reference dataset (used later as the drift baseline).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import data
import registry


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=config.RANDOM_STATE)),
    ])


def evaluate(pipeline, X_test, y_test) -> dict:
    pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(rmse, 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
    }


def main() -> dict:
    df = data.train_reference()
    X, y = df[config.FEATURE_NAMES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    cv = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    metrics["cv_mae"] = round(float(-cv.mean()), 4)
    metrics["n_train"] = int(len(X_train))
    metrics["trained_at"] = datetime.now(timezone.utc).isoformat()
    metrics["model_type"] = "GradientBoostingRegressor"

    # persist model, metrics and the reference dataset (drift baseline)
    registry.save_model(pipeline, metrics)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.REFERENCE_DATA_PATH, index=False)

    _log_mlflow(pipeline, metrics)
    print("Training complete:", json.dumps(metrics, indent=2))
    return metrics


def _log_mlflow(pipeline, metrics: dict) -> None:
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params(pipeline.named_steps["model"].get_params())
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(pipeline, "model")
        print("Logged run to MLflow:", config.MLFLOW_TRACKING_URI)
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped ({exc})")


if __name__ == "__main__":
    main()
