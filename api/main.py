"""
AXIOM biological-age model — serving API (FastAPI).

Endpoints:
  GET  /            service info
  GET  /health      liveness + model readiness
  GET  /model-info  current model type + metrics
  GET  /metrics     Prometheus metrics
  POST /predict     biological-age prediction from a biomarker panel
"""
from __future__ import annotations

import os
import sys
import time

# make the src/ package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)

import predict  # noqa: E402
import registry  # noqa: E402
from schemas import BiomarkerInput, PredictionResponse  # noqa: E402

REQUESTS = Counter("axiom_ml_requests_total", "HTTP requests", ["path", "status"])
PRED_LATENCY = Histogram("axiom_ml_prediction_seconds", "Prediction latency (s)")
PREDICTIONS = Counter("axiom_ml_predictions_total", "Predictions served")
PRED_VALUE = Histogram(
    "axiom_ml_predicted_age", "Distribution of predicted biological age",
    buckets=[20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80],
)
MODEL_MAE = Gauge("axiom_ml_model_mae", "MAE of the deployed model")

app = FastAPI(title="AXIOM Biological-Age API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        predict.get_model()
        MODEL_MAE.set(registry.load_metrics().get("mae", 0.0))
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] model not loaded yet: {exc}")


@app.middleware("http")
async def _observe(request: Request, call_next):
    response = await call_next(request)
    REQUESTS.labels(path=request.url.path, status=response.status_code).inc()
    return response


@app.get("/")
def root():
    return {"service": "AXIOM Biological-Age API", "status": "ok",
            "endpoints": ["/health", "/model-info", "/metrics", "/predict"]}


@app.get("/health")
def health():
    try:
        predict.get_model()
        return {"status": "ok", "model": "loaded"}
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "degraded", "model": "missing"})


@app.get("/model-info")
def model_info():
    m = registry.load_metrics()
    return {"model_type": m.get("model_type", "unknown"),
            "mae": m.get("mae"), "rmse": m.get("rmse"), "r2": m.get("r2"),
            "trained_at": m.get("trained_at")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: BiomarkerInput):
    with PRED_LATENCY.time():
        value = predict.predict_one(payload.model_dump())
    PREDICTIONS.inc()
    PRED_VALUE.observe(value)
    meta = registry.load_metrics()
    return PredictionResponse(
        biological_age=value,
        model_type=meta.get("model_type", "unknown"),
        model_mae=meta.get("mae"),
    )
