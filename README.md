# AXIOM ML Platform — Block 4 (MLOps)

An **industrialised, end-to-end AI solution** for AXIOM: a model that predicts a
user's **biological age** from a biomarker panel, served through a FastAPI
prediction API, packaged with Docker, deployed via CI/CD, retrained
automatically (champion/challenger), and monitored in production for **data
drift** (Evidently) and **performance**.

> Fictional project for the Block 4 "AI Solutions" assessment. Builds on the
> AXIOM data platform (Blocks 2–3). Designed to run **alongside** the Block 3
> stack (offset ports).

---

## 1. Specifications (business framing)

- **Problem.** AXIOM's headline metric is biological age. Computing it from raw
  biomarkers is valuable but must stay accurate as lab populations shift.
- **ML task.** Supervised **regression**: biomarker panel → biological age (years).
- **Target metric.** Mean Absolute Error (MAE) in years; secondary R².
- **Constraints.** Low-latency serving, reproducible training, automated
  retraining on drift, full observability.

## 2. ML solution

- **Model.** `StandardScaler → GradientBoostingRegressor` (scikit-learn Pipeline).
- **Data.** Reproducible synthetic cohort (`src/data.py`) with a known,
  noisy biomarker→age relationship; a `drift` switch simulates production shift.
- **Tracking.** Runs, params, metrics and the model are logged to **MLflow**.
- **Quality gate.** CI fails if `MAE ≥ 4` or `R² ≤ 0.6`.

## 3. Architecture

```
        biomarker panel (JSON)
                │
                ▼
        FastAPI  /predict     ──►  Prometheus ──► Grafana   (serving metrics, latency)
        (model.joblib)                                  
                ▲                                   
   train.py ────┘  ── logs ──►  MLflow  (experiments, model registry)
                                                    
   retrain/retrain.py  (champion/challenger)        
   monitoring/drift_report.py ──►  Evidently HTML  (data drift + production MAE)

   CI/CD: GitHub Actions  →  test + train + quality gate  →  build image (GHCR)  →  deploy (K8s)
```

Diagrams: [`docs/architecture.png`](docs/architecture.png),
[`docs/mlops_lifecycle.png`](docs/mlops_lifecycle.png).

## 4. Repository structure

```
axiom-ml-platform/
├── notebooks/   # EDA notebook
├── src/         # data, features, train, predict, registry, config
├── tests/       # pytest (data, training quality gate, API contract)
├── models/      # serialized model + metrics (produced by training)
├── api/         # FastAPI serving app (+ Prometheus metrics)
├── k8s/         # Deployment, Service, HPA
├── monitoring/  # Evidently drift report + Prometheus/Grafana
├── retrain/     # champion/challenger retraining
├── .github/workflows/   # CI, CD, scheduled retraining
├── Dockerfile · docker-compose.yml · requirements.txt · Makefile
```

## 5. CI/CD (`.github/workflows/`)

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | push / PR | install, **pytest**, train, **quality gate**, upload model artifact |
| `cd.yml` | push to main / tag | build & push image to **GHCR**, deploy step (K8s) |
| `retrain.yml` | weekly cron / manual | retrain (champion/challenger) + drift report, upload artifacts |

## 6. Automated retraining

`retrain/retrain.py` trains a **challenger** on accumulated + drifted data and
**promotes it only if it beats the champion's MAE** (override with
`FORCE_PROMOTE=true`). Scheduled via `retrain.yml` (cron) — can also be wired to
fire on a drift alert.

## 7. Monitoring

- **Serving** — Prometheus scrapes the API (`/metrics`): prediction rate, p95
  latency, predicted-age distribution, deployed-model MAE → **Grafana** dashboard.
- **Drift & performance** — `monitoring/drift_report.py` uses **Evidently** to
  compare the training reference vs a live production batch and reports the share
  of drifted features and the production MAE vs training MAE.

---

## 8. Quickstart (local — zero cost)

Prerequisites: **Docker Desktop**. Ports `8000 / 5001 / 9091 / 3001 / 8090`
(chosen to coexist with the Block 3 stack).

```bash
cd axiom-ml-platform
make demo        # up + train + sample traffic + drift report
```

| Service | URL |
|---|---|
| Prediction API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3001 (admin/admin) |
| Prometheus | http://localhost:9091 |
| Evidently drift report | http://localhost:8090/drift_report.html |

Try a prediction:
```bash
curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' \
  -d '{"fasting_glucose":104,"hba1c":5.9,"hscrp":3.2,"vitamin_d":30}' | jq
```

Other targets: `make train`, `make retrain`, `make monitor`, `make load`,
`make test`, `make clean`.

## 9. Tests
```bash
make test     # containerised pytest: data generator, model quality gate, API contract
```
