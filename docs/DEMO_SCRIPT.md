# Demo video script (5 min, screencast)

Flow: Prediction API → CI/CD → Monitoring (drift, metrics) → Retraining.

## Before recording
```bash
cd axiom-ml-platform
make up         # wait ~20s
make train      # produces models/ + an MLflow run
```
Tabs: API docs (`:8000/docs`), MLflow (`:5001`), Grafana (`:3001`), drift report (`:8090`).

---

## 0:00 — Intro & specs (30s)
> "AXIOM's biological-age model, industrialised: a prediction API, CI/CD,
> automated retraining and production monitoring. The model regresses biological
> age from a biomarker panel; we track MAE."

Show the repo (`src/ api/ retrain/ monitoring/ k8s/ .github/workflows/`).

## 0:30 — Model & tracking (60s)
```bash
make train
```
Open **MLflow** (`:5001`) → the `axiom-biological-age` experiment → show params,
MAE/RMSE/R², logged model.

## 1:30 — Prediction API (60s)
Open **`:8000/docs`** → `POST /predict`, run the example → returns `biological_age`.
Show `GET /model-info`.
```bash
make load       # generates traffic
```

## 2:30 — Monitoring (70s)
Open **Grafana** (`:3001`) → *AXIOM ML — Serving & Performance*: prediction rate,
p95 latency, deployed-model MAE, mean predicted age.
```bash
make monitor
```
Open **`:8090/drift_report.html`** → Evidently data-drift report; mention the
production-MAE-vs-training-MAE summary.

## 3:40 — Retraining (50s)
```bash
make retrain
```
> "A challenger is trained on accumulated, drifted data and promoted only if it
> beats the champion's MAE — the core of safe automated retraining."

## 4:30 — CI/CD (30s)
Show `.github/workflows/`: `ci.yml` (test + train + **quality gate**),
`cd.yml` (build & push image to GHCR, deploy to K8s), `retrain.yml` (weekly cron).
Show `k8s/` manifests.

> "Test-gated training, automated deployment, scheduled retraining, and live
> drift + performance monitoring — a full MLOps loop."

---

### Commands
| Action | Command |
|---|---|
| Everything | `make demo` |
| Train | `make train` |
| Traffic | `make load` |
| Drift report | `make monitor` |
| Retrain | `make retrain` |
| Tear down | `make clean` |
