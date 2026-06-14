# AXIOM ML Platform — convenience targets
# Requires Docker Desktop. Ports are offset to coexist with the Block 3 stack.

COMPOSE = docker compose

.PHONY: help build up down clean ps logs train retrain monitor load test demo

help:
	@echo "Targets:"
	@echo "  make up       - build & start API, MLflow, Prometheus, Grafana, report server"
	@echo "  make train    - train the model (writes models/ + logs to MLflow)"
	@echo "  make load     - send sample prediction traffic (for monitoring)"
	@echo "  make monitor  - generate the Evidently drift report"
	@echo "  make retrain  - champion/challenger retraining"
	@echo "  make test     - run unit tests (containerised)"
	@echo "  make demo     - up + train + load + monitor (one-shot)"
	@echo "  make down     - stop    |    make clean - stop & wipe volumes"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --build
	@echo "API        http://localhost:8000  (docs: /docs)"
	@echo "MLflow     http://localhost:5001"
	@echo "Grafana    http://localhost:3001   (admin/admin)"
	@echo "Prometheus http://localhost:9091"
	@echo "Drift report http://localhost:8090/drift_report.html  (after 'make monitor')"

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

train:
	$(COMPOSE) run --rm api python src/train.py

retrain:
	$(COMPOSE) run --rm api python retrain/retrain.py

monitor:
	$(COMPOSE) run --rm api python monitoring/drift_report.py

load:
	@echo "Sending 60 sample predictions to the API..."
	@for i in $$(seq 1 60); do \
		curl -s -o /dev/null -X POST http://localhost:8000/predict \
		  -H 'content-type: application/json' \
		  -d '{"vitamin_d":30,"ferritin":40,"fasting_glucose":105,"hba1c":5.9,"tsh":2.2,"hscrp":3.4,"hdl":46,"ldl":135,"triglycerides":165,"vitamin_b12":360}'; \
	done; echo "done."

test:
	docker run --rm -v "$(PWD)":/w -w /w python:3.12-slim \
		bash -c "pip install -q -r requirements.txt && pytest -q tests"

demo: up
	@echo "Waiting for services..."; sleep 15
	$(MAKE) train
	$(MAKE) load
	$(MAKE) monitor
	@echo ""
	@echo "Demo ready:"
	@echo "  API docs       http://localhost:8000/docs"
	@echo "  Grafana        http://localhost:3001"
	@echo "  MLflow         http://localhost:5001"
	@echo "  Drift report   http://localhost:8090/drift_report.html"
