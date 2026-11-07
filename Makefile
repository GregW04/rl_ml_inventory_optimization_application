# Local services and maintenance for the VN2 project. `make help` lists the targets.
# Always uses the project venv's mlflow: the global one migrates mlruns.db to a schema the venv rejects.

SHELL := /bin/bash
ROOT := $(abspath .)
VENV := $(ROOT)/.venv/bin
MLFLOW_PORT ?= 5050
MLFLOW_URI := http://127.0.0.1:$(MLFLOW_PORT)
APP_PORT ?= 8501
LOGS := $(ROOT)/artifacts/logs
# Airflow lives in its own venv and home; its tasks run the project's CLI with $(VENV)/python.
AIRFLOW_VERSION := 3.3.2
AIRFLOW_VENV := $(ROOT)/.venv-airflow/bin
AIRFLOW_PORT ?= 8080
AIRFLOW_CONSTRAINTS := https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW_VERSION)/constraints-3.12.txt
# Standalone starts its components as `airflow` from PATH. One machine with 8 GB: localhost only,
# examples off, at most 2 tasks at a time, 1 API worker.
AIRFLOW_ENV := PATH=$(AIRFLOW_VENV):$$PATH AIRFLOW_HOME=$(ROOT)/airflow_home AIRFLOW__CORE__DAGS_FOLDER=$(ROOT)/airflow/dags \
  AIRFLOW__CORE__LOAD_EXAMPLES=False AIRFLOW__CORE__PARALLELISM=2 \
  AIRFLOW__API__HOST=127.0.0.1 AIRFLOW__API__PORT=$(AIRFLOW_PORT) AIRFLOW__API__WORKERS=1 \
  AIRFLOW__API__BASE_URL=http://127.0.0.1:$(AIRFLOW_PORT) \
  MLFLOW_TRACKING_URI=$(MLFLOW_URI)

.PHONY: help mlflow-server mlflow-stop app app-stop backup airflow-install airflow-init airflow airflow-stop

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

# MLflow's job runner (7 huey workers, ~1.2 GB) only serves GenAI features (trace evaluation,
# scorers), which this project doesn't use: it stays off.
mlflow-server: ## start the tracking server (SQLite backend, proxied artifacts) on port $(MLFLOW_PORT)
	@mkdir -p $(LOGS) mlartifacts
	@if curl -sf $(MLFLOW_URI)/health >/dev/null; then echo "MLflow server already running at $(MLFLOW_URI)"; else \
	  MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false nohup $(VENV)/mlflow server --backend-store-uri sqlite:///$(ROOT)/mlruns.db \
	    --artifacts-destination $(ROOT)/mlartifacts --serve-artifacts \
	    --host 127.0.0.1 --port $(MLFLOW_PORT) --workers 1 > $(LOGS)/mlflow-server.log 2>&1 & \
	  for i in $$(seq 1 60); do curl -sf $(MLFLOW_URI)/health >/dev/null && break; sleep 1; done; \
	  if curl -sf $(MLFLOW_URI)/health >/dev/null; then echo "MLflow server at $(MLFLOW_URI)"; \
	  else echo "MLflow server did not start; see $(LOGS)/mlflow-server.log"; exit 1; fi; fi

mlflow-stop: ## stop the tracking server
	@pkill -TERM -f "mlflow server --backend-store-uri sqlite:///$(ROOT)/mlruns.db" && echo "stopped" || echo "not running"

# A running app keeps the src/ and app/lib.py it imported at start, so `make app` replaces it
# instead of starting a second copy on the next port. Restart it after changing the code.
app: mlflow-server ## (re)start the Streamlit app against the tracking server on port $(APP_PORT)
	@pkill -TERM -f "streamlit run app/Home.py" && echo "stopped the running app" && sleep 2 || true
	@MLFLOW_TRACKING_URI=$(MLFLOW_URI) nohup $(VENV)/streamlit run app/Home.py --server.headless true \
	  --server.port $(APP_PORT) > $(LOGS)/streamlit.log 2>&1 &
	@echo "Streamlit at http://localhost:$(APP_PORT) (log: $(LOGS)/streamlit.log)"

app-stop: ## stop the Streamlit app
	@pkill -TERM -f "streamlit run app/Home.py" && echo "stopped" || echo "not running"

backup: ## consistent online backup of mlruns.db into backups/
	@mkdir -p backups && sqlite3 mlruns.db ".backup backups/mlruns-$$(date +%Y%m%d-%H%M).db" && ls -t backups/*.db | head -1

airflow-install: ## create .venv-airflow: Airflow $(AIRFLOW_VERSION) with the official constraints
	uv venv --python 3.12 $(ROOT)/.venv-airflow
	uv pip install --python $(AIRFLOW_VENV)/python -r airflow/requirements.txt -c $(AIRFLOW_CONSTRAINTS)

airflow-init: ## create Airflow's metadata DB and the 1-slot pool for refits and retrains
	@$(AIRFLOW_ENV) $(AIRFLOW_VENV)/airflow db migrate > /dev/null
	@$(AIRFLOW_ENV) $(AIRFLOW_VENV)/airflow pools set heavy 1 "refit and retrain: one at a time (8 GB RAM)"

airflow: mlflow-server ## start Airflow standalone on port $(AIRFLOW_PORT) (DAGs: airflow/dags)
	@mkdir -p $(LOGS)
	@if curl -sf http://127.0.0.1:$(AIRFLOW_PORT)/api/v2/monitor/health >/dev/null; then echo "Airflow already running"; else \
	  $(AIRFLOW_ENV) nohup $(AIRFLOW_VENV)/airflow standalone > $(LOGS)/airflow.log 2>&1 & \
	  for i in $$(seq 1 120); do curl -sf http://127.0.0.1:$(AIRFLOW_PORT)/api/v2/monitor/health >/dev/null && break; sleep 1; done; \
	  if curl -sf http://127.0.0.1:$(AIRFLOW_PORT)/api/v2/monitor/health >/dev/null; then \
	    echo "Airflow at http://127.0.0.1:$(AIRFLOW_PORT) (user admin, password in airflow_home/simple_auth_manager_passwords.json.generated)"; \
	  else echo "Airflow did not start; see $(LOGS)/airflow.log"; exit 1; fi; fi

airflow-stop: ## stop Airflow standalone and its components
	@pkill -TERM -f "$(AIRFLOW_VENV)/airflow" && echo "stopped" || echo "not running"

