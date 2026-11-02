# Local services and maintenance for the VN2 project. `make help` lists the targets.
# Always uses the project venv's mlflow: the global one migrates mlruns.db to a schema the venv rejects.

SHELL := /bin/bash
ROOT := $(abspath .)
VENV := $(ROOT)/.venv/bin
MLFLOW_PORT ?= 5050
MLFLOW_URI := http://127.0.0.1:$(MLFLOW_PORT)
LOGS := $(ROOT)/artifacts/logs

.PHONY: help mlflow-server mlflow-stop app app-stop backup

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

mlflow-server: ## start the tracking server (SQLite backend, proxied artifacts) on port $(MLFLOW_PORT)
	@mkdir -p $(LOGS) mlartifacts
	@if curl -sf $(MLFLOW_URI)/health >/dev/null; then echo "MLflow server already running at $(MLFLOW_URI)"; else \
	  nohup $(VENV)/mlflow server --backend-store-uri sqlite:///$(ROOT)/mlruns.db \
	    --artifacts-destination $(ROOT)/mlartifacts --serve-artifacts \
	    --host 127.0.0.1 --port $(MLFLOW_PORT) --workers 1 > $(LOGS)/mlflow-server.log 2>&1 & \
	  for i in $$(seq 1 60); do curl -sf $(MLFLOW_URI)/health >/dev/null && break; sleep 1; done; \
	  if curl -sf $(MLFLOW_URI)/health >/dev/null; then echo "MLflow server at $(MLFLOW_URI)"; \
	  else echo "MLflow server did not start; see $(LOGS)/mlflow-server.log"; exit 1; fi; fi

mlflow-stop: ## stop the tracking server
	@pkill -TERM -f "mlflow server --backend-store-uri sqlite:///$(ROOT)/mlruns.db" && echo "stopped" || echo "not running"

app: mlflow-server ## start the Streamlit app against the tracking server
	@MLFLOW_TRACKING_URI=$(MLFLOW_URI) nohup $(VENV)/streamlit run app/Home.py --server.headless true \
	  > $(LOGS)/streamlit.log 2>&1 &
	@echo "Streamlit at http://localhost:8501 (log: $(LOGS)/streamlit.log)"

app-stop: ## stop the Streamlit app
	@pkill -TERM -f "streamlit run app/Home.py" && echo "stopped" || echo "not running"

backup: ## consistent online backup of mlruns.db into backups/
	@mkdir -p backups && sqlite3 mlruns.db ".backup backups/mlruns-$$(date +%Y%m%d-%H%M).db" && ls -t backups/*.db | head -1
