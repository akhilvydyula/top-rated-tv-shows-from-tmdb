# One-command workflows for local debugging.
# Copy Makefile.local.example -> Makefile.local to override PY, DATA, MLflow URI, ports, etc.
# Use GNU Make (Git Bash / WSL / Linux / macOS).

-include Makefile.local

PY ?= python
DATA ?= data/sample_tv_shows.csv
MODEL_OUT ?= models
HOST ?= 127.0.0.1
PORT ?= 8000
UVICORN_RELOAD ?=
MLFLOW_TRACKING_URI ?= file:$(subst \,/,$(CURDIR))/mlruns
MLFLOW_EXPERIMENT ?= tmdb-tv-rating
MLFLOW_UI_HOST ?= 127.0.0.1
MLFLOW_UI_PORT ?= 5000

export MLFLOW_TRACKING_URI
export MLFLOW_EXPERIMENT

.PHONY: help install install-dev wheel clean-build train serve test lint format mlflow-ui all-check

help:
	@echo "Common targets (set PY=.venv/Scripts/python or .venv/bin/python after venv):"
	@echo "  make install-dev   pip install -e '.[dev]'"
	@echo "  make wheel         build dist/*.whl"
	@echo "  make train         train + MLflow run (URI=$(MLFLOW_TRACKING_URI))"
	@echo "  make serve         API on http://$(HOST):$(PORT)  (UVICORN_RELOAD=--reload for dev)"
	@echo "  make mlflow-ui     MLflow UI (tracking URI above)"
	@echo "  make test / lint / format / all-check"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

wheel: install-dev
	$(PY) -m build --wheel
	@echo "Built wheel under dist/"

clean-build:
	rm -rf dist build *.egg-info

train: install
	$(PY) -m ml.train --data "$(DATA)" --out "$(MODEL_OUT)"

serve: install
	MODEL_PATH="$(MODEL_OUT)/tmdb_rating_pipeline.joblib" $(PY) -m uvicorn app.main:app --host $(HOST) --port $(PORT) $(UVICORN_RELOAD)

test: install-dev
	MLFLOW_DISABLE=1 $(PY) -m pytest tests -v --tb=short

lint: install-dev
	$(PY) -m ruff check app ml tests
	$(PY) -m ruff format --check app ml tests

format:
	$(PY) -m ruff format app ml tests
	$(PY) -m ruff check --fix app ml tests

mlflow-ui: install
	$(PY) -m mlflow ui --backend-store-uri "$(MLFLOW_TRACKING_URI)" --host $(MLFLOW_UI_HOST) --port $(MLFLOW_UI_PORT)

all-check: lint test wheel
