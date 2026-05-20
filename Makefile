# TMDB TV Shows — Streamlit app (local + Render)
# Copy Makefile.local.example to Makefile.local to override PY, DATA, PORT, etc.

-include Makefile.local

PY        ?= python
PIP       ?= pip
DATA      ?= data/sample_tv_shows.csv
MODEL_DIR ?= models
PORT      ?= 8501
HOST      ?= 0.0.0.0

STREAMLIT := $(PY) -m streamlit run streamlit_app.py \
	--server.address=$(HOST) \
	--server.headless=true \
	--browser.gatherUsageStats=false

.PHONY: help install install-dev train train-quick serve start test lint clean

help:
	@echo "Targets:"
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Editable install with dev tools (pytest, ruff)"
	@echo "  make train         Train model (MLflow on)"
	@echo "  make train-quick   Train model (--no-mlflow, faster)"
	@echo "  make serve         Run Streamlit locally (PORT=$(PORT))"
	@echo "  make start         Run Streamlit for Render (uses \$$PORT)"
	@echo "  make test          Run pytest"
	@echo "  make lint          Run ruff"
	@echo "  make clean         Remove caches"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"

train:
	$(PY) -m ml.train --data $(DATA) --out $(MODEL_DIR)

train-quick:
	$(PY) -m ml.train --data $(DATA) --out $(MODEL_DIR) --no-mlflow

serve:
	$(STREAMLIT) --server.port=$(PORT)

# Render sets PORT at runtime; fallback to 8501 for local smoke tests
start:
	$(STREAMLIT) --server.port=$${PORT:-$(PORT)}

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check app ml tests streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ app/__pycache__ ml/__pycache__ tests/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
