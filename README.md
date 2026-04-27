# TMDB top-rated TV shows — rating model API

MVP stack: **scikit-learn** regression (predicts TMDB `vote_average` on a 0–10 scale), **FastAPI** inference service, **Docker**, and **CI/CD** (GitLab CI + GitHub Actions).

Dataset reference: [Top rated TV shows from TMDB (Kaggle)](https://www.kaggle.com/datasets/rosemeenshaikh/op-rated-tv-shows-from-tmdb).

## Model

- **Target:** `vote_average`
- **Features:** `popularity`, `log1p(vote_count)`, `adult`, air year from `first_air_date`, character lengths of `name` / `overview`
- **Algorithm:** `HistGradientBoostingRegressor` inside a `Pipeline` with scaling and imputation (see `ml/train.py`)

After training, metrics are written to `models/metrics.json` alongside `models/tmdb_rating_pipeline.joblib`. With MLflow enabled (default), runs, params, metrics, and a `sklearn-model` artifact are logged under `mlruns/` (or your `MLFLOW_TRACKING_URI`).

## Packaging (wheel)

`pyproject.toml` defines the installable package **`tmdb-tv-rating-mvp`** (`app` + `ml`).

```bash
pip install -e ".[dev]"    # editable dev install
python -m build --wheel    # produces dist/*.whl
pip install dist/tmdb_tv_rating_mvp-*.whl
```

Console script: `tmdb-train` (same as `python -m ml.train`).

## Makefile (one-shot commands)

Copy `Makefile.local.example` to `Makefile.local` to set e.g. `PY`, `DATA`, `MLFLOW_TRACKING_URI`, `HOST`, `PORT`, `UVICORN_RELOAD=--reload`.

```bash
make install-dev   # pip install -e ".[dev]"
make train         # train + MLflow (see MLFLOW_* env vars)
make mlflow-ui     # browse runs (default file store: ./mlruns)
make serve         # API (set UVICORN_RELOAD=--reload for hot reload)
make test / lint / format / wheel / all-check
```

On Windows, use **Git Bash** or **WSL** so GNU Make is available.

## MLflow

- **Local file store (default):** `MLFLOW_TRACKING_URI` defaults to `file:<cwd>/mlruns` if unset.
- **Remote server:** set `MLFLOW_TRACKING_URI=http://your-host:5000` and optional `MLFLOW_MODEL_NAME` to register the sklearn model (not used with `file:` URIs).
- **Disable:** `MLFLOW_DISABLE=1` or `python -m ml.train --no-mlflow`.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -e ".[dev]"
# or: pip install -r requirements.txt -r requirements-dev.txt

python -m ml.train --data data/sample_tv_shows.csv --out models
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/predict` with JSON body matching `ShowInput` in `app/schemas.py`

Use your Kaggle CSV path instead of `data/sample_tv_shows.csv` once downloaded (same columns as in the dataset description).

## Docker

```bash
docker compose up --build
```

The image trains on `data/sample_tv_shows.csv` at build time. For production, copy your full Kaggle export into the build context and adjust the `RUN python -m ml.train ...` line in the `Dockerfile`.

## CI/CD

- **GitLab:** `.gitlab-ci.yml` — `lint`, `test` (pytest with MLflow off), **`wheel`** (uploads `dist/*.whl`), **`mlflow_train`** (trains with MLflow file store, uploads `mlruns/` + `models/*` so you can compare runs across pushes). **Deploy** remains a manual Docker job on `main` when registry variables and a Docker runner are configured.
- **GitHub:** `.github/workflows/ci.yml` — lint + test, then **`wheel`** and **`mlflow-train`** jobs that upload the wheel and MLflow/model artifacts.

## Project layout

| Path | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, deps, wheel build |
| `Makefile` | Shortcut commands for train / serve / MLflow UI / wheel |
| `ml/train.py` | Train and save pipeline + metrics |
| `ml/features.py` | Shared feature engineering |
| `app/main.py` | FastAPI app |
| `data/sample_tv_shows.csv` | Small demo dataset for tests and Docker |
| `tests/` | Pytest suite |

## License

Use follows the Kaggle dataset license; model and code are provided as-is for learning and prototyping.
