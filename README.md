# TMDB top-rated TV shows — Streamlit dashboard

[![License: MIT](https://img.shields.io/github/license/akhilvydyula/top-rated-tv-shows-from-tmdb)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![GitHub stars](https://img.shields.io/github/stars/akhilvydyula/top-rated-tv-shows-from-tmdb?style=social)](https://github.com/akhilvydyula/top-rated-tv-shows-from-tmdb/stargazers)
[![Open Source](https://img.shields.io/badge/open%20source-welcome-brightgreen)](#open-source)

Interactive app to explore [Top rated TV shows from TMDB (Kaggle)](https://www.kaggle.com/datasets/rosemeenshaikh/op-rated-tv-shows-from-tmdb) and predict `vote_average` with a scikit-learn pipeline.

## Makefile commands

```bash
make help          # list targets
make install       # pip install -r requirements.txt
make install-dev   # install + editable dev extras
make train         # train model (MLflow on)
make train-quick   # train without MLflow (faster)
make serve         # Streamlit locally → http://localhost:8501
make test          # pytest
```

Override paths via `Makefile.local` (copy from `Makefile.local.example`).

**Windows:** use [Git Bash](https://git-scm.com/) or WSL so `make` is available.

## Local quick start

```bash
make install-dev
make train-quick    # skip if models/tmdb_rating_pipeline.joblib exists
make serve
```

## Deploy on Render

**Do not use Gunicorn** — this app is Streamlit, not Django/Flask.

### Manual Web Service (copy into the Render form)

| Field | Value |
|-------|--------|
| **Language** | Python 3 |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `bash scripts/render-start.sh` |

Or with Make (if available on the builder):

| Field | Value |
|-------|--------|
| **Build Command** | `make install` |
| **Start Command** | `make start` |

**Environment variables** (Add Environment Variable):

| Name | Value |
|------|--------|
| `PYTHON_VERSION` | `3.11.11` |
| `MODEL_PATH` | `models/tmdb_rating_pipeline.joblib` |

Leave **Root Directory** blank unless this repo lives in a monorepo subfolder.

1. Push this repo to GitHub (include `models/tmdb_rating_pipeline.joblib`).
2. [Render Dashboard](https://dashboard.render.com/) → **New** → **Web Service** → connect repo → paste the table above.
3. **Deploy web service**. Render sets `PORT` automatically; the start script binds Streamlit to it.

**Blueprint:** **New** → **Blueprint** and select `render.yaml` for the same settings.

The trained pipeline in `models/` is used at runtime (`MODEL_PATH` in `render.yaml`). To retrain before deploy locally:

```bash
make train-quick
git add models/
git commit -m "Update trained model"
```

Optional: add `make train-quick` to Render’s build command only if you accept longer builds and do not commit the `.joblib` file.

## App sections (executive dashboard)

| Tab | Purpose |
|-----|---------|
| **Executive brief** | KPIs, auto-generated strategic takeaways |
| **Market landscape** | Quadrant matrix, correlations, quality tiers |
| **Portfolio strategy** | Decade, theme, and segment analysis |
| **Competitive intel** | Stars, hidden gems, risk watchlist |
| **Deep dive** | Filterable catalog explorer |
| **Forecast** | ML greenlight sensitivity model |
| **Appendix** | Model metrics and retrain commands |

## Model

- **Trainer:** `make train` or `make train-quick` (same as `python -m ml.train`)
- **Artifacts:** `models/tmdb_rating_pipeline.joblib`, `models/metrics.json`

```bash
make train DATA=path/to/kaggle.csv
```

## Project layout

| Path | Purpose |
|------|---------|
| `Makefile` | install / train / serve / Render start |
| `render.yaml` | Render Blueprint |
| `streamlit_app.py` | Streamlit UI |
| `app/model_runtime.py` | Model load and inference |
| `ml/` | Training and features |
| `data/sample_tv_shows.csv` | Demo dataset |
| `models/` | Trained pipeline and metrics |

## Open source

This repository is **open source** under the [MIT License](LICENSE). Stars, issues, and pull requests are welcome — they help others discover the project and improve it for the community.

### How you can help

- **Star** the repo if you find it useful — it helps visibility on GitHub Explore and search.
- **Open an issue** for bugs, ideas, or questions.
- **Submit a pull request** with a focused change and a clear description.
- **Share** the project with anyone learning Streamlit dashboards or regression pipelines.

Maintained by [Akhil Vydyula](https://github.com/akhilvydyula) as part of the Skills Marathon ML portfolio.

## License

Application code is released under the [MIT License](LICENSE). Follow the [Kaggle dataset license](https://www.kaggle.com/datasets/rosemeenshaikh/op-rated-tv-shows-from-tmdb) for the underlying data.
