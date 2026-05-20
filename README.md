# TMDB top-rated TV shows — Streamlit dashboard

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

1. Push this repo to GitHub/GitLab.
2. [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint** (uses `render.yaml`)  
   **or** **Web Service** → connect repo and set:
   - **Build command:** `make install`
   - **Start command:** `make start`
   - **Python version:** `3.11.11` (or match `runtime.txt`)
3. Deploy. Render sets `PORT`; `make start` binds Streamlit to it.

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

## License

Follow the Kaggle dataset license; code is provided as-is for learning and prototyping.
