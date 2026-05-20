# TMDB top-rated TV shows — Streamlit dashboard

Interactive app to explore [Top rated TV shows from TMDB (Kaggle)](https://www.kaggle.com/datasets/rosemeenshaikh/op-rated-tv-shows-from-tmdb) and predict `vote_average` with a scikit-learn pipeline.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
python -m ml.train --data data/sample_tv_shows.csv --out models
streamlit run streamlit_app.py
```

Upload your full Kaggle CSV in the sidebar, or pass a path when training:

```bash
python -m ml.train --data path/to/kaggle.csv --out models --no-mlflow
```

## App sections

| Tab | Purpose |
|-----|---------|
| **Overview** | Dataset stats and top shows table |
| **Explore** | Filters, rating distribution, popularity vs rating |
| **Predict** | Manual or dataset-driven rating prediction |
| **Model** | Holdout metrics and training metadata |

## Model

- **Target:** `vote_average` as rounded 0–10 stars (classification) or continuous regression (`--task regression`)
- **Features:** popularity, vote count, adult flag, air year, title/overview text (TF-IDF + SVD)
- **Trainer:** `python -m ml.train` (`ml/train.py`)

```bash
python -m ml.train --data data/sample_tv_shows.csv --out models
python -m ml.train --data path/to/kaggle.csv --out models --task classification --tune
python -m ml.train --data path/to/kaggle.csv --out models --no-mlflow
```

Metrics are saved to `models/metrics.json`; the pipeline to `models/tmdb_rating_pipeline.joblib`.

## Project layout

| Path | Purpose |
|------|---------|
| `streamlit_app.py` | Streamlit UI |
| `app/model_runtime.py` | Model load and inference |
| `ml/train.py`, `ml/features.py` | Training and features |
| `data/sample_tv_shows.csv` | Demo dataset |
| `models/` | Trained pipeline and metrics |
| `tests/` | Training tests |

## Dev

```bash
pip install -e ".[dev]"
pytest
```

## License

Follow the Kaggle dataset license; code is provided as-is for learning and prototyping.
