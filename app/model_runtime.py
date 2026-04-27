from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.features import build_feature_frame


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    path = Path(
        os.environ.get("MODEL_PATH", "models/tmdb_rating_pipeline.joblib")
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m ml.train --data path/to/tv_shows.csv"
        )
    return joblib.load(path)


def predict_vote_average(
    *,
    name: str,
    overview: str,
    popularity: float,
    vote_count: int,
    adult: bool,
    first_air_date: str | None,
) -> float:
    pipe = load_pipeline()
    date_str = first_air_date or "2000-01-01"
    row = pd.DataFrame(
        [
            {
                "first_air_date": date_str,
                "name": name,
                "overview": overview,
                "popularity": popularity,
                "vote_count": vote_count,
                "adult": adult,
            }
        ]
    )
    X = build_feature_frame(row)
    pred = float(pipe.predict(X)[0])
    return max(0.0, min(10.0, pred))


def clear_model_cache() -> None:
    load_pipeline.cache_clear()
