from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from ml.features import build_feature_frame


def _model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "models/tmdb_rating_pipeline.joblib")).resolve()


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    path = _model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m ml.train --data path/to/tv_shows.csv"
        )
    return joblib.load(path)


@lru_cache(maxsize=4)
def _training_meta(model_dir: str) -> dict[str, Any] | None:
    meta_path = Path(model_dir) / "training_meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_training_meta() -> dict[str, Any] | None:
    return _training_meta(str(_model_path().parent))


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
    if is_classifier(pipe):
        proba = pipe.predict_proba(X)[0]
        cls_arr = pipe.classes_
        meta = load_training_meta() or {}
        cmap = meta.get("class_to_mean_vote")
        if isinstance(cmap, dict) and cmap:
            weights = np.array(
                [float(cmap.get(str(int(c)), float(c))) for c in cls_arr],
                dtype=float,
            )
        else:
            weights = cls_arr.astype(float)
        pred = float(np.dot(proba, weights))
    else:
        pred = float(pipe.predict(X)[0])
    return max(0.0, min(10.0, pred))


def clear_model_cache() -> None:
    load_pipeline.cache_clear()
    _training_meta.cache_clear()
