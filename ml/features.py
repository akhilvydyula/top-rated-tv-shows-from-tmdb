"""Feature builders shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_year(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    year = dt.dt.year.astype("float64")
    med = float(year.median()) if year.notna().any() else 2000.0
    return year.fillna(med)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw TMDB rows into model inputs (numeric + text blob for TF-IDF)."""
    pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0.0)
    overview = df["overview"].fillna("").astype(str)
    name = df["name"].fillna("").astype(str)
    blob = (name + " " + overview).str.slice(0, 8000)
    out = pd.DataFrame(
        {
            "popularity": pop,
            "log_pop": np.log1p(pop.clip(lower=0)),
            "log_vote_count": np.log1p(vc.clip(lower=0)),
            "votes_per_log_pop": vc / (np.log1p(pop) + 0.5),
            "adult": df["adult"].astype(str).str.lower().isin(("true", "1", "yes")).astype(int),
            "air_year": _parse_year(df["first_air_date"]),
            "overview_len": overview.str.len().clip(0, 5000),
            "name_len": name.str.len().clip(0, 500),
            "overview_words": overview.str.split().str.len().fillna(0).astype(int).clip(0, 2000),
            "text_blob": blob,
        }
    )
    out["votes_per_log_pop"] = out["votes_per_log_pop"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
