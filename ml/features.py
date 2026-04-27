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
    """Turn raw TMDB rows into numeric model inputs (no target)."""
    out = pd.DataFrame(
        {
            "popularity": pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0),
            "log_vote_count": np.log1p(
                pd.to_numeric(df["vote_count"], errors="coerce").fillna(0.0)
            ),
            "adult": df["adult"].astype(str).str.lower().isin(("true", "1", "yes")).astype(int),
            "air_year": _parse_year(df["first_air_date"]),
            "overview_len": df["overview"].fillna("").astype(str).str.len().clip(0, 5000),
            "name_len": df["name"].fillna("").astype(str).str.len().clip(0, 500),
        }
    )
    return out
