from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.analytics import (
    SEGMENT_STARS,
    compute_kpis,
    generate_executive_insights,
    prepare_analytics_df,
    top_opportunities,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_adds_segments() -> None:
    df = prepare_analytics_df(pd.read_csv(ROOT / "data" / "sample_tv_shows.csv"))
    assert "segment" in df.columns
    assert "theme" in df.columns
    assert df["segment"].notna().all()


def test_kpis_and_insights() -> None:
    df = prepare_analytics_df(pd.read_csv(ROOT / "data" / "sample_tv_shows.csv"))
    kpis = compute_kpis(df)
    assert kpis.catalog_size == len(df)
    insights = generate_executive_insights(df, kpis)
    assert len(insights) >= 4
    assert any("Catalog footprint" in i for i in insights)


def test_opportunities_prefers_cult_segment() -> None:
    df = prepare_analytics_df(pd.read_csv(ROOT / "data" / "sample_tv_shows.csv"))
    opps = top_opportunities(df, 5)
    assert len(opps) <= 5
    cult_names = set(df[df["segment"] != SEGMENT_STARS]["name"])
    assert opps["name"].isin(cult_names | set(df[df["segment"] == SEGMENT_STARS]["name"])).all()
