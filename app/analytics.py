"""Executive analytics for TMDB top-rated TV catalog (market-sentiment proxy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Strategic quadrants (popularity vs quality)
SEGMENT_STARS = "Stars — scale winners"
SEGMENT_CULT = "Hidden gems — quality, low buzz"
SEGMENT_MASS = "Mass reach — buzz ahead of quality"
SEGMENT_RISK = "Underperformers — review slate"

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Crime & thriller": ("crime", "murder", "detective", "drug", "gang", "heist", "prison", "mafia"),
    "Sci-fi & fantasy": ("space", "alien", "dragon", "witch", "supernatural", "robot", "future", "galaxy"),
    "Drama & prestige": ("family", "political", "reign", "royal", "historical", "war", "tragedy"),
    "Comedy & animation": ("comedy", "sitcom", "animated", "cartoon", "parody", "funny"),
    "Documentary & reality": ("documentary", "earth", "nature", "reality", "competition"),
}


@dataclass(frozen=True)
class ExecutiveKPIs:
    catalog_size: int
    median_rating: float
    median_popularity: float
    total_audience_votes: int
    pct_premium: float  # rating >= 8.5
    pct_high_engagement: float  # vote_count >= p75
    avg_engagement_index: float
    stars_count: int
    hidden_gems_count: int
    risk_count: int
    rating_trend_recent: float | None  # recent decade vs prior
    top_decade: str
    dominant_theme: str


def _is_adult(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _infer_theme(overview: str, name: str) -> str:
    text = f"{name} {overview}".lower()
    scores = {theme: sum(1 for kw in kws if kw in text) for theme, kws in THEME_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General / mixed"


def prepare_analytics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich raw TMDB rows with executive metrics."""
    out = df.copy()
    out["first_air_date"] = pd.to_datetime(out["first_air_date"], errors="coerce")
    out["air_year"] = out["first_air_date"].dt.year
    out["decade"] = (out["air_year"] // 10 * 10).astype("Int64").astype(str) + "s"
    out["adult_flag"] = _is_adult(out["adult"]).astype(int)

    pop = pd.to_numeric(out["popularity"], errors="coerce").fillna(0)
    rating = pd.to_numeric(out["vote_average"], errors="coerce").fillna(0)
    votes = pd.to_numeric(out["vote_count"], errors="coerce").fillna(0).astype(int)

    out["popularity"] = pop
    out["vote_average"] = rating
    out["vote_count"] = votes
    out["log_votes"] = np.log1p(votes)
    out["engagement_index"] = votes / (pop + 1.0)  # audience conviction per buzz unit
    out["quality_pop_score"] = rating * np.log1p(pop)  # combined reach × quality
    out["overview_words"] = out["overview"].fillna("").str.split().str.len().clip(0, 500)
    out["theme"] = [
        _infer_theme(str(o), str(n))
        for o, n in zip(out["overview"], out["name"], strict=True)
    ]

    pop_med = pop.median()
    rating_med = rating.median()
    out["segment"] = SEGMENT_RISK
    out.loc[(pop >= pop_med) & (rating >= rating_med), "segment"] = SEGMENT_STARS
    out.loc[(pop < pop_med) & (rating >= rating_med), "segment"] = SEGMENT_CULT
    out.loc[(pop >= pop_med) & (rating < rating_med), "segment"] = SEGMENT_MASS

    out["rating_tier"] = pd.cut(
        rating,
        bins=[0, 7.0, 8.0, 8.5, 10.0],
        labels=["Below market", "Solid", "Strong", "Premium"],
        include_lowest=True,
    )
    return out.sort_values("quality_pop_score", ascending=False)


def compute_kpis(df: pd.DataFrame) -> ExecutiveKPIs:
    prepared = prepare_analytics_df(df) if "segment" not in df.columns else df
    votes = prepared["vote_count"]
    rating = prepared["vote_average"]
    recent = prepared[prepared["air_year"] >= 2015]["vote_average"].mean()
    prior = prepared[prepared["air_year"] < 2015]["vote_average"].mean()
    trend = float(recent - prior) if len(prepared[prepared["air_year"] >= 2015]) and len(
        prepared[prepared["air_year"] < 2015]
    ) else None

    decade_avg = prepared.groupby("decade", observed=True)["vote_average"].mean()
    top_decade = decade_avg.idxmax() if len(decade_avg) else "—"
    theme_avg = prepared.groupby("theme", observed=True)["vote_average"].mean()
    dominant = theme_avg.idxmax() if len(theme_avg) else "—"

    seg = prepared["segment"].value_counts()
    return ExecutiveKPIs(
        catalog_size=len(prepared),
        median_rating=float(rating.median()),
        median_popularity=float(prepared["popularity"].median()),
        total_audience_votes=int(votes.sum()),
        pct_premium=float((rating >= 8.5).mean() * 100),
        pct_high_engagement=float((votes >= votes.quantile(0.75)).mean() * 100),
        avg_engagement_index=float(prepared["engagement_index"].mean()),
        stars_count=int(seg.get(SEGMENT_STARS, 0)),
        hidden_gems_count=int(seg.get(SEGMENT_CULT, 0)),
        risk_count=int(seg.get(SEGMENT_RISK, 0)),
        rating_trend_recent=trend,
        top_decade=str(top_decade),
        dominant_theme=str(dominant),
    )


def generate_executive_insights(
    df: pd.DataFrame, kpis: ExecutiveKPIs
) -> list[tuple[str, str]]:
    """Narrative bullets for C-suite readout: (headline, body) pairs."""
    prepared = prepare_analytics_df(df) if "segment" not in df.columns else df
    insights: list[tuple[str, str]] = []

    insights.append(
        (
            "Catalog footprint",
            f"{kpis.catalog_size:,} top-rated titles indexed; median audience score "
            f"{kpis.median_rating:.2f}/10 with {kpis.pct_premium:.0f}% in the premium band (≥8.5).",
        )
    )

    if kpis.rating_trend_recent is not None:
        direction = "improving" if kpis.rating_trend_recent > 0.05 else (
            "softening" if kpis.rating_trend_recent < -0.05 else "stable"
        )
        insights.append(
            (
                "Quality trajectory",
                f"Post-2015 launches average {kpis.rating_trend_recent:+.2f} pts vs earlier "
                f"cohorts — sentiment is {direction}.",
            )
        )

    insights.append(
        (
            "Portfolio mix",
            f"{kpis.stars_count} scale winners (high buzz + high quality), "
            f"{kpis.hidden_gems_count} hidden gems (acquisition targets), "
            f"{kpis.risk_count} underperformers for slate review.",
        )
    )

    top_star = prepared[prepared["segment"] == SEGMENT_STARS].head(3)["name"].tolist()
    if top_star:
        insights.append(
            (
                "Benchmark titles",
                f"{', '.join(top_star)} lead the star quadrant — "
                "use as comps for greenlight ROI modeling.",
            )
        )

    gems = prepared[prepared["segment"] == SEGMENT_CULT].nlargest(3, "vote_average")
    if len(gems):
        insights.append(
            (
                "Undervalued IP",
                f"{', '.join(gems['name'].tolist())} — strong scores, below-median buzz; "
                "suited for catalog fill or localized marketing pushes.",
            )
        )

    mass = prepared[prepared["segment"] == SEGMENT_MASS].nlargest(2, "popularity")
    if len(mass):
        insights.append(
            (
                "Reach vs quality gap",
                f"{', '.join(mass['name'].tolist())} — high awareness; "
                "monitor retention and sequel economics.",
            )
        )

    insights.append(
        (
            "Content concentration",
            f"{kpis.dominant_theme} shows the highest mean rating; "
            f"{kpis.top_decade} cohort leads by decade.",
        )
    )

    votes_label = (
        f"{kpis.total_audience_votes / 1_000_000:.1f}M"
        if kpis.total_audience_votes >= 1_000_000
        else f"{kpis.total_audience_votes:,}"
    )
    insights.append(
        (
            "Audience signal volume",
            f"{votes_label} cumulative TMDB votes — proxy for global engagement depth "
            "(not proprietary platform viewership).",
        )
    )

    return insights


def decade_summary(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_analytics_df(df) if "decade" not in df.columns else df
    g = prepared.groupby("decade", observed=True)
    return (
        g.agg(
            titles=("name", "count"),
            avg_rating=("vote_average", "mean"),
            avg_popularity=("popularity", "mean"),
            total_votes=("vote_count", "sum"),
            premium_share=("vote_average", lambda s: (s >= 8.5).mean()),
        )
        .round(2)
        .reset_index()
        .sort_values("decade")
    )


def theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_analytics_df(df) if "theme" not in df.columns else df
    g = prepared.groupby("theme", observed=True)
    return (
        g.agg(
            titles=("name", "count"),
            avg_rating=("vote_average", "mean"),
            avg_popularity=("popularity", "mean"),
            engagement_index=("engagement_index", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_rating", ascending=False)
    )


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_analytics_df(df) if "segment" not in df.columns else df
    return (
        prepared.groupby("segment", observed=True)
        .agg(
            titles=("name", "count"),
            avg_rating=("vote_average", "mean"),
            avg_popularity=("popularity", "mean"),
            avg_votes=("vote_count", "mean"),
        )
        .round(2)
        .reset_index()
    )


def top_opportunities(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Hidden gems ranked by quality × engagement efficiency."""
    prepared = prepare_analytics_df(df) if "segment" not in df.columns else df
    gems = prepared[prepared["segment"] == SEGMENT_CULT].copy()
    gems["opportunity_score"] = gems["vote_average"] * gems["engagement_index"]
    cols = [
        "name",
        "vote_average",
        "popularity",
        "vote_count",
        "engagement_index",
        "air_year",
        "theme",
    ]
    return gems.nlargest(n, "opportunity_score")[cols]


def risk_watchlist(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    prepared = prepare_analytics_df(df) if "segment" not in df.columns else df
    risk = prepared[prepared["segment"] == SEGMENT_RISK].copy()
    risk["risk_score"] = risk["popularity"] / (risk["vote_average"] + 0.1)
    cols = ["name", "vote_average", "popularity", "vote_count", "air_year", "theme"]
    return risk.nlargest(n, "risk_score")[cols]


def correlation_insights(df: pd.DataFrame) -> dict[str, float]:
    prepared = prepare_analytics_df(df) if "engagement_index" not in df.columns else df
    return {
        "rating_vs_popularity": float(prepared["vote_average"].corr(prepared["popularity"])),
        "rating_vs_votes": float(prepared["vote_average"].corr(prepared["log_votes"])),
        "popularity_vs_votes": float(prepared["popularity"].corr(prepared["log_votes"])),
    }


def filter_dataframe(
    df: pd.DataFrame,
    *,
    rating_range: tuple[float, float],
    decades: list[str],
    segments: list[str],
    themes: list[str],
    search: str,
) -> pd.DataFrame:
    prepared = prepare_analytics_df(df) if "segment" not in df.columns else df
    mask = (
        (prepared["vote_average"] >= rating_range[0])
        & (prepared["vote_average"] <= rating_range[1])
    )
    if decades:
        mask &= prepared["decade"].isin(decades)
    if segments:
        mask &= prepared["segment"].isin(segments)
    if themes:
        mask &= prepared["theme"].isin(themes)
    if search:
        mask &= prepared["name"].str.lower().str.contains(search.lower(), na=False)
    return prepared[mask]
