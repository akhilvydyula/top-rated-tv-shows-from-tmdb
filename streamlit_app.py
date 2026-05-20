"""Executive Streamlit dashboard — TMDB top TV catalog intelligence."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.analytics import (
    SEGMENT_MASS,
    SEGMENT_STARS,
    compute_kpis,
    correlation_insights,
    decade_summary,
    filter_dataframe,
    generate_executive_insights,
    prepare_analytics_df,
    risk_watchlist,
    segment_summary,
    theme_summary,
    top_opportunities,
)
from app.model_runtime import load_pipeline, load_training_meta, predict_vote_average

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sample_tv_shows.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"

EXEC_CSS = """
<style>
    .exec-header { font-size: 0.95rem; color: #6b7280; margin-bottom: 0.25rem; }
    .exec-insight {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #e50914;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #f3f4f6;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .disclaimer { font-size: 0.8rem; color: #9ca3af; }
    div[data-testid="stMetricValue"] { font-size: 1.75rem; }
</style>
"""


@st.cache_data
def load_shows(csv_path: str) -> pd.DataFrame:
    return prepare_analytics_df(pd.read_csv(csv_path))


@st.cache_data
def load_metrics() -> dict | None:
    if METRICS_PATH.is_file():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def model_ready() -> bool:
    try:
        load_pipeline()
        return True
    except FileNotFoundError:
        return False


def _quadrant_scatter(df: pd.DataFrame) -> go.Figure:
    pop_med = df["popularity"].median()
    rating_med = df["vote_average"].median()
    fig = px.scatter(
        df,
        x="popularity",
        y="vote_average",
        color="segment",
        size="vote_count",
        hover_name="name",
        hover_data=["air_year", "theme", "vote_count"],
        labels={"popularity": "Market buzz (TMDB popularity)", "vote_average": "Audience score"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_hline(y=rating_med, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    fig.add_vline(x=pop_med, line_dash="dash", line_color="#94a3b8", opacity=0.7)
    fig.update_layout(
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.add_annotation(
        x=pop_med * 1.5,
        y=9.2,
        text="Stars",
        showarrow=False,
        font=dict(size=11, color="#64748b"),
    )
    fig.add_annotation(
        x=pop_med * 0.3,
        y=9.2,
        text="Hidden gems",
        showarrow=False,
        font=dict(size=11, color="#64748b"),
    )
    return fig


def page_executive_brief(df: pd.DataFrame, kpis) -> None:
    st.markdown('<p class="exec-header">CONFIDENTIAL · MARKET INTELLIGENCE BRIEF</p>', unsafe_allow_html=True)
    st.subheader("Executive summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Catalog titles", f"{kpis.catalog_size:,}")
    c2.metric("Median score", f"{kpis.median_rating:.2f}")
    c3.metric("Premium share", f"{kpis.pct_premium:.0f}%", help="Titles rated ≥ 8.5")
    c4.metric("Scale winners", kpis.stars_count)
    c5.metric("Hidden gems", kpis.hidden_gems_count)

    st.markdown("#### Strategic takeaways")
    for bullet in generate_executive_insights(df, kpis):
        st.markdown(
            f'<div class="exec-insight">{bullet}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="disclaimer">TMDB public ratings are a market-sentiment proxy — not Netflix '
        "proprietary viewership, churn, or revenue. Use for competitive benchmarking and "
        "greenlight sensitivity analysis.</p>",
        unsafe_allow_html=True,
    )


def page_market_landscape(df: pd.DataFrame) -> None:
    st.subheader("Market landscape")
    corr = correlation_insights(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rating ↔ buzz", f"{corr['rating_vs_popularity']:.2f}")
    c2.metric("Rating ↔ votes", f"{corr['rating_vs_votes']:.2f}")
    c3.metric("Buzz ↔ votes", f"{corr['popularity_vs_votes']:.2f}")

    st.plotly_chart(_quadrant_scatter(df), use_container_width=True)
    st.caption(
        "Bubble size = vote volume. Dashed lines = catalog medians. "
        "**Stars** = prioritize franchise extensions; **Hidden gems** = catalog/acquisition plays."
    )

    left, right = st.columns(2)
    with left:
        tier = df["rating_tier"].value_counts().sort_index()
        fig = px.bar(
            x=tier.index.astype(str),
            y=tier.values,
            labels={"x": "Quality tier", "y": "Titles"},
            color=tier.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, height=320, coloraxis_showscale=False)
        st.markdown("**Quality distribution**")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(
            df,
            x="vote_average",
            nbins=20,
            labels={"vote_average": "Audience score"},
            color_discrete_sequence=["#e50914"],
        )
        fig.update_layout(height=320, showlegend=False)
        st.markdown("**Score histogram**")
        st.plotly_chart(fig, use_container_width=True)


def page_portfolio_strategy(df: pd.DataFrame) -> None:
    st.subheader("Portfolio & era strategy")

    dec = decade_summary(df)
    theme = theme_summary(df)
    seg = segment_summary(df)

    t1, t2, t3 = st.tabs(["By decade", "By content theme", "Strategic segments"])

    with t1:
        fig = px.bar(
            dec,
            x="decade",
            y="avg_rating",
            text="titles",
            labels={"avg_rating": "Avg score", "decade": "Launch decade"},
            color="avg_rating",
            color_continuous_scale="RdYlGn",
        )
        fig.update_traces(texttemplate="%{text} titles", textposition="outside")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dec, use_container_width=True, hide_index=True)

    with t2:
        fig = px.treemap(
            theme,
            path=["theme"],
            values="titles",
            color="avg_rating",
            color_continuous_scale="RdYlGn",
            hover_data=["avg_popularity", "engagement_index"],
        )
        fig.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(theme, use_container_width=True, hide_index=True)

    with t3:
        fig = px.bar(
            seg,
            x="segment",
            y="titles",
            color="avg_rating",
            labels={"titles": "Count", "avg_rating": "Avg score"},
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=360, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg, use_container_width=True, hide_index=True)


def page_competitive_intel(df: pd.DataFrame) -> None:
    st.subheader("Competitive intelligence")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Top scale winners")
        stars = df[df["segment"] == SEGMENT_STARS].head(12)[
            ["name", "vote_average", "popularity", "vote_count", "air_year", "theme"]
        ]
        st.dataframe(stars, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Acquisition / catalog opportunities")
        st.dataframe(top_opportunities(df, 12), use_container_width=True, hide_index=True)

    st.markdown("#### Slate risk watch (high buzz, weaker scores)")
    st.dataframe(risk_watchlist(df, 10), use_container_width=True, hide_index=True)

    mass = df[df["segment"] == SEGMENT_MASS].nlargest(8, "popularity")
    if len(mass):
        st.markdown("#### Mass reach — quality gap titles")
        st.dataframe(
            mass[["name", "vote_average", "popularity", "vote_count", "theme"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Engagement efficiency leaders")
    leaders = df.nlargest(10, "engagement_index")[
        ["name", "engagement_index", "vote_average", "popularity", "vote_count"]
    ]
    st.dataframe(leaders, use_container_width=True, hide_index=True)
    st.caption("Engagement index = votes ÷ (popularity + 1) — audience conviction per unit of buzz.")


def page_deep_dive(df: pd.DataFrame) -> None:
    st.subheader("Interactive catalog explorer")

    decades = sorted(df["decade"].dropna().unique().tolist())
    segments = df["segment"].unique().tolist()
    themes = sorted(df["theme"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        rating_range = st.slider(
            "Score range",
            float(df["vote_average"].min()),
            float(df["vote_average"].max()),
            (float(df["vote_average"].min()), float(df["vote_average"].max())),
            0.1,
        )
    with c2:
        pick_decades = st.multiselect("Decade", decades, default=decades)
    with c3:
        pick_segments = st.multiselect("Segment", segments, default=segments)

    c4, c5 = st.columns(2)
    with c4:
        pick_themes = st.multiselect("Theme", themes, default=themes)
    with c5:
        search = st.text_input("Search title", "")

    filtered = filter_dataframe(
        df,
        rating_range=rating_range,
        decades=pick_decades,
        segments=pick_segments,
        themes=pick_themes,
        search=search.strip(),
    )
    st.caption(f"**{len(filtered)}** titles match filters · {len(df)} in full catalog")

    view = st.selectbox(
        "Sort by",
        ["Quality × reach", "Score", "Popularity", "Votes", "Engagement index"],
    )
    sort_map = {
        "Quality × reach": "quality_pop_score",
        "Score": "vote_average",
        "Popularity": "popularity",
        "Votes": "vote_count",
        "Engagement index": "engagement_index",
    }
    show = filtered.sort_values(sort_map[view], ascending=False)[
        [
            "name",
            "vote_average",
            "popularity",
            "vote_count",
            "segment",
            "theme",
            "air_year",
            "engagement_index",
        ]
    ]
    st.dataframe(show, use_container_width=True, hide_index=True)

    if len(filtered) >= 3:
        fig = px.scatter(
            filtered,
            x="popularity",
            y="vote_average",
            size="vote_count",
            color="theme",
            hover_name="name",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def page_predict(df: pd.DataFrame) -> None:
    if not model_ready():
        st.error("Model not found. Run: `make train-quick`")
        return

    st.subheader("Greenlight sensitivity — score forecast")
    st.caption("ML model estimates audience score from metadata (for scenario planning).")

    source = st.radio("Input", ["Manual scenario", "Benchmark from catalog"], horizontal=True)

    if source == "Benchmark from catalog":
        pick = st.selectbox("Title", df["name"].tolist())
        row = df[df["name"] == pick].iloc[0]
        name, overview = row["name"], row["overview"]
        popularity, vote_count = float(row["popularity"]), int(row["vote_count"])
        adult = bool(row["adult_flag"])
        air = row["first_air_date"]
        first_air_date = air.date() if pd.notna(air) and hasattr(air, "date") else date(2010, 1, 1)
        actual = float(row["vote_average"])
    else:
        name = st.text_input("Working title", "Untitled drama")
        overview = st.text_area("Logline / overview", "A prestige limited series in a near-future setting.")
        col1, col2 = st.columns(2)
        popularity = col1.number_input("Expected buzz index", 0.0, 200.0, 55.0, 1.0)
        vote_count = col2.number_input("Projected vote volume", 0, 500_000, 5000, 500)
        adult = st.checkbox("Adult / mature", False)
        first_air_date = st.date_input("Target premiere", date(2026, 1, 1))
        actual = None

    if st.button("Run forecast", type="primary"):
        pred = predict_vote_average(
            name=name,
            overview=overview,
            popularity=popularity,
            vote_count=vote_count,
            adult=adult,
            first_air_date=first_air_date.isoformat(),
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast score", f"{pred:.2f} / 10")
        benchmark = df["vote_average"].median()
        m2.metric("Catalog median", f"{benchmark:.2f}")
        delta = pred - benchmark
        m3.metric("vs median", f"{delta:+.2f}", delta_color="normal" if delta >= 0 else "inverse")
        if actual is not None:
            st.metric("Actual (catalog)", f"{actual:.2f}", delta=f"{pred - actual:+.2f} model gap")


def page_data_model() -> None:
    st.subheader("Data & model appendix")
    metrics = load_metrics()
    meta = load_training_meta()
    if metrics:
        st.json(metrics)
    if meta:
        st.json(meta)
    st.code("make train-quick", language="bash")


def main() -> None:
    st.set_page_config(
        page_title="TV Market Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(EXEC_CSS, unsafe_allow_html=True)

    st.title("Global TV Catalog Intelligence")
    st.markdown(
        "**Board-ready view** of top-rated television · TMDB market proxy · "
        "[Dataset source](https://www.kaggle.com/datasets/rosemeenshaikh/op-rated-tv-shows-from-tmdb)"
    )

    with st.sidebar:
        st.header("Executive control panel")
        uploaded = st.file_uploader("Upload full Kaggle CSV", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            df = prepare_analytics_df(raw)
            st.caption("Uploaded catalog")
        else:
            df = load_shows(str(DEFAULT_DATA))
            st.caption("Sample catalog (upload full data for production brief)")

        kpis = compute_kpis(df)
        st.divider()
        st.metric("Titles", kpis.catalog_size)
        st.metric("Premium %", f"{kpis.pct_premium:.0f}%")
        if model_ready():
            st.success("Forecast model ready")
        else:
            st.warning("Run `make train-quick`")

        report_date = st.date_input("Briefing date", date.today())
        st.caption(f"As of {report_date.isoformat()}")

    tabs = st.tabs(
        [
            "Executive brief",
            "Market landscape",
            "Portfolio strategy",
            "Competitive intel",
            "Deep dive",
            "Forecast",
            "Appendix",
        ]
    )
    with tabs[0]:
        page_executive_brief(df, kpis)
    with tabs[1]:
        page_market_landscape(df)
    with tabs[2]:
        page_portfolio_strategy(df)
    with tabs[3]:
        page_competitive_intel(df)
    with tabs[4]:
        page_deep_dive(df)
    with tabs[5]:
        page_predict(df)
    with tabs[6]:
        page_data_model()


if __name__ == "__main__":
    main()
