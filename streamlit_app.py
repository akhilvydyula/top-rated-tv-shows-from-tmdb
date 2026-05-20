"""Streamlit dashboard for TMDB top-rated TV shows exploration and rating prediction."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from app.model_runtime import load_pipeline, load_training_meta, predict_vote_average

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sample_tv_shows.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"


@st.cache_data
def load_shows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["first_air_date"] = pd.to_datetime(df["first_air_date"], errors="coerce")
    df["air_year"] = df["first_air_date"].dt.year
    return df


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


def page_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shows", len(df))
    c2.metric("Avg rating", f"{df['vote_average'].mean():.2f}")
    c3.metric("Avg popularity", f"{df['popularity'].mean():.1f}")
    c4.metric("Year range", f"{int(df['air_year'].min())}–{int(df['air_year'].max())}")

    st.markdown(
        "Explore ratings and popularity, then use **Predict** to estimate "
        "`vote_average` with the trained scikit-learn pipeline."
    )
    st.dataframe(
        df.sort_values("vote_average", ascending=False)[
            ["name", "vote_average", "popularity", "vote_count", "first_air_date"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def page_explore(df: pd.DataFrame) -> None:
    st.subheader("Filters")
    min_rating, max_rating = float(df["vote_average"].min()), float(df["vote_average"].max())
    rating_range = st.slider(
        "Rating range",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating),
        step=0.1,
    )
    adult_filter = st.selectbox("Adult content", ["All", "No", "Yes"])
    search = st.text_input("Search by title", "").strip().lower()

    filtered = df[
        (df["vote_average"] >= rating_range[0]) & (df["vote_average"] <= rating_range[1])
    ]
    if adult_filter == "No":
        filtered = filtered[~filtered["adult"].astype(str).str.lower().isin(("true", "1", "yes"))]
    elif adult_filter == "Yes":
        filtered = filtered[filtered["adult"].astype(str).str.lower().isin(("true", "1", "yes"))]
    if search:
        filtered = filtered[filtered["name"].str.lower().str.contains(search, na=False)]

    st.caption(f"{len(filtered)} show(s) match your filters")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Rating distribution**")
        st.bar_chart(filtered["vote_average"].value_counts().sort_index())
    with right:
        st.markdown("**Popularity vs rating**")
        chart_df = filtered[["popularity", "vote_average"]].rename(
            columns={"popularity": "Popularity", "vote_average": "Rating"}
        )
        st.scatter_chart(chart_df, x="Popularity", y="Rating")


def page_predict(df: pd.DataFrame) -> None:
    if not model_ready():
        st.error(
            "Trained model not found. Run:\n\n"
            "`python -m ml.train --data data/sample_tv_shows.csv --out models`"
        )
        return

    st.subheader("Predict rating")
    source = st.radio("Input source", ["Manual entry", "Pick from dataset"], horizontal=True)

    if source == "Pick from dataset":
        names = df["name"].tolist()
        pick = st.selectbox("Show", names)
        row = df[df["name"] == pick].iloc[0]
        name = row["name"]
        overview = row["overview"]
        popularity = float(row["popularity"])
        vote_count = int(row["vote_count"])
        adult = bool(
            str(row["adult"]).lower() in ("true", "1", "yes") if pd.notna(row["adult"]) else False
        )
        air = row["first_air_date"]
        first_air_date = air.date() if hasattr(air, "date") and pd.notna(air) else date(2000, 1, 1)
        actual = float(row["vote_average"])
    else:
        name = st.text_input("Title", "Breaking Bad")
        overview = st.text_area("Overview", "A chemistry teacher turns to manufacturing meth.")
        col1, col2 = st.columns(2)
        popularity = col1.number_input("Popularity", min_value=0.0, value=50.0, step=1.0)
        vote_count = col2.number_input("Vote count", min_value=0, value=1000, step=100)
        adult = st.checkbox("Adult", value=False)
        first_air_date = st.date_input("First air date", value=date(2010, 1, 1))
        actual = None

    if st.button("Predict", type="primary"):
        pred = predict_vote_average(
            name=name,
            overview=overview,
            popularity=popularity,
            vote_count=vote_count,
            adult=adult,
            first_air_date=first_air_date.isoformat(),
        )
        st.metric("Predicted vote average", f"{pred:.2f} / 10")
        if actual is not None:
            delta = pred - actual
            st.metric("Actual vote average", f"{actual:.2f}", delta=f"{delta:+.2f}")


def page_model() -> None:
    st.subheader("Model info")
    metrics = load_metrics()
    meta = load_training_meta()

    if metrics:
        st.json(metrics)
        if "accuracy" in metrics:
            st.info(
                f"Exact star accuracy: **{metrics['accuracy']:.1%}**. "
                f"Within 1 star: **{metrics.get('pct_within_1_star', 0):.1%}**."
            )
    else:
        st.warning("No `models/metrics.json` found. Train the model to see metrics.")

    if meta:
        st.markdown("**Training metadata**")
        st.json(meta)

    st.markdown(
        "**Retrain**\n\n"
        "```bash\n"
        "python -m ml.train --data data/sample_tv_shows.csv --out models\n"
        "```"
    )


def main() -> None:
    st.set_page_config(
        page_title="TMDB Top TV Shows",
        page_icon="📺",
        layout="wide",
    )
    st.title("📺 TMDB Top-Rated TV Shows")
    st.caption(
        "[Kaggle dataset](https://www.kaggle.com/datasets/rosemeenshaikh/op-rated-tv-shows-from-tmdb) "
        "· scikit-learn rating model"
    )

    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
        if uploaded is not None:
            data_path = uploaded
            df = pd.read_csv(data_path)
            df["first_air_date"] = pd.to_datetime(df["first_air_date"], errors="coerce")
            df["air_year"] = df["first_air_date"].dt.year
        else:
            data_path = DEFAULT_DATA
            df = load_shows(str(data_path))
        st.caption(f"Using: `{data_path}`" if isinstance(data_path, Path) else "Uploaded file")
        if model_ready():
            st.success("Model loaded")
        else:
            st.warning("Model not trained")

    tab_overview, tab_explore, tab_predict, tab_model = st.tabs(
        ["Overview", "Explore", "Predict", "Model"]
    )
    with tab_overview:
        page_overview(df)
    with tab_explore:
        page_explore(df)
    with tab_predict:
        page_predict(df)
    with tab_model:
        page_model()


if __name__ == "__main__":
    main()
