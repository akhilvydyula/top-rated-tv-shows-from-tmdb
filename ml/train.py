"""Train a regression model to predict TMDB vote_average from listing features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import build_feature_frame


def _tracking_uri_default() -> str:
    return (Path.cwd() / "mlruns").resolve().as_uri()


def _use_mlflow_flag(use_mlflow: bool | None) -> bool:
    if use_mlflow is not None:
        return use_mlflow
    return os.environ.get("MLFLOW_DISABLE", "").lower() not in ("1", "true", "yes")


def build_pipeline() -> Pipeline:
    numeric = ["popularity", "log_vote_count", "air_year", "overview_len", "name_len"]
    binary = ["adult"]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("bin", "passthrough", binary),
        ]
    )
    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=300,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline([("prep", pre), ("model", reg)])


def train(
    data_path: Path,
    out_dir: Path,
    test_size: float = 0.2,
    *,
    use_mlflow: bool | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    required = {
        "first_air_date",
        "name",
        "overview",
        "popularity",
        "vote_average",
        "vote_count",
        "adult",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    y = pd.to_numeric(df["vote_average"], errors="coerce")
    mask = y.notna() & (y >= 0) & (y <= 10)
    df, y = df.loc[mask].reset_index(drop=True), y.loc[mask]

    X = build_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe = build_pipeline()
    do_mlflow = _use_mlflow_flag(use_mlflow)

    if do_mlflow:
        import mlflow
        from mlflow import sklearn as mlflow_sklearn

        tracking = os.environ.get("MLFLOW_TRACKING_URI") or _tracking_uri_default()
        experiment = os.environ.get("MLFLOW_EXPERIMENT", "tmdb-tv-rating")
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(experiment)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "test_size": test_size,
                    "data_path": str(data_path.resolve()),
                    "n_rows_raw": int(len(df)),
                }
            )
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            metrics = {
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
                "n_train": float(len(X_train)),
                "n_test": float(len(X_test)),
            }
            mlflow.log_metrics(metrics)

            model_path = out_dir / "tmdb_rating_pipeline.joblib"
            joblib.dump(pipe, model_path)
            (out_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mae": metrics["mae"],
                        "r2": metrics["r2"],
                        "n_train": int(metrics["n_train"]),
                        "n_test": int(metrics["n_test"]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            reg_name = os.environ.get("MLFLOW_MODEL_NAME")
            log_model_kwargs = {"artifact_path": "sklearn-model"}
            if reg_name and not str(tracking).lower().startswith("file:"):
                log_model_kwargs["registered_model_name"] = reg_name
            mlflow_sklearn.log_model(pipe, **log_model_kwargs)

            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(out_dir / "metrics.json"))
        return {
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
        }

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model_path = out_dir / "tmdb_rating_pipeline.joblib"
    joblib.dump(pipe, model_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/tv_shows.csv"))
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = p.parse_args()
    metrics = train(
        args.data,
        args.out,
        test_size=args.test_size,
        use_mlflow=not args.no_mlflow,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
