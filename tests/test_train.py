from __future__ import annotations

from pathlib import Path

from ml.train import train

ROOT = Path(__file__).resolve().parents[1]


def test_train_produces_metrics_and_model(tmp_path: Path) -> None:
    metrics = train(
        ROOT / "data" / "sample_tv_shows.csv",
        tmp_path,
        test_size=0.2,
        task="classification",
        use_mlflow=False,
    )
    assert metrics["task"] == "classification"
    assert "accuracy" in metrics and "f1_macro" in metrics
    assert metrics["n_train"] > 0
    assert (tmp_path / "tmdb_rating_pipeline.joblib").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "training_meta.json").is_file()


def test_train_regression_metrics(tmp_path: Path) -> None:
    metrics = train(
        ROOT / "data" / "sample_tv_shows.csv",
        tmp_path,
        test_size=0.2,
        task="regression",
        use_mlflow=False,
    )
    assert metrics["task"] == "regression"
    assert "mae" in metrics and "r2" in metrics
    assert (tmp_path / "training_meta.json").is_file()
