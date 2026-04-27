from __future__ import annotations

from pathlib import Path

from ml.train import train

ROOT = Path(__file__).resolve().parents[1]


def test_train_produces_metrics_and_model(tmp_path: Path) -> None:
    metrics = train(
        ROOT / "data" / "sample_tv_shows.csv",
        tmp_path,
        test_size=0.2,
        use_mlflow=False,
    )
    assert "mae" in metrics and "r2" in metrics
    assert metrics["n_train"] > 0
    assert (tmp_path / "tmdb_rating_pipeline.joblib").is_file()
    assert (tmp_path / "metrics.json").is_file()
