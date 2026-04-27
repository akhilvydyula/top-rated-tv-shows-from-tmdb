from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _trained_model_and_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    out = tmp_path_factory.mktemp("models")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.train",
            "--no-mlflow",
            "--data",
            str(ROOT / "data" / "sample_tv_shows.csv"),
            "--out",
            str(out),
        ],
        check=True,
        cwd=str(ROOT),
    )
    os.environ["MODEL_PATH"] = str(out / "tmdb_rating_pipeline.joblib")
    from app.model_runtime import clear_model_cache

    clear_model_cache()
