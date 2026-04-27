from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_score() -> None:
    r = client.post(
        "/predict",
        json={
            "name": "Test Show",
            "overview": "A test overview for unit testing the API.",
            "popularity": 50.0,
            "vote_count": 1000,
            "adult": False,
            "first_air_date": "2015-06-01",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "predicted_vote_average" in data
    assert 0 <= data["predicted_vote_average"] <= 10
