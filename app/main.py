from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_runtime import clear_model_cache, load_pipeline, predict_vote_average
from app.schemas import HealthOut, PredictionOut, ShowInput


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    clear_model_cache()


app = FastAPI(
    title="TMDB Top TV — Rating API",
    version="0.1.0",
    description=(
        "Predicts vote_average (0–10) from popularity, votes, text length, "
        "air year, and adult flag."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    try:
        load_pipeline()
        return HealthOut(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthOut(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionOut)
def predict(body: ShowInput) -> PredictionOut:
    try:
        load_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    d = body.first_air_date.isoformat() if body.first_air_date else None
    score = predict_vote_average(
        name=body.name,
        overview=body.overview,
        popularity=body.popularity,
        vote_count=body.vote_count,
        adult=body.adult,
        first_air_date=d,
    )
    return PredictionOut(predicted_vote_average=round(score, 3))
