from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ShowInput(BaseModel):
    """Single show payload matching Kaggle TMDB columns used for training."""

    name: str = Field(..., min_length=1, max_length=500)
    overview: str = Field(default="", max_length=20000)
    popularity: float = Field(..., ge=0)
    vote_count: int = Field(..., ge=0)
    adult: bool = False
    first_air_date: date | None = Field(
        default=None,
        description="ISO date; if omitted, a default year is imputed like training.",
    )


class PredictionOut(BaseModel):
    predicted_vote_average: float = Field(
        ..., ge=0, le=10, description="Predicted TMDB-style 0–10 rating."
    )


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
