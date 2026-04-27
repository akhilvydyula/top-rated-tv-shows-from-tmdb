FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/tmdb_rating_pipeline.joblib \
    MLFLOW_DISABLE=1

COPY pyproject.toml README.md ./
COPY ml ./ml
COPY app ./app
COPY data/sample_tv_shows.csv ./data/sample_tv_shows.csv

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    python -m ml.train --no-mlflow --data data/sample_tv_shows.csv --out models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
