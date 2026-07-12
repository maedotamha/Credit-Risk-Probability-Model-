"""FastAPI service for the Bati Bank credit risk model.

Loads the best model registered in the MLflow Model Registry (see
``src/train.py``) at startup and exposes a ``/predict`` endpoint that scores raw
customer transaction data with a risk probability.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.pydantic_models import (
    CustomerRiskPrediction,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.predict import load_model, predict_risk_probability

logger = logging.getLogger("credit_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
        logger.info("Loaded registered model from the MLflow Model Registry.")
    except Exception:
        # A model-loading failure (e.g. no run has been trained/registered yet)
        # should not prevent the service from starting - /health must still work
        # so orchestration tooling can tell the container is up, and /predict
        # fails loudly with a clear 503 rather than the process crashing.
        logger.exception("Failed to load model at startup; /predict will return 503.")
        app.state.model = None
    yield
    app.state.model = None


app = FastAPI(title="Credit Risk Probability Model", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and register a model first.")

    transactions = pd.DataFrame([t.model_dump() for t in payload.transactions])
    try:
        result = predict_risk_probability(model, transactions)
    except Exception as exc:  # noqa: BLE001 - surface as a client-facing 422
        raise HTTPException(status_code=422, detail=f"Could not score the supplied transactions: {exc}")

    predictions = [
        CustomerRiskPrediction(CustomerId=row.CustomerId, risk_probability=float(row.risk_probability))
        for row in result.itertuples()
    ]
    return PredictionResponse(predictions=predictions)
