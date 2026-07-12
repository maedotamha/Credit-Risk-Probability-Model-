"""Pydantic schemas for the prediction API.

The ``/predict`` endpoint accepts raw Xente-schema transactions (not
pre-engineered features): the registered model is a full serving pipeline that
does its own feature engineering, so the loan origination team only needs to
supply the same transaction data the eCommerce platform already produces, for
one or more customers. A response is returned per unique ``CustomerId``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class TransactionRecord(BaseModel):
    """One raw transaction row, matching the Xente dataset schema exactly."""

    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: datetime
    PricingStrategy: int
    FraudResult: int = Field(ge=0, le=1)


class PredictionRequest(BaseModel):
    """One or more transactions, possibly spanning several customers.

    Multiple transactions per ``CustomerId`` are expected and required: the
    model's feature pipeline aggregates behavioral history per customer, so a
    single transaction cannot capture the recency/frequency/monetary and
    diversity signals the model was trained on.
    """

    transactions: list[TransactionRecord] = Field(min_length=1)


class CustomerRiskPrediction(BaseModel):
    CustomerId: str
    risk_probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predictions: list[CustomerRiskPrediction]
