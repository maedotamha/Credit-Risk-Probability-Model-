"""API tests. The model is mocked so these run without a trained MLflow model
(CI does not train a model before running the test suite)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class DummyModel:
    """Stands in for the registered serving pipeline: one fixed probability
    per unique CustomerId, in sorted order (matching predict_risk_probability's
    real sorting contract)."""

    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        n_customers = transactions["CustomerId"].nunique()
        p = np.full(n_customers, 0.25)
        return np.column_stack([1 - p, p])


@pytest.fixture
def client():
    with patch("src.api.main.load_model", return_value=DummyModel()):
        from src.api.main import app

        with TestClient(app) as test_client:
            yield test_client


def _sample_transaction(customer_id: str, transaction_id: str) -> dict:
    return {
        "TransactionId": transaction_id,
        "BatchId": "BatchId_1",
        "AccountId": "AccountId_1",
        "SubscriptionId": "SubscriptionId_1",
        "CustomerId": customer_id,
        "CurrencyCode": "UGX",
        "CountryCode": 256,
        "ProviderId": "ProviderId_1",
        "ProductId": "ProductId_1",
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_2",
        "Amount": 100.0,
        "Value": 100.0,
        "TransactionStartTime": "2019-01-01T10:00:00Z",
        "PricingStrategy": 2,
        "FraudResult": 0,
    }


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_probability_per_customer(client):
    payload = {
        "transactions": [
            _sample_transaction("CustomerId_1", "t1"),
            _sample_transaction("CustomerId_2", "t2"),
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    customer_ids = {p["CustomerId"] for p in body["predictions"]}
    assert customer_ids == {"CustomerId_1", "CustomerId_2"}
    for prediction in body["predictions"]:
        assert prediction["risk_probability"] == pytest.approx(0.25)


def test_predict_endpoint_rejects_empty_transaction_list(client):
    response = client.post("/predict", json={"transactions": []})

    assert response.status_code == 422


def test_predict_endpoint_returns_503_when_model_not_loaded():
    with patch("src.api.main.load_model", side_effect=RuntimeError("no model registered")):
        from src.api.main import app

        with TestClient(app) as test_client:
            response = test_client.post(
                "/predict", json={"transactions": [_sample_transaction("CustomerId_1", "t1")]}
            )

    assert response.status_code == 503
