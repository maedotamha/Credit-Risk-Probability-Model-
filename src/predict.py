"""Inference helpers for the credit risk project.

Loads the registered MLflow serving pipeline (Task 3 feature engineering +
trained classifier, see ``src/train.py``) and scores new, raw customer
transaction data. Used by the FastAPI service in ``src/api/main.py``.
"""

from __future__ import annotations

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


def load_model(model_uri: str | None = None) -> Pipeline:
    """Load the registered serving pipeline from the MLflow Model Registry.

    :param model_uri: defaults to the latest version of ``REGISTERED_MODEL_NAME``
        (``models:/<name>/latest``). Pass an explicit URI (e.g. a specific
        version or a stage alias) to pin a particular model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if model_uri is None:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
    return mlflow.sklearn.load_model(model_uri)


def predict_risk_probability(model: Pipeline, transactions: pd.DataFrame) -> pd.DataFrame:
    """Score raw customer transactions, returning one risk probability per customer.

    :param transactions: raw Xente-schema transaction rows for one or more
        customers (matching ``REQUIRED_TRANSACTION_COLUMNS`` in
        ``src/data_processing.py``). May contain multiple transactions per
        customer; the model's feature pipeline aggregates to one row per
        ``CustomerId``.

    The feature pipeline's internal ``groupby("CustomerId")`` sorts by customer
    ID (pandas' default), so the returned probabilities are aligned against
    *sorted* unique customer IDs, not their first-appearance order in
    ``transactions``.
    """
    customer_ids = pd.Series(transactions["CustomerId"].unique()).sort_values().reset_index(drop=True)
    probabilities = model.predict_proba(transactions)[:, 1]
    return pd.DataFrame({"CustomerId": customer_ids, "risk_probability": probabilities})
