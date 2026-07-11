import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.train import evaluate, split_features_target


def test_split_features_target_drops_non_feature_columns():
    processed = pd.DataFrame(
        {
            "CustomerId": ["c1", "c2"],
            "Recency": [10, 20],
            "Frequency": [5, 1],
            "Monetary": [100.0, 5.0],
            "total_amount": [50.0, 5.0],
            "is_high_risk": [0, 1],
        }
    )

    X, y = split_features_target(processed)

    assert list(X.columns) == ["total_amount"]
    assert y.tolist() == [0, 1]


def test_evaluate_returns_expected_metric_keys_and_perfect_scores_for_perfect_model():
    X_train = pd.DataFrame({"x": [0, 0, 0, 1, 1, 1]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(X_train, y_train)

    metrics = evaluate(model, X_train, y_train)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert all(np.isfinite(v) for v in metrics.values())
