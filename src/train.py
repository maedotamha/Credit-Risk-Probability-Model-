"""Model training entry point for the Bati Bank credit risk project.

Loads raw transactions, builds the processed (Task 3 + Task 4) dataset, trains and
tunes two classifiers (Logistic Regression and XGBoost), tracks every run with
MLflow, evaluates against held-out test data, and registers the best-performing
model in the MLflow Model Registry.

Run with: python -m src.train
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import ColumnDropper, build_processed_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "credit-risk-model"
REGISTERED_MODEL_NAME = "credit-risk-is-high-risk"

RAW_DATA_PATH = Path("data") / "raw" / "data.csv"
PROCESSED_DATA_PATH = Path("data") / "processed" / "model_input.csv"

# Recency/Frequency/Monetary are the exact inputs the Task 4 K-Means proxy label was
# derived from; training on them would let the model trivially reconstruct cluster
# membership instead of learning generalizable behavioral signal, and a genuinely
# new applicant has no comparable RFM history to re-cluster against at inference
# time. CustomerId is an identifier, not a feature.
NON_FEATURE_COLUMNS = ["CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"]


def load_processed_data(raw_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, Pipeline]:
    """Run the Task 3 + Task 4 pipeline over the raw data and cache the result.

    Returns both the processed DataFrame and the *fitted* Task 3 feature pipeline,
    since the latter needs to be bundled with the trained classifier for serving
    (see ``build_serving_pipeline``).
    """
    df = pd.read_csv(raw_path, parse_dates=["TransactionStartTime"])
    processed, feature_pipeline = build_processed_dataset(
        df, random_state=RANDOM_STATE, return_pipeline=True
    )
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    return processed, feature_pipeline


def split_features_target(processed: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = processed["is_high_risk"]
    X = processed.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in processed.columns])
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }
    base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }
    base = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )
    # n_jobs=1 on the search (not -1): nesting joblib parallelism with XGBoost's
    # own multithreading crashes intermittently on Windows (access violation in
    # QuantileDMatrix under the loky backend). The dataset is small (~3.7k rows),
    # so this is not a meaningful speed cost.
    search = RandomizedSearchCV(
        base,
        param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def build_serving_pipeline(feature_pipeline: Pipeline, classifier) -> Pipeline:
    """Bundle the fitted Task 3 feature pipeline with a trained classifier.

    Both sub-objects are already fitted; this Pipeline is constructed purely for
    serving (``.predict`` / ``.predict_proba``) and is never itself re-fit, which
    sidesteps the row-count mismatch between transaction-level input and
    customer-level labels that would otherwise block a single end-to-end
    ``Pipeline.fit(X, y)`` call.
    """
    return Pipeline(
        steps=[
            ("features", feature_pipeline),
            ("drop_id", ColumnDropper(columns=["CustomerId"])),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    processed, feature_pipeline = load_processed_data()
    X, y = split_features_target(processed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    trained = {}

    with mlflow.start_run(run_name="logistic_regression"):
        model, best_params = train_logistic_regression(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        signature = mlflow.models.infer_signature(X_train, model.predict_proba(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_train.head(3))
        trained["logistic_regression"] = (model, metrics)
        print("logistic_regression:", metrics)

    with mlflow.start_run(run_name="xgboost"):
        model, best_params = train_xgboost(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        signature = mlflow.models.infer_signature(X_train, model.predict_proba(X_train))
        mlflow.xgboost.log_model(model, name="model", signature=signature, input_example=X_train.head(3))
        trained["xgboost"] = (model, metrics)
        print("xgboost:", metrics)

    best_name = max(trained, key=lambda name: trained[name][1]["roc_auc"])
    best_model, best_metrics = trained[best_name]
    print(f"\nBest model by ROC-AUC: {best_name} -> {best_metrics}")

    # The registered model is the full serving pipeline (raw transactions in, risk
    # probability out), not the bare classifier - see build_serving_pipeline.
    serving_pipeline = build_serving_pipeline(feature_pipeline, best_model)
    raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["TransactionStartTime"])
    raw_df["TransactionStartTime"] = raw_df["TransactionStartTime"].dt.tz_localize(None)
    sample_input = raw_df[raw_df["CustomerId"].isin(raw_df["CustomerId"].unique()[:3])]
    sample_output = serving_pipeline.predict_proba(sample_input)[:, 1]
    signature = mlflow.models.infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name=f"register_{best_name}"):
        mlflow.log_param("model_type", best_name)
        mlflow.log_metrics(best_metrics)
        # cloudpickle (not the "skops"-safe default): the serving pipeline embeds
        # our custom transformer classes (ColumnDropper, CustomerAggregator,
        # DatetimeFeatureExtractor), which skops' security audit refuses to
        # serialize. code_paths bundles src/ into the artifact so the API service
        # can deserialize it even without an identical local import path.
        mlflow.sklearn.log_model(
            serving_pipeline,
            name="model",
            signature=signature,
            input_example=sample_input,
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
            code_paths=["src"],
        )

    print(f"Registered '{best_name}' as '{REGISTERED_MODEL_NAME}' in the MLflow Model Registry.")


if __name__ == "__main__":
    main()
