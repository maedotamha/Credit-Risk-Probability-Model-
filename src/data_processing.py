"""Feature engineering for the Bati Bank credit risk project.

Transforms raw Xente transaction-level data into a customer-level, model-ready
DataFrame. Credit risk decisions are made per customer (the API returns one risk
probability per customer), so the pipeline aggregates transaction-level behavior -
including the time-based features called out in the assignment (transaction hour,
day, month, year) - into customer-level summary statistics rather than leaving the
final dataset at transaction granularity.

The full transformation is expressed as a single ``sklearn.pipeline.Pipeline`` so it
is reproducible and can be fit once and reused identically at training and inference
time (see ``build_feature_pipeline``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_TRANSACTION_COLUMNS = {
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult",
}

# Columns produced by CustomerAggregator, split by how they should be preprocessed.
NUMERIC_FEATURES = [
    "total_amount",
    "avg_amount",
    "std_amount",
    "median_amount",
    "total_value",
    "avg_value",
    "std_value",
    "median_value",
    "transaction_count",
    "total_debit_amount",
    "total_credit_amount",
    "net_amount",
    "credit_debit_ratio",
    "unique_product_categories",
    "unique_channels",
    "unique_providers",
    "active_days",
    "active_months",
    "avg_transaction_hour",
    "std_transaction_hour",
    "fraud_transaction_count",
]
CATEGORICAL_FEATURES = [
    "preferred_channel",
    "preferred_category",
    "preferred_provider",
    "preferred_pricing_strategy",
]


def missing_required_columns(columns: list[str] | set[str]) -> set[str]:
    """Return required Xente columns that are absent from an input dataset."""
    return REQUIRED_TRANSACTION_COLUMNS.difference(set(columns))


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour/day/month/year from ``TransactionStartTime`` (row-level).

    Row count and identity are preserved; this only adds columns, so it can run
    before the customer-level aggregation step.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "DatetimeFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ts = pd.to_datetime(X["TransactionStartTime"], utc=True)
        X["TransactionHour"] = ts.dt.hour
        X["TransactionDay"] = ts.dt.day
        X["TransactionMonth"] = ts.dt.month
        X["TransactionYear"] = ts.dt.year
        X["_TransactionDate"] = ts.dt.date
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level rows into one row per ``CustomerId``.

    Produces the numeric and categorical customer-level features listed in
    ``NUMERIC_FEATURES`` / ``CATEGORICAL_FEATURES``. The result is indexed by
    ``CustomerId`` so downstream steps (encoding, scaling, and later the
    ``is_high_risk`` proxy label) can be joined back unambiguously.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CustomerAggregator":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "_TransactionDate" not in df.columns:
            df = DatetimeFeatureExtractor().transform(df)

        is_debit = df["Amount"] > 0
        is_credit = df["Amount"] < 0

        grouped = df.groupby("CustomerId")

        agg = pd.DataFrame(index=grouped.size().index)
        agg["total_amount"] = grouped["Amount"].sum()
        agg["avg_amount"] = grouped["Amount"].mean()
        agg["std_amount"] = grouped["Amount"].std()
        agg["median_amount"] = grouped["Amount"].median()
        agg["total_value"] = grouped["Value"].sum()
        agg["avg_value"] = grouped["Value"].mean()
        agg["std_value"] = grouped["Value"].std()
        agg["median_value"] = grouped["Value"].median()
        agg["transaction_count"] = grouped.size()
        agg["total_debit_amount"] = df.loc[is_debit].groupby("CustomerId")["Amount"].sum()
        agg["total_credit_amount"] = df.loc[is_credit].groupby("CustomerId")["Amount"].sum().abs()
        agg["net_amount"] = agg["total_debit_amount"].fillna(0) - agg["total_credit_amount"].fillna(0)
        debit_denom = agg["total_debit_amount"].replace(0, np.nan)
        agg["credit_debit_ratio"] = agg["total_credit_amount"].fillna(0) / debit_denom
        agg["unique_product_categories"] = grouped["ProductCategory"].nunique()
        agg["unique_channels"] = grouped["ChannelId"].nunique()
        agg["unique_providers"] = grouped["ProviderId"].nunique()
        agg["active_days"] = grouped["_TransactionDate"].nunique()
        agg["active_months"] = grouped["TransactionMonth"].nunique()
        agg["avg_transaction_hour"] = grouped["TransactionHour"].mean()
        agg["std_transaction_hour"] = grouped["TransactionHour"].std()
        agg["fraud_transaction_count"] = grouped["FraudResult"].sum()

        def _mode(series: pd.Series):
            m = series.mode()
            return m.iloc[0] if not m.empty else np.nan

        agg["preferred_channel"] = grouped["ChannelId"].agg(_mode)
        agg["preferred_category"] = grouped["ProductCategory"].agg(_mode)
        agg["preferred_provider"] = grouped["ProviderId"].agg(_mode)
        agg["preferred_pricing_strategy"] = grouped["PricingStrategy"].agg(_mode).astype(str)

        # std is NaN (not 0) for single-transaction customers; that is a real
        # "unknown variability" case, imputed downstream rather than assumed to be 0.
        agg.index.name = "CustomerId"
        return agg.reset_index()


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop identifier columns that should not reach the classifier.

    Used at the end of the serving pipeline: the Task 3 feature pipeline keeps
    ``CustomerId`` (via ``remainder='passthrough'``) for traceability, but the
    classifier is trained without it, so it must be dropped before ``predict``.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or ["CustomerId"]

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns if c in X.columns])


def build_feature_pipeline() -> Pipeline:
    """Build the full raw-transactions -> model-ready-DataFrame pipeline.

    The returned ``Pipeline`` is unfitted. Calling ``fit_transform`` on a raw Xente
    transaction DataFrame returns a customer-level DataFrame (indexed by row, with
    a ``CustomerId`` column preserved via ``remainder='passthrough'``) whose numeric
    columns are imputed + standardized and whose categorical columns are imputed +
    one-hot encoded.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # keeps CustomerId
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    # DatetimeFeatureExtractor and CustomerAggregator already return pandas
    # DataFrames directly, so only the ColumnTransformer needs set_output.
    pipeline = Pipeline(
        steps=[
            ("extract_datetime", DatetimeFeatureExtractor()),
            ("aggregate", CustomerAggregator()),
            ("preprocess", preprocessor),
        ]
    )
    return pipeline


def engineer_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: fit and apply the feature pipeline in one call."""
    pipeline = build_feature_pipeline()
    return pipeline.fit_transform(transactions)


# ---------------------------------------------------------------------------
# Weight of Evidence / Information Value
# ---------------------------------------------------------------------------


def calculate_woe_iv(
    df: pd.DataFrame, feature: str, target: str, n_bins: int = 5
) -> tuple[pd.DataFrame, float]:
    """Compute Weight of Evidence and Information Value for one feature.

    ``target`` must be a binary 0/1 column (e.g. ``is_high_risk``). Numeric features
    are bucketed into ``n_bins`` quantile bins first; categorical features are used
    as-is. Returns a per-bin summary DataFrame and the feature's total IV.

    WoE is only meaningful once a target exists, so this is applied after the Task 4
    proxy label has been created - and it is diagnostic/interpretability tooling for
    the Logistic Regression baseline, not a required step for every model.
    """
    work = df[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(work[feature]) and work[feature].nunique() > n_bins:
        work[feature] = pd.qcut(work[feature], q=n_bins, duplicates="drop")

    grouped = work.groupby(feature, observed=True)[target].agg(["sum", "count"])
    grouped = grouped.rename(columns={"sum": "bad", "count": "total"})
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    # Laplace-style smoothing avoids log(0) for bins with zero good/bad counts.
    dist_bad = (grouped["bad"] + 0.5) / (total_bad + 0.5 * len(grouped))
    dist_good = (grouped["good"] + 0.5) / (total_good + 0.5 * len(grouped))

    grouped["woe"] = np.log(dist_good / dist_bad)
    grouped["iv"] = (dist_good - dist_bad) * grouped["woe"]

    total_iv = grouped["iv"].sum()
    return grouped.reset_index(), float(total_iv)


def rank_features_by_iv(
    df: pd.DataFrame, features: list[str], target: str, n_bins: int = 5
) -> pd.DataFrame:
    """Rank candidate features by Information Value against a binary target."""
    rows = []
    for feature in features:
        try:
            _, iv = calculate_woe_iv(df, feature, target, n_bins=n_bins)
        except (ValueError, TypeError):
            iv = np.nan
        rows.append({"feature": feature, "iv": iv})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 4: RFM-based proxy target ("is_high_risk")
#
# The raw dataset has no default label. As a stand-in, customers are segmented by
# Recency/Frequency/Monetary behavior into 3 K-Means clusters; the least engaged
# cluster (long time since last transaction, few transactions, low transaction
# value) is treated as the high-risk proxy. This is an explicit modeling
# assumption, not a measured default outcome - see the final report for the
# business risks this introduces.
# ---------------------------------------------------------------------------

RANDOM_STATE = 42


def calculate_rfm(transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary value per ``CustomerId``.

    :param snapshot_date: reference date Recency is measured against. Defaults to
        one day after the latest transaction in the data, so every customer has a
        strictly positive Recency and the metric is stable if rerun on the same
        historical extract.
    """
    df = transactions.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.Timestamp(snapshot_date)
        if snapshot_date.tzinfo is None:
            snapshot_date = snapshot_date.tz_localize("UTC")

    grouped = df.groupby("CustomerId")
    rfm = pd.DataFrame(
        {
            "Recency": (snapshot_date - grouped["TransactionStartTime"].max()).dt.days,
            "Frequency": grouped.size(),
            "Monetary": grouped["Value"].sum(),
        }
    )
    rfm.index.name = "CustomerId"
    return rfm.reset_index()


def assign_high_risk_label(
    rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Cluster customers on scaled RFM values and label the least-engaged cluster.

    The least-engaged cluster is identified automatically (not by manual
    inspection) as the cluster with the highest mean Recency and lowest mean
    Frequency/Monetary, using z-scored cluster-level means so the three RFM
    dimensions contribute comparably regardless of their raw scale.

    Frequency and Monetary are log1p-transformed before scaling: both are
    extremely right-skewed (Task 1 EDA: skew ~51 on the underlying transaction
    values), and clustering on raw values lets a handful of extreme outliers
    dominate the distance metric, producing a degenerate "cluster" of a few
    outlier customers instead of 3 meaningful behavioral segments.
    """
    features = pd.DataFrame(
        {
            "Recency": rfm["Recency"],
            "Frequency": np.log1p(rfm["Frequency"]),
            "Monetary": np.log1p(rfm["Monetary"]),
        }
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled)

    result = rfm.copy()
    result["Cluster"] = clusters

    cluster_profile = result.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    z_profile = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
    # High recency is bad (long time since last purchase); high frequency/monetary
    # is good. A larger risk_score means less engaged / higher assumed risk.
    risk_score = z_profile["Recency"] - z_profile["Frequency"] - z_profile["Monetary"]
    high_risk_cluster = risk_score.idxmax()

    result["is_high_risk"] = (result["Cluster"] == high_risk_cluster).astype(int)
    return result


def build_processed_dataset(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
    return_pipeline: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Pipeline]:
    """End-to-end Task 3 + Task 4 output: model-ready features plus ``is_high_risk``.

    This is the "main processed dataset" used for model training in Task 5: one
    row per customer, engineered/encoded/scaled features from the Task 3 feature
    pipeline, joined to the RFM-derived ``is_high_risk`` proxy label from
    ``calculate_rfm`` + ``assign_high_risk_label``.

    :param return_pipeline: if True, also return the *fitted* feature pipeline.
        Training code needs this reference so it can be bundled with the trained
        classifier into one serving pipeline (see ``src/train.py``) - at inference
        time the API must reapply the exact scaler/encoder learned during training
        to new raw transactions, not refit on whatever single customer it receives.
    """
    feature_pipeline = build_feature_pipeline()
    features = feature_pipeline.fit_transform(transactions)

    rfm = calculate_rfm(transactions, snapshot_date=snapshot_date)
    rfm_labeled = assign_high_risk_label(rfm, n_clusters=n_clusters, random_state=random_state)

    processed = features.merge(
        rfm_labeled[["CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"]],
        on="CustomerId",
        how="left",
    )
    if return_pipeline:
        return processed, feature_pipeline
    return processed
