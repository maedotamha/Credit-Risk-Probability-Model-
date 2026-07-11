import pandas as pd
import pytest

from src.data_processing import (
    CustomerAggregator,
    DatetimeFeatureExtractor,
    assign_high_risk_label,
    build_processed_dataset,
    calculate_rfm,
    calculate_woe_iv,
    engineer_customer_features,
    missing_required_columns,
)


def test_missing_required_columns_returns_empty_set_for_complete_schema():
    columns = {
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

    assert missing_required_columns(columns) == set()


def test_missing_required_columns_flags_absent_fields():
    assert missing_required_columns(["CustomerId", "Amount"]) == {
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "Value",
        "TransactionStartTime",
        "PricingStrategy",
        "FraudResult",
    }


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionId": ["t1", "t2", "t3", "t4"],
            "CustomerId": ["c1", "c1", "c2", "c2"],
            "AccountId": ["a1", "a1", "a2", "a2"],
            "SubscriptionId": ["s1", "s1", "s2", "s2"],
            "BatchId": ["b1", "b1", "b2", "b2"],
            "CurrencyCode": ["UGX"] * 4,
            "CountryCode": [256] * 4,
            "ProviderId": ["ProviderId_1", "ProviderId_1", "ProviderId_2", "ProviderId_2"],
            "ProductId": ["ProductId_1", "ProductId_1", "ProductId_2", "ProductId_2"],
            "ProductCategory": ["airtime", "airtime", "data_bundles", "data_bundles"],
            "ChannelId": ["ChannelId_2", "ChannelId_2", "ChannelId_3", "ChannelId_3"],
            "Amount": [100.0, -50.0, 200.0, 200.0],
            "Value": [100.0, 50.0, 200.0, 200.0],
            "TransactionStartTime": [
                "2019-01-01T10:00:00Z",
                "2019-01-02T14:00:00Z",
                "2019-01-01T09:00:00Z",
                "2019-01-01T09:00:00Z",
            ],
            "PricingStrategy": [2, 2, 4, 4],
            "FraudResult": [0, 0, 0, 1],
        }
    )


def test_datetime_feature_extractor_adds_expected_columns(sample_transactions):
    result = DatetimeFeatureExtractor().transform(sample_transactions)

    expected_cols = {"TransactionHour", "TransactionDay", "TransactionMonth", "TransactionYear"}
    assert expected_cols.issubset(result.columns)
    assert result["TransactionHour"].tolist() == [10, 14, 9, 9]
    assert result["TransactionMonth"].tolist() == [1, 1, 1, 1]
    assert result["TransactionYear"].tolist() == [2019, 2019, 2019, 2019]


def test_customer_aggregator_computes_expected_values(sample_transactions):
    result = CustomerAggregator().transform(sample_transactions)
    result = result.set_index("CustomerId")

    assert set(result.index) == {"c1", "c2"}
    assert result.loc["c1", "transaction_count"] == 2
    assert result.loc["c1", "total_amount"] == pytest.approx(50.0)
    assert result.loc["c1", "total_debit_amount"] == pytest.approx(100.0)
    assert result.loc["c1", "total_credit_amount"] == pytest.approx(50.0)
    assert result.loc["c2", "transaction_count"] == 2
    assert result.loc["c2", "total_amount"] == pytest.approx(400.0)
    assert result.loc["c2", "fraud_transaction_count"] == 1
    assert result.loc["c1", "fraud_transaction_count"] == 0


def test_engineer_customer_features_returns_one_row_per_customer(sample_transactions):
    result = engineer_customer_features(sample_transactions)

    assert len(result) == sample_transactions["CustomerId"].nunique()
    assert "CustomerId" in result.columns
    assert result.isna().sum().sum() == 0


def test_calculate_woe_iv_is_positive_for_a_discriminating_feature():
    df = pd.DataFrame(
        {
            "score": [1, 2, 3, 4, 100, 101, 102, 103],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    woe_table, iv = calculate_woe_iv(df, "score", "target", n_bins=2)

    assert iv > 0
    assert "woe" in woe_table.columns
    assert not woe_table["woe"].isna().any()


@pytest.fixture
def rfm_transactions() -> pd.DataFrame:
    """Synthetic transactions with an unambiguous engaged vs. disengaged split.

    engaged_1/engaged_2: many recent, moderate-to-high-value transactions.
    dormant_1/dormant_2: a single small transaction, long before the snapshot date.
    """
    common = {
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_2",
        "ProviderId": "ProviderId_1",
        "PricingStrategy": 2,
        "FraudResult": 0,
    }
    rows = []
    for customer in ["engaged_1", "engaged_2"]:
        for day in range(1, 21):
            rows.append(
                {
                    "CustomerId": customer,
                    "Amount": 500.0,
                    "Value": 500.0,
                    "TransactionStartTime": f"2019-02-{day:02d}T10:00:00Z",
                    **common,
                }
            )
    for customer in ["dormant_1", "dormant_2"]:
        rows.append(
            {
                "CustomerId": customer,
                "Amount": 20.0,
                "Value": 20.0,
                "TransactionStartTime": "2018-11-16T10:00:00Z",
                **common,
            }
        )
    return pd.DataFrame(rows)


def test_calculate_rfm_uses_fixed_snapshot_date_and_expected_values(rfm_transactions):
    snapshot = pd.Timestamp("2019-03-01", tz="UTC")
    rfm = calculate_rfm(rfm_transactions, snapshot_date=snapshot).set_index("CustomerId")

    assert rfm.loc["engaged_1", "Frequency"] == 20
    assert rfm.loc["dormant_1", "Frequency"] == 1
    assert rfm.loc["engaged_1", "Recency"] < rfm.loc["dormant_1", "Recency"]
    assert rfm.loc["engaged_1", "Monetary"] == pytest.approx(10_000.0)


def test_assign_high_risk_label_flags_the_dormant_customers(rfm_transactions):
    snapshot = pd.Timestamp("2019-03-01", tz="UTC")
    rfm = calculate_rfm(rfm_transactions, snapshot_date=snapshot)

    labeled = assign_high_risk_label(rfm, n_clusters=3, random_state=42).set_index("CustomerId")

    assert labeled.loc["dormant_1", "is_high_risk"] == 1
    assert labeled.loc["dormant_2", "is_high_risk"] == 1
    assert labeled.loc["engaged_1", "is_high_risk"] == 0
    assert labeled.loc["engaged_2", "is_high_risk"] == 0


def test_assign_high_risk_label_is_reproducible_with_fixed_random_state(rfm_transactions):
    rfm = calculate_rfm(rfm_transactions, snapshot_date=pd.Timestamp("2019-03-01", tz="UTC"))

    first = assign_high_risk_label(rfm, random_state=42)["is_high_risk"].tolist()
    second = assign_high_risk_label(rfm, random_state=42)["is_high_risk"].tolist()

    assert first == second


def test_build_processed_dataset_merges_is_high_risk_without_dropping_rows(rfm_transactions):
    processed = build_processed_dataset(rfm_transactions, snapshot_date=pd.Timestamp("2019-03-01", tz="UTC"))

    assert len(processed) == rfm_transactions["CustomerId"].nunique()
    assert "is_high_risk" in processed.columns
    assert processed["is_high_risk"].isna().sum() == 0
