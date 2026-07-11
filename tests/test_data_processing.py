import pandas as pd
import pytest

from src.data_processing import (
    CustomerAggregator,
    DatetimeFeatureExtractor,
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
