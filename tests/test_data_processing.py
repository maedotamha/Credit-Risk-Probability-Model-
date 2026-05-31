from src.data_processing import missing_required_columns


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

