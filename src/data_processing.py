"""Data processing utilities for the credit risk project.

Production feature engineering will be implemented in Task 3.
"""

from __future__ import annotations

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


def missing_required_columns(columns: list[str] | set[str]) -> set[str]:
    """Return required Xente columns that are absent from an input dataset."""
    return REQUIRED_TRANSACTION_COLUMNS.difference(set(columns))

