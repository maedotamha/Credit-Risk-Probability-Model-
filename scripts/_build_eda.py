import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import new_notebook, save

cells = []

cells.append(("md", """# Exploratory Data Analysis: Xente Transactions

Task 2 EDA for the Bati Bank credit risk project. This notebook explores the raw
Xente transaction dataset (`data/raw/data.csv`) to understand its structure, quality,
and the behavioral signals available for feature engineering and RFM-based proxy
target construction (Task 3 and Task 4).

This notebook is exploration only - reusable logic lives in `src/data_processing.py`."""))

cells.append(("code", """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path.cwd().parent))

pd.set_option("display.max_columns", 30)
plt.rcParams["figure.figsize"] = (10, 5)
sns.set_style("whitegrid")

DATA_PATH = Path("..") / "data" / "raw" / "data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["TransactionStartTime"])
df.shape
"""))

cells.append(("md", """## 1. Overview of the data"""))

cells.append(("code", """print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
df.info()
"""))

cells.append(("code", """df.head()
"""))

cells.append(("code", """print("Unique customers:", df['CustomerId'].nunique())
print("Unique accounts:", df['AccountId'].nunique())
print("Date range:", df['TransactionStartTime'].min(), "->", df['TransactionStartTime'].max())
print("Span (days):", (df['TransactionStartTime'].max() - df['TransactionStartTime'].min()).days)
"""))

cells.append(("md", """## 2. Summary statistics"""))

cells.append(("code", """df.describe(include=[np.number]).T
"""))

cells.append(("code", """df.describe(include=["object"]).T
"""))

cells.append(("md", """## 3. Distribution of numerical features

`Amount` can be positive (debit) or negative (credit); `Value` is its absolute value.
`CountryCode` is investigated separately since a constant column carries no signal."""))

cells.append(("code", """print("CountryCode unique values:", df['CountryCode'].unique())
print("CurrencyCode unique values:", df['CurrencyCode'].unique())
print("PricingStrategy unique values:", sorted(df['PricingStrategy'].unique()))
"""))

cells.append(("code", """fig, axes = plt.subplots(2, 2, figsize=(14, 9))

sns.histplot(df['Value'], bins=80, ax=axes[0, 0])
axes[0, 0].set_title("Value - full range")

sns.histplot(df[df['Value'] < df['Value'].quantile(0.99)]['Value'], bins=80, ax=axes[0, 1])
axes[0, 1].set_title("Value - below 99th percentile")

sns.histplot(df['Amount'], bins=80, ax=axes[1, 0])
axes[1, 0].set_title("Amount - full range (signed)")

sns.histplot(np.log1p(df['Value']), bins=80, ax=axes[1, 1])
axes[1, 1].set_title("log1p(Value)")

plt.tight_layout()
plt.savefig("../reports/figures/eda_numerical_distributions.png", dpi=120)
plt.show()
"""))

cells.append(("code", """skew_kurt = pd.DataFrame({
    "skew": [df['Amount'].skew(), df['Value'].skew()],
    "kurtosis": [df['Amount'].kurtosis(), df['Value'].kurtosis()],
}, index=["Amount", "Value"])
skew_kurt
"""))

cells.append(("code", """print("Share of debit transactions (Amount > 0):", (df['Amount'] > 0).mean().round(4))
print("Share of credit transactions (Amount < 0):", (df['Amount'] < 0).mean().round(4))
print("Share of zero-amount transactions:", (df['Amount'] == 0).mean().round(4))
"""))

cells.append(("md", """## 4. Distribution of categorical features"""))

cells.append(("code", """cat_cols = ["ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), cat_cols):
    order = df[col].astype(str).value_counts().index
    sns.countplot(y=df[col].astype(str), order=order, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("../reports/figures/eda_categorical_distributions.png", dpi=120)
plt.show()
"""))

cells.append(("code", """print("ProductCategory value counts:")
print(df['ProductCategory'].value_counts())
print()
print("ChannelId value counts:")
print(df['ChannelId'].value_counts())
"""))

cells.append(("code", """print("FraudResult distribution:")
print(df['FraudResult'].value_counts())
print("Fraud rate:", df['FraudResult'].mean().round(5))
"""))

cells.append(("md", """## 5. Correlation analysis (numerical features)"""))

cells.append(("code", """num_cols = ["Amount", "Value", "PricingStrategy", "FraudResult"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation - numerical features")
plt.tight_layout()
plt.savefig("../reports/figures/eda_correlation_heatmap.png", dpi=120)
plt.show()
"""))

cells.append(("md", """`Value` and `Amount` are near-perfectly correlated (Value is the absolute value of
Amount), which is expected and means one is largely redundant once transaction
direction is captured separately (e.g. via a debit/credit indicator)."""))

cells.append(("md", """## 6. Missing values"""))

cells.append(("code", """missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(3)
pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
"""))

cells.append(("md", """## 7. Outlier detection"""))

cells.append(("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df['Value'], ax=axes[0])
axes[0].set_title("Value - boxplot (full range)")
sns.boxplot(y=df['Amount'], ax=axes[1])
axes[1].set_title("Amount - boxplot (full range)")
plt.tight_layout()
plt.savefig("../reports/figures/eda_outlier_boxplots.png", dpi=120)
plt.show()
"""))

cells.append(("code", """q1, q3 = df['Value'].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
outliers = df[df['Value'] > upper_fence]
print(f"IQR outlier fence (Value): {upper_fence:,.2f}")
print(f"Transactions above fence: {len(outliers):,} ({len(outliers)/len(df):.2%})")
print(f"Unique customers affected: {outliers['CustomerId'].nunique():,}")
outliers['Value'].describe()
"""))

cells.append(("md", """## 8. Customer-level transaction counts

Credit decisions are made at the customer level, but the raw data is transaction-level.
This checks how much history is typically available per customer, which matters for
RFM reliability (Task 4)."""))

cells.append(("code", """txn_per_customer = df.groupby('CustomerId').size()
txn_per_customer.describe()
"""))

cells.append(("code", """fig, ax = plt.subplots()
sns.histplot(txn_per_customer[txn_per_customer < txn_per_customer.quantile(0.99)], bins=50, ax=ax)
ax.set_title("Transactions per customer (below 99th percentile)")
ax.set_xlabel("Transaction count")
plt.tight_layout()
plt.savefig("../reports/figures/eda_transactions_per_customer.png", dpi=120)
plt.show()
print("Customers with only 1 transaction:", (txn_per_customer == 1).sum(),
      f"({(txn_per_customer == 1).mean():.1%})")
"""))

cells.append(("md", """## Top insights

1. **The dataset is transaction-level (95,662 rows) and customer-level credit
   decisions require aggregation to only 3,742 unique customers** (mean 25.6
   transactions/customer, but median just 7 and 19.0% of customers (712) have only
   a single transaction). Frequency/monetary signal is unreliable for the ~1-in-5
   single-transaction customers, which is a direct risk to the RFM-based proxy
   target in Task 4: sparse history should not be conflated with genuine
   disengagement.
2. **`Amount`/`Value` are extremely right-skewed with heavy outliers** (skew ~51,
   kurtosis ~3,363 for both). 9.43% of transactions (9,021, affecting 1,157
   customers) sit above the IQR outlier fence. Aggregates like mean/total will be
   pulled by a few large transactions, so median, count, and standard deviation
   should accompany any total/average feature, and RFM's monetary component should
   be scaled (not raw) before K-Means clustering.
3. **`CountryCode` (always 256) and `CurrencyCode` (always UGX) are constant columns**
   and carry zero predictive signal for this dataset slice - they should be dropped
   from modeling features rather than one-hot encoded.
4. **`FraudResult` is extremely imbalanced (0.202% positive, 193 of 95,662) and is a
   distinct risk event from credit default.** It is not usable as, or a substitute
   for, the credit risk target; it may only be useful as an auxiliary behavioral
   feature.
5. **`Value` and `Amount` are almost perfectly correlated** (Value = abs(Amount)), so
   using both as independent numerical features would be redundant - a debit/credit
   direction indicator (60.1% debits, 39.9% credits, 0% zero-amount) plus `Value`
   captures the same information more efficiently.
6. **No missing values were found in any column** of the raw dataset, which
   simplifies Task 3: the pipeline still needs an imputer for robustness against
   future data drift, but no rows/columns need to be dropped for missingness here.
7. **Two product categories (`financial_services`, `airtime`) account for ~94.5% of
   all transactions**, and two channels (`ChannelId_2`, `ChannelId_3`) account for
   ~98.4% of transactions - category/channel diversity per customer is a plausible
   engineered feature, but rare categories will need care (e.g. grouping) to avoid
   sparse one-hot columns."""))

nb = new_notebook(cells)
save(nb, str(Path(__file__).resolve().parents[1] / "notebooks" / "eda.ipynb"))
print("wrote notebook, cells:", len(cells))
