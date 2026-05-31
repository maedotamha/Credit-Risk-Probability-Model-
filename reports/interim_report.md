# Building a Credit Risk Probability Model from Alternative Data: Interim Report

**Project:** Credit Risk Probability Model for Alternative Data  
**Client context:** Bati Bank buy-now-pay-later partnership  
**Interim scope:** Task 1 and Task 2  
**Submission date:** Sunday, 31 May 2026  
**Team:** Kerod, Mahbubah, Feven

## From eCommerce Behavior to Credit Decisions

Bati Bank's new buy-now-pay-later partnership creates a familiar credit-risk challenge with an unfamiliar data source. The bank needs to decide which customers can safely receive credit, but the available Xente dataset is not a traditional loan book. It is a transaction dataset: who transacted, what they bought, through which channel, when the transaction happened, and how much value moved.

That difference matters. A classic credit model learns from historical repayment behavior. This project does not yet have repayment outcomes, so the first modeling task is not simply to choose an algorithm. The first task is to define a defensible risk signal from customer behavior, document the assumptions behind it, and keep the model auditable enough for a banking environment.

This interim report covers two foundations:

- The business and regulatory understanding that guides the modeling choices.
- The exploratory data analysis that shapes the feature engineering plan.

## The Business Question

The core business question is:

> Can Bati Bank use eCommerce transaction behavior to estimate the probability that a customer is high risk for BNPL credit?

The expected product is a deployed model service that the loan origination team can call in real time. For every new applicant, the service should eventually return:

- A risk probability.
- A credit score derived from that probability.
- A recommended loan amount and duration.

At the interim stage, the focus is narrower: understand the data, understand the regulatory expectations, and establish a credible path toward a proxy target.

## Why Basel II Changes the Shape of the Project

Basel II pushes this project beyond a pure machine-learning exercise. In a bank, a credit model must be measurable, documented, explainable, and monitorable. This has direct implications for the way the project is designed.

First, every important modeling choice needs a record. If the project defines a high-risk customer using RFM clustering, that definition must be traceable: the snapshot date, the recency formula, the number of clusters, the scaling method, and the rule used to select the high-risk cluster all need to be documented.

Second, the project needs an interpretable benchmark. A Logistic Regression model, ideally supported by Weight of Evidence and Information Value analysis, gives the risk team a clear reference point. More complex models can still be tested, but they need to prove that their extra predictive lift is worth the extra governance burden.

Third, reproducibility is not optional. Random seeds, deterministic preprocessing, saved pipeline artifacts, and MLflow experiment tracking are not engineering decorations. They are part of the model control environment.

The practical modeling principle is simple: the best model is not only the one with the highest ROC-AUC. It is the one Bati Bank can explain, monitor, and defend.

## The Missing Label Problem

The dataset does not include an actual default label. There is no column showing whether a customer received a loan and failed to repay it within a specified period. That creates the central modeling risk in the project.

To train a supervised model, the project must create a proxy target. The proposed proxy is based on RFM behavior:

- **Recency:** customers who have not transacted recently may be less engaged.
- **Frequency:** customers with fewer transactions provide less behavioral evidence and may be less reliable.
- **Monetary value:** customers with low transaction value may represent lower engagement or lower economic activity on the platform.

The planned method is to aggregate customer-level RFM features, scale them, cluster customers into three groups with K-Means, and label the least engaged cluster as `is_high_risk = 1`.

This is a practical approach, but it must be treated carefully. A low-engagement customer is not automatically a future defaulter. The label is a business assumption designed to create a first risk model before repayment history exists. Once actual BNPL repayment data becomes available, the proxy should be validated, recalibrated, or replaced.

## Business Risks of Proxy-Based Prediction

The proxy target introduces several risks that should be visible to Bati Bank's leadership and risk team.

**Proxy risk:** the model may predict inactivity rather than repayment failure.

**Fairness risk:** some customers may transact less frequently for reasons unrelated to creditworthiness. A low-frequency customer could still be financially reliable.

**Product-launch risk:** behavior before BNPL availability may not perfectly represent behavior after customers are offered credit.

**Approval bias:** if early model decisions restrict access for a segment, the bank may never observe repayment outcomes for those rejected customers.

**Monitoring risk:** the RFM-risk relationship may drift as the eCommerce partner grows, pricing changes, or customer acquisition campaigns bring in new user types.

The report therefore recommends using the proxy model as an initial decision-support tool, not as an unchecked automated denial engine.

## Initial EDA Snapshot

The Xente dataset contains transaction-level records. Each row represents a single transaction and includes customer identifiers, account and subscription identifiers, product and provider information, transaction channel, transaction amount, timestamp, pricing strategy, and a binary fraud flag.

Publicly documented summaries of the Xente challenge dataset report:

- **95,662 transactions**
- **16 columns**
- **No missing values across the listed fields**
- Numeric fields including `CountryCode`, `Amount`, `Value`, `PricingStrategy`, and `FraudResult`
- Object/string fields including IDs, product details, channel, currency, and transaction timestamp
- A constant `CountryCode` of 256 in the available training data

Because the local repository did not include raw data at the time this report was written, these EDA observations should be rerun in `notebooks/eda.ipynb` once `data/raw` is populated. The important interim conclusion is that the dataset appears structurally suitable for feature engineering, but not directly suitable for supervised default modeling until the proxy target is created.

## What the EDA Tells Us So Far

### 1. The Dataset Is Rich, But Mostly Behavioral

The columns are useful for understanding activity patterns: customer identity, product category, provider, channel, transaction amount, transaction value, pricing strategy, and transaction time. These features can support strong behavioral engineering.

However, the dataset does not contain borrower-level income, employment, bureau history, collateral, repayment status, days past due, or write-off information. This confirms that the model should be framed as an alternative-data credit risk model, not a traditional bureau-style scorecard.

### 2. The Transaction Amount Distribution Is Highly Skewed

The reported summary statistics show that `Amount` and `Value` are heavily skewed. The median transaction value is much lower than the maximum value, which suggests that a small number of high-value transactions can dominate averages.

This affects modeling in two ways:

- Aggregates such as total amount and average amount should be complemented by robust features such as median, count, and standard deviation.
- Scaling, clipping, or log transformations should be considered before distance-based methods such as K-Means.

This is especially important for RFM clustering because monetary value can overpower recency and frequency if the features are not scaled.

### 3. Negative Amounts Need Business Interpretation

The `Amount` column can be positive or negative, while `Value` is the absolute value. In the challenge description, positive values represent debits from the customer and negative values represent credits.

This means the feature pipeline should not blindly treat all values as equivalent. Useful engineered features may include:

- Total debit amount.
- Total credit amount.
- Net transaction amount.
- Absolute transaction value.
- Count of credit-like transactions.
- Ratio of credits to debits.

For credit risk, a customer's transaction direction can be as important as transaction size.

### 4. Some Columns May Have Low Predictive Value

`CountryCode` appears constant in the public dataset summary, and `CurrencyCode` is likely also constant for this challenge context. Constant or near-constant columns should not be allowed to add noise to the model.

Identifiers such as `TransactionId`, `BatchId`, `AccountId`, and `SubscriptionId` should also be handled carefully. They may be useful for aggregation and traceability, but they should not be one-hot encoded as ordinary predictive categories because that can create leakage or high-cardinality noise.

### 5. FraudResult Is Not the Credit Default Target

The dataset includes `FraudResult`, but fraud is not the same as loan default. FraudResult can be explored as a risk-related signal, and it may help the bank understand suspicious transaction patterns. Still, it should not be renamed or reused as the default label.

This distinction is important for model governance. A fraud model answers "was this transaction fraudulent?" A credit model answers "how likely is this customer to fail to repay credit?" Those are related but different risk events.

## EDA Concerns and How the Project Addresses Them

**Concern: The dataset has no default label.**  
The project will define a proxy target using RFM clustering. The report will explicitly document that the proxy is not ground truth and must be validated when repayment data becomes available.

**Concern: The model must be interpretable under banking expectations.**  
The training phase will include an interpretable baseline such as Logistic Regression. WoE and IV will be considered for feature screening and scorecard-style explanation.

**Concern: Transaction values are skewed and contain outliers.**  
The feature pipeline will use scaling and may include robust transformations for monetary features. Outliers will be inspected rather than automatically removed, because high-value transactions may be legitimate business signals.

**Concern: Categorical variables and IDs may create leakage.**  
Operational identifiers will be used for grouping and joins, not as naive predictive features. Product, provider, category, channel, and pricing fields can be encoded after checking cardinality.

**Concern: RFM clustering can be unstable.**  
The clustering step will use a fixed `random_state`, scaled RFM inputs, and documented cluster profiling. The selected high-risk cluster will be justified by its recency, frequency, and monetary profile.

**Concern: The final model must be deployable.**  
The planned architecture separates exploration from production code. EDA stays in the notebook, while reusable transformations belong in `src/data_processing.py`, training in `src/train.py`, and serving in a FastAPI application.

## Feature Engineering Direction

The EDA points toward customer-level aggregation. The raw data is transaction-level, but credit decisions are made at the customer level. The next phase should therefore create features such as:

- Total transaction amount per customer.
- Average and median transaction amount.
- Transaction count.
- Standard deviation of transaction amount.
- Number of active days.
- Recency from a fixed snapshot date.
- Frequency over the observed period.
- Monetary value from absolute transaction value.
- Product category diversity.
- Channel diversity.
- Preferred channel.
- Debit and credit summaries.
- Time-based features such as transaction hour, day, month, and year.

These features will make the data more aligned with the credit scoring question.

## Proposed Modeling Path

The next project phase should follow this sequence:

1. Load and validate the raw Xente dataset.
2. Reproduce the EDA in `notebooks/eda.ipynb`.
3. Build a deterministic preprocessing pipeline in `src/data_processing.py`.
4. Aggregate RFM metrics by `CustomerId`.
5. Scale RFM values and cluster customers into three groups.
6. Label the least engaged cluster as `is_high_risk = 1`.
7. Train at least two models, including one interpretable baseline and one higher-performance model.
8. Track experiments with MLflow.
9. Package the best model behind a FastAPI `/predict` endpoint.
10. Add Docker and GitHub Actions for reproducibility and testing.

## Interim Conclusion

The early project work clarifies the most important point: this is not simply a fraud prediction problem and not yet a true default prediction problem. It is an alternative-data credit risk project where customer behavior must be transformed into a carefully documented proxy for credit risk.

The dataset appears suitable for behavioral feature engineering. Its transaction history can support RFM segmentation, customer-level aggregation, and model-ready features. The main limitation is the absence of true repayment outcomes, which makes transparency essential.

For Bati Bank, the right interim decision is to proceed with a proxy-based model, but to govern it as a first-generation risk signal. The model should support credit decisioning, not silently replace credit judgment. Its assumptions should be visible, its performance should be tracked, and its proxy target should be revisited as soon as real BNPL repayment data is collected.

## References

- Challenge brief: Credit Risk Probability Model for Alternative Data.
- Dataset: Xente Challenge transaction dataset.
- Basel II regulatory framing and credit risk model governance references from the challenge brief.
- Public Xente dataset summaries used only for interim EDA orientation until local raw data is added.

