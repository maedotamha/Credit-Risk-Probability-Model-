# Credit Risk Probability Model for Alternative Data

An end-to-end implementation plan for building, deploying, and automating a credit risk model for Bati Bank's buy-now-pay-later partnership.

## Project Context

Bati Bank needs a real-time credit scoring service for customers of an eCommerce partner. The source dataset is transaction-level Xente data, which contains rich behavioral signals such as transaction amount, product category, channel, provider, and timestamp, but does not contain a direct loan default label.

The project therefore treats credit risk modeling as a two-stage problem:

1. Build a defensible proxy target for high-risk customers from observed behavior.
2. Train and deploy a model that estimates risk probability for new applicants.

## Repository Structure

```text
credit-risk-model/
+-- .github/workflows/ci.yml
+-- data/
|   +-- raw/
|   +-- processed/
+-- notebooks/
|   +-- eda.ipynb
+-- reports/
|   +-- interim_report.md
+-- src/
+-- tests/
+-- .gitignore
+-- README.md
```

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The Basel II Accord makes credit risk modeling more than a prediction exercise. For Bati Bank, the model must support risk measurement, governance, documentation, and review. That means the project should not only output a probability of risk, but also explain how the estimate was produced, what data was used, what assumptions were made, and how model behavior will be monitored over time.

This affects the modeling choices in practical ways:

- The target definition must be documented because the dataset does not contain actual repayment outcomes.
- Feature engineering must be reproducible, preferably through versioned sklearn pipelines.
- The model comparison should include an interpretable baseline such as Logistic Regression, ideally with Weight of Evidence style transformations where appropriate.
- Higher-performing models such as Random Forest or Gradient Boosting should be evaluated with explainability and monitoring in mind.
- Model artifacts, parameters, metrics, and dataset versions should be tracked with MLflow so the risk team can audit decisions later.

In a regulated financial context, a model that cannot be explained is difficult to defend, even if it performs well on a leaderboard metric.

### 2. Why a Proxy Default Variable Is Necessary

The raw Xente transaction dataset does not include a true credit default outcome. It shows customer behavior on the eCommerce platform, not whether a customer later failed to repay a loan. Since supervised credit modeling requires a target variable, a proxy label is necessary to turn historical behavior into a training signal.

For this project, the proxy target will be based on RFM behavior:

- Recency: how recently the customer transacted.
- Frequency: how often the customer transacted.
- Monetary value: how much value the customer generated.

Customers with low engagement, low transaction frequency, and low monetary activity can be clustered and labeled as the high-risk proxy segment. This does not mean they truly defaulted. It means their behavioral pattern is treated as a warning signal until real loan repayment data becomes available.

The business risks are important:

- Proxy risk: the model may learn disengagement rather than true default.
- Exclusion risk: low-activity but creditworthy customers could be unfairly rejected or offered lower limits.
- Drift risk: behavior on an eCommerce platform may change when a BNPL product is introduced.
- Regulatory risk: proxy definitions must be transparent because they influence access to credit.
- Feedback-loop risk: if the model denies credit to some groups, future repayment data may become biased toward approved customers.

The proxy is therefore a starting assumption, not ground truth.

### 3. Interpretable Models vs High-Performance Models

A simple scorecard-style model such as Logistic Regression with WoE features is easier to explain, document, and challenge. It gives the bank a clearer view of how each variable changes risk and is often a strong baseline for regulated credit scoring.

However, behavioral transaction data can contain nonlinear patterns. A Gradient Boosting or Random Forest model may capture interactions between channel, product category, timing, transaction value, and customer frequency more effectively. The trade-off is that these models require stronger explainability tooling, tighter monitoring, and more careful governance.

The recommended approach is not to choose interpretability or performance blindly. The project should train both types of models, compare them with metrics such as precision, recall, F1, and ROC-AUC, and then decide whether the performance gain from a complex model is large enough to justify the governance cost.

## Status — complete

- **Task 1** — Business understanding: see above.
- **Task 2** — EDA: [notebooks/eda.ipynb](notebooks/eda.ipynb), executed against the real 95,662-row Xente dataset.
- **Task 3** — Feature engineering pipeline: [src/data_processing.py](src/data_processing.py) (`build_feature_pipeline`, a single `sklearn.Pipeline`).
- **Task 4** — RFM proxy target: `calculate_rfm` / `assign_high_risk_label` / `build_processed_dataset` in the same module.
- **Task 5** — Model training, tuning, and MLflow tracking: [src/train.py](src/train.py). XGBoost (ROC-AUC 0.851) beats Logistic Regression (ROC-AUC 0.800) and is registered as `credit-risk-is-high-risk` in the MLflow Model Registry.
- **Task 6** — Deployment: [src/api/main.py](src/api/main.py) (FastAPI `/predict`), `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.
- **Tests**: 16 passing unit tests across `tests/` (`pytest`), zero `flake8` findings.

### Reports

- Interim (Tasks 1-2): [reports/interim_report.md](reports/interim_report.md)
- Final (Medium-style, all tasks): [reports/final_report.md](reports/final_report.md)

### Running it

```bash
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m src.train          # trains, tunes, and registers the best model in MLflow
python -m uvicorn src.api.main:app --reload   # or: docker compose up --build
```
