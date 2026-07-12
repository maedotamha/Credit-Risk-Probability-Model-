"""Render reports/final_report.md into a polished, blog-styled PDF.

Run from the repo root:
    .venv/Scripts/python.exe scripts/build_final_report_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]

NAVY = colors.HexColor("#1F3864")
ACCENT = colors.HexColor("#C0392B")
INK = colors.HexColor("#222222")
MUTED = colors.HexColor("#5A5A5A")
CODE_BG = colors.HexColor("#F4F4F4")
LIGHT_GRAY = colors.HexColor("#F5F6F8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("HeroTitle", fontSize=24, leading=29, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceAfter=6))
styles.add(ParagraphStyle("Byline", fontSize=11.5, leading=16, textColor=MUTED,
                           fontName="Helvetica-Oblique", spaceAfter=18))
styles.add(ParagraphStyle("H2", fontSize=15, leading=19, textColor=NAVY,
                           fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H3", fontSize=11.5, leading=15, textColor=ACCENT,
                           fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", fontSize=10.3, leading=15.5, textColor=INK, spaceAfter=9,
                           alignment=4))  # justified
styles.add(ParagraphStyle("BulletItem", fontSize=10.3, leading=15, textColor=INK,
                           leftIndent=16, spaceAfter=4))
styles.add(ParagraphStyle("CodeBlock", fontName="Courier", fontSize=8.3, leading=11,
                           backColor=CODE_BG, borderPadding=8, textColor=colors.HexColor("#111111")))
styles.add(ParagraphStyle("Footer", fontSize=9.5, leading=14, textColor=MUTED,
                           fontName="Helvetica-Oblique", spaceBefore=16))


def p(text: str, style: str = "Body"):
    return Paragraph(text, styles[style])


def code_block(text: str):
    return Preformatted(text, styles["CodeBlock"])


def model_table():
    data = [
        ["Model", "Tuning", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["Logistic Regression", "GridSearchCV", "70.2%", "52.4%", "56.2%", "54.2%", "0.800"],
        ["XGBoost (best)", "RandomizedSearchCV", "75.7%", "61.8%", "59.1%", "60.4%", "0.851"],
    ]
    t = Table(data, colWidths=[1.25 * inch, 1.35 * inch, 0.68 * inch, 0.72 * inch, 0.62 * inch, 0.5 * inch, 0.65 * inch])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FCEEEC")),
    ]
    t.setStyle(TableStyle(style))
    return t


def build():
    doc = SimpleDocTemplate(
        str(ROOT / "reports" / "final_report.pdf"),
        pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.8 * inch, bottomMargin=0.75 * inch,
        title="Teaching a Bank to Score Risk It Has Never Seen",
        author="Bati Bank Credit Risk Project",
    )

    story = []
    story.append(p("Teaching a Bank to Score Risk It Has Never Seen: "
                    "A Credit Model Built on Shopping Behavior, Not Credit History", "HeroTitle"))
    story.append(p(
        "How Bati Bank's buy-now-pay-later partnership turns eCommerce transaction data into a "
        "real-time credit risk API &mdash; and why the target variable itself is the most "
        "important design decision in the whole project.", "Byline"))
    story.append(HRFlowable(width="100%", thickness=0.75, color=NAVY, spaceAfter=14))

    story.append(p("The problem: a credit model with no default label", "H2"))
    story.append(p(
        "Bati Bank is partnering with an eCommerce platform to offer buy-now-pay-later (BNPL) "
        "credit. The bank's job is the same one it's always had &mdash; decide who gets credit, "
        "and how much &mdash; but the data is unfamiliar. Instead of a loan book with repayment "
        "history, Bati Bank has something else entirely: 95,662 raw transaction records from the "
        "Xente platform, covering 3,742 customers over a 90-day window (mid-November 2018 to "
        "mid-February 2019). Every row shows what a customer bought, through which channel, for "
        "how much, and when. None of it says whether that customer ever failed to repay a loan "
        "&mdash; because none of these customers has taken a loan yet. The BNPL product doesn't "
        "exist for them yet either."))
    story.append(p(
        "This is the central fact that shapes every decision in this project: <b>there is no "
        "default label to learn from.</b> Before any model gets trained, the team has to answer "
        "a harder question first &mdash; what does “risk” even mean here, and how do "
        "you measure it from shopping behavior alone?"))

    story.append(p("Why this isn't just a machine learning problem", "H2"))
    story.append(p(
        "The Basel II Capital Accord is the reason a bank can't just throw the data at a "
        "gradient boosting model and ship whatever comes out. Basel II's emphasis on risk "
        "measurement means a credit model has to be <i>explainable</i>, not just accurate "
        "&mdash; the bank needs to be able to say why a customer was scored the way they were, "
        "document every assumption behind the target variable, and monitor the model's behavior "
        "over time. A black-box score with no audit trail is not defensible in a regulated "
        "lending context, no matter how good its ROC-AUC looks on a held-out test set."))
    story.append(p("That requirement runs through the whole build:"))
    for item in [
        "The target variable had to be constructed, not found &mdash; and that construction has "
        "to be documented and defensible, not just “whatever K-Means produced.”",
        "The feature pipeline had to be deterministic and reproducible (fixed random seeds "
        "everywhere: clustering, train/test splits, hyperparameter search).",
        "Model comparison had to include an interpretable baseline (Logistic Regression), not "
        "just the best-performing model.",
        "Every experiment had to be tracked &mdash; parameters, metrics, and artifacts &mdash; "
        "so the choice of final model has a paper trail.",
    ]:
        story.append(p(f"&bull; {item}", "BulletItem"))

    story.append(p("Building the proxy target: RFM segmentation as a stand-in for default", "H2"))
    story.append(p(
        "Since there's no default label, the project builds a <b>proxy</b>: a behavioral signal "
        "that stands in for credit risk until real repayment data exists. The method is RFM "
        "analysis &mdash; Recency, Frequency, Monetary value &mdash; a classic "
        "customer-segmentation technique repurposed here as a risk signal."))
    story.append(p("For each customer:"))
    for item in [
        "<b>Recency</b>: days since their last transaction, measured from a snapshot date one "
        "day after the last transaction in the dataset (2019-02-14).",
        "<b>Frequency</b>: total number of transactions.",
        "<b>Monetary</b>: total transaction value.",
    ]:
        story.append(p(f"&bull; {item}", "BulletItem"))
    story.append(p(
        "These three numbers were scaled and fed into K-Means with k=3 and a fixed "
        "random_state=42 for reproducibility. The intent: separate customers into an "
        "actively-engaged segment, a moderately-engaged segment, and a disengaged segment, then "
        "treat the disengaged segment as the high-risk proxy &mdash; the assumption being that a "
        "customer who has stopped transacting, transacts rarely, and spends little is a customer "
        "the bank knows the least about and should be most cautious with."))
    story.append(p(
        "<b>The first version of this clustering was wrong, and it's worth explaining why.</b> "
        "Monetary value in this dataset is extremely skewed &mdash; a small number of customers "
        "each generated tens of millions of shillings in transaction value, dwarfing the typical "
        "customer's total. Clustering on the raw scaled values let those few outliers dominate "
        "the distance metric: K-Means produced one cluster of 4 extreme-value customers and "
        "lumped everyone else into two much larger, less meaningful groups. The fix was a "
        "log1p transform on Frequency and Monetary before scaling &mdash; standard practice for "
        "heavy-tailed variables, and one the Task 1 EDA had already flagged as necessary. After "
        "the fix, K-Means produced three genuinely balanced segments (1,173 / 1,202 / 1,367 "
        "customers), and the cluster boundaries lined up with intuition: the smallest, most "
        "disengaged group had a mean recency of 62 days versus 13 days for the most engaged "
        "group."))
    story.append(p(
        "The high-risk cluster itself isn't picked by eyeballing the numbers &mdash; it's "
        "selected automatically, by z-scoring each cluster's mean Recency, Frequency, and "
        "Monetary value and picking the cluster with the worst composite score (highest recency, "
        "lowest frequency, lowest monetary). That cluster became is_high_risk = 1 for "
        "<b>1,173 customers (31.3% of the base)</b>."))

    story.append(p("The honest caveat", "H3"))
    story.append(p(
        "This proxy is a modeling assumption, not measured truth, and it carries real business "
        "risk:"))
    for item in [
        "<b>Proxy risk</b>: the model learns “disengagement,” not “default.” "
        "A customer who stopped shopping on this platform because they switched to a competitor "
        "is not the same as a customer who would fail to repay a loan.",
        "<b>Exclusion risk</b>: a low-activity but genuinely creditworthy customer could be "
        "scored as high-risk and denied credit or offered worse terms, purely because they don't "
        "shop often &mdash; not because they're a bad credit risk.",
        "<b>Feedback-loop risk</b>: once the bank starts approving or denying credit based on "
        "this model, it stops observing outcomes for the customers it denies. Future retraining "
        "data will be biased toward the customers the model already approved.",
        "<b>Product-launch risk</b>: behavior on the platform <i>before</i> BNPL exists may not "
        "represent behavior <i>after</i> customers know credit is on offer.",
    ]:
        story.append(p(f"&bull; {item}", "BulletItem"))
    story.append(p(
        "None of this is a reason not to build the model &mdash; it's a reason to treat it as a "
        "first-generation decision-support signal, not an unchecked auto-approval engine, and to "
        "revisit the proxy the moment real repayment data exists."))

    story.append(p("From transactions to features: an engineering decision that matters more than it looks", "H2"))
    story.append(p(
        "Credit decisions happen per customer, but the raw data is per transaction. The feature "
        "pipeline (src/data_processing.py, expressed as a single sklearn.Pipeline) rolls 95,662 "
        "transaction rows up into 3,742 customer-level records, computing things like "
        "total/average/median transaction amount, transaction count, debit/credit split, "
        "category and channel diversity, active days, and time-of-day patterns &mdash; 45 "
        "columns in total after encoding and scaling."))
    story.append(p(
        "One design decision is worth calling out explicitly, because it's easy to get wrong: "
        "<b>the classifier is never trained on Recency, Frequency, or Monetary directly.</b> "
        "Those three numbers are exactly what the K-Means proxy label was built from &mdash; a "
        "correlation check confirmed Recency alone correlates with is_high_risk at <b>0.85</b>. "
        "Feeding them into the classifier wouldn't build a predictive model; it would just teach "
        "the model to reverse-engineer which side of the clustering boundary a customer fell on, "
        "which is both circular and useless in production, since a brand-new applicant has no "
        "comparable RFM history to re-cluster against anyway. The classifier instead learns from "
        "the <i>other</i> engineered features &mdash; transaction count, active months, category "
        "and channel diversity, and so on &mdash; which still correlate meaningfully with the "
        "proxy label (0.2&ndash;0.4 range) without being definitionally identical to it."))

    story.append(p("Comparing two models, honestly", "H2"))
    story.append(p(
        "Two classifiers were trained and tuned on a stratified 80/20 train/test split "
        "(random_state=42 throughout):"))
    story.append(Spacer(1, 6))
    story.append(model_table())
    story.append(Spacer(1, 10))
    story.append(p(
        "Every run &mdash; parameters, metrics, and the model artifact itself &mdash; was "
        "logged to MLflow. XGBoost wins on every metric and was registered as "
        "credit-risk-is-high-risk (version 1) in the MLflow Model Registry."))
    story.append(p(
        "That said, the interpretability-versus-performance trade-off from the Task 1 business "
        "understanding is real here, not theoretical. Logistic Regression's coefficients can be "
        "read directly by a risk analyst; XGBoost's decision boundary can't be inspected the "
        "same way without additional tooling (SHAP values, partial dependence plots) that this "
        "phase of the project didn't build out. XGBoost's ~5-point ROC-AUC advantage is "
        "meaningful, but a bank moving this into production would need to weigh that gain "
        "against the added governance cost of explaining a boosted-tree model's decisions to a "
        "credit committee &mdash; exactly the trade-off Basel II's documentation expectations "
        "are designed to force into the open, rather than letting the highest leaderboard number "
        "win by default."))

    story.append(p("Shipping it: an API that takes raw transactions, not pre-computed features", "H2"))
    story.append(p(
        "The registered MLflow model isn't just the XGBoost classifier &mdash; it's the "
        "<i>entire</i> serving pipeline: the fitted feature-engineering steps plus the trained "
        "classifier, bundled into one object. This matters operationally. It means the FastAPI "
        "service (src/api/main.py) can accept the same raw transaction data the eCommerce "
        "platform already produces, and the model handles feature engineering internally "
        "&mdash; no risk of the API's feature computation drifting out of sync with what the "
        "model was actually trained on."))
    story.append(p("A real request against the running service, scoring two customers from held-out transaction history:"))
    story.append(code_block(
        'curl -X POST http://localhost:8000/predict \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"transactions": [ ...raw transaction rows for CustomerId_4406 '
        'and CustomerId_988... ]}\''
    ))
    story.append(Spacer(1, 6))
    story.append(code_block(
        '{\n'
        '  "predictions": [\n'
        '    {"CustomerId": "CustomerId_4406", "risk_probability": 0.000770587648730725},\n'
        '    {"CustomerId": "CustomerId_988",  "risk_probability": 0.0004339662555139512}\n'
        '  ]\n'
        '}'
    ))
    story.append(Spacer(1, 8))
    story.append(p(
        "Both customers here score very low risk &mdash; consistent with them being active, "
        "frequent transactors in the underlying data. The /predict endpoint accepts multiple "
        "transactions per customer (required, since a single transaction can't capture the "
        "behavioral pattern the model was trained on) and returns one probability per unique "
        "CustomerId. A /health endpoint reports service status independently of whether a model "
        "is currently loaded, so the container can be monitored even if model loading fails "
        "&mdash; in which case /predict returns a clear 503 rather than the process crashing."))
    story.append(p(
        "The whole thing is packaged behind a Dockerfile and docker-compose.yml, with a GitHub "
        "Actions workflow (.github/workflows/ci.yml) that runs flake8 and the full pytest suite "
        "&mdash; 16 tests spanning feature engineering, RFM clustering and reproducibility, "
        "training helpers, and the API itself (using a mocked model, so CI doesn't need a "
        "trained model artifact to verify the service's request/response contract) &mdash; on "
        "every push and pull request against main."))

    story.append(p("What this project is, and isn't", "H2"))
    story.append(p(
        "This is a working, end-to-end credit risk scoring service: real data in, a documented "
        "and reproducible proxy target, two compared and tracked models, and a deployable API "
        "that returns a risk probability for a real customer in real time. That's a genuine "
        "capability Bati Bank didn't have before."))
    story.append(p(
        "What it isn't, yet, is a validated default model. The is_high_risk label is a "
        "considered assumption about what disengagement implies about creditworthiness, not a "
        "measurement of it. The honest next step &mdash; the one this report keeps returning to "
        "&mdash; is closing the loop: once the BNPL product launches and real repayment outcomes "
        "start accumulating, those outcomes need to replace the RFM proxy as the training "
        "target, and the model built here should be judged against how well its original "
        "assumption held up, not treated as a finished product."))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceBefore=14, spaceAfter=8))
    story.append(p(
        "Code, tests, and full experiment history: "
        "github.com/maedotamha/Credit-Risk-Probability-Model-", "Footer"))

    doc.build(story)
    print(f"Wrote {ROOT / 'reports' / 'final_report.pdf'}")


if __name__ == "__main__":
    build()
