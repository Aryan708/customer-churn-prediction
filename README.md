# Customer Churn Prediction & Retention Targeting

End-to-end customer analytics project on the **IBM Telco Customer Churn** dataset (7,043 customers, 26.5% churn).
It goes from business question to a decision: **who should get a retention offer, and how would we prove it works?**

## What it does

| Step | What | Where |
|---|---|---|
| 1 | Load, clean and validate the data (nulls, duplicates, charge consistency) | `src/churn_analysis.py`, `sql/analysis.sql` |
| 2 | **SQL customer analytics**: churn by contract, **tenure cohort**, **segment**, payment method; lost monthly revenue (window function) | `sql/analysis.sql` |
| 3 | **Leakage check**: single-feature AUC screen | script §3 |
| 4 | **Interpretable baseline (logistic regression)** vs Random Forest vs Gradient Boosting, stratified 5-fold CV | script §4 |
| 5 | Held-out evaluation: ROC-AUC with **bootstrap 95% CI**, PR-AUC, **Brier score, calibration curve**, **lift by decile** | script §4–5 |
| 6 | Churn drivers: odds ratios + permutation importance | script §6 |
| 7 | **Retention targeting economics**: net value of targeting the top-k% riskiest customers vs random targeting | script §7 |
| 8 | **A/B test design** for the retention offer: hypothesis, guardrails, **sample size / power** | script §8 |

## Key results (test set, 1,409 customers)

- **Logistic regression chosen**: test ROC-AUC **0.842** (95% CI 0.819–0.862), CV 0.846 ± 0.013. Tree models were no better
  than it by more than 0.01 AUC, so the more interpretable model was kept.
- **Top-decile lift 2.8x**: the riskiest 10% churn at 75% vs a 26.5% base rate; the top 30% captures **65% of churners**.
- **Drivers**: short tenure, month-to-month contracts, fibre-optic internet, electronic-check payment and paperless billing raise churn.
  Two-year contracts cut churn to 2.8% (vs 42.7% month-to-month). Customers in their first 6 months churn at 53%.
- **Targeting**: under the stated offer assumptions, targeting the top 30% by model score yields about **3x the net value of
  targeting 30% at random**; net value peaks at roughly the top 50–60%.
- **A/B test**: to detect a 15% relative churn reduction in the top-30% group (57% → 49%) at α = 0.05 and 80% power needs
  **~526 customers per arm**.

Full tables: [`reports/results.md`](reports/results.md) · Figures: [`reports/figures/`](reports/figures)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/churn_analysis.py
```

Runs in about 20 seconds and regenerates `reports/results.md`, `reports/summary.json` and the figures.
Business assumptions for the targeting simulation (discount, save rate, value horizon) are constants at the top of the script.

## Data

`data/telco.csv`: IBM sample dataset "Telco Customer Churn" (public; also on Kaggle as *WA_Fn-UseC_-Telco-Customer-Churn*).

## Limitations

- One snapshot in time: there are no churn dates, so this is a classification problem rather than survival analysis.
- The targeting economics use illustrative assumptions; the real save rate must come from the A/B test.
- The model predicts *who* will churn, not *who can be persuaded to stay*. An uplift model trained on A/B test data would be the next step.
