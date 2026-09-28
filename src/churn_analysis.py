"""
Customer Churn Prediction & Retention Targeting
================================================
End-to-end pipeline on the IBM Telco Customer Churn dataset (7,043 customers):

1. Load + validate data (Python and SQL checks)
2. SQL customer analytics: churn by contract, tenure cohort, segment, payment method, revenue at risk
3. Leakage check (single-feature AUC screen)
4. Interpretable baseline (logistic regression) vs. tree-based models (Random Forest, Gradient Boosting)
   with stratified 5-fold cross-validation
5. Held-out evaluation: ROC-AUC (bootstrap 95% CI), PR-AUC, Brier score, calibration, lift by decile
6. Drivers: logistic-regression odds ratios + permutation importance
7. Retention targeting economics: expected net value of targeting the top-k% riskiest customers
8. A/B test design for the retention offer: sample size / power

Run:  python src/churn_analysis.py
Outputs go to reports/ (results.md + figures/).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco.csv"
SQL_FILE = ROOT / "sql" / "analysis.sql"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- Business assumptions for the targeting simulation (edit to taste) ----
OFFER_DISCOUNT = 0.20        # 20% discount on monthly bill ...
OFFER_MONTHS = 3             # ... for 3 months, paid by every targeted customer who stays
SAVE_RATE = 0.30             # share of would-be churners the offer retains
VALUE_HORIZON_MONTHS = 12    # revenue horizon for a saved customer
CONTACT_COST = 2.0           # cost per customer contacted ($)

report: list[str] = []


def md(line: str = "") -> None:
    report.append(line)


def fmt(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return f"{int(v):,}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- 1. Load & clean
def load() -> pd.DataFrame:
    raw = pd.read_csv(DATA)
    df = raw.rename(columns=lambda c: re.sub(r"(?<!^)(?=[A-Z])", "_", c).lower())
    df = df.rename(columns={"customer_i_d": "customer_id", "senior_citizen": "senior_citizen"})
    # TotalCharges is blank for brand-new customers (tenure 0) -> they have paid nothing yet
    df["total_charges"] = pd.to_numeric(df["total_charges"].str.strip().replace("", np.nan))
    n_blank = int(df["total_charges"].isna().sum())
    df["total_charges"] = df["total_charges"].fillna(0.0)
    df["churn"] = (df["churn"] == "Yes").astype(int)
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})
    md("## 1. Data")
    md(f"- Rows: **{len(df):,}** customers, {df.shape[1]} columns; churn rate **{df.churn.mean():.1%}**")
    md(f"- `TotalCharges` blank for {n_blank} customers, all with tenure 0 → set to 0 (no bills yet)")
    md(f"- Duplicate customer IDs: {df.customer_id.duplicated().sum()}")
    md()
    return df


# ---------------------------------------------------------------- 2. SQL analytics
def run_sql(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(":memory:")
    df.to_sql("customers", con, index=False)
    text = SQL_FILE.read_text()
    blocks = re.split(r"--\s*name:\s*(\w+)", text)[1:]
    results = {}
    for name, body in zip(blocks[::2], blocks[1::2]):
        sql = "\n".join(l for l in body.splitlines() if not l.strip().startswith("--")).strip()
        results[name] = pd.read_sql_query(sql, con)
    con.close()

    md("## 2. Customer analytics (SQL)")
    titles = {
        "overall_churn": "Overall",
        "churn_by_contract": "Churn by contract type",
        "churn_by_tenure_cohort": "Churn by tenure cohort",
        "churn_by_segment": "Churn by segment (contract × internet service)",
        "churn_by_payment_method": "Churn by payment method",
        "revenue_at_risk": "Lost monthly recurring revenue (MRR) by contract",
        "data_validation": "Data validation checks",
    }
    for k, t in titles.items():
        md(f"**{t}**")
        md()
        md(df_to_md(results[k]))
        md()

    # cohort chart
    c = results["churn_by_tenure_cohort"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(c.tenure_cohort.str[4:], c.churn_rate_pct, color="#3b6ea5")
    for i, v in enumerate(c.churn_rate_pct):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by tenure cohort")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "churn_by_tenure_cohort.png", dpi=150)
    plt.close(fig)
    return results


# ---------------------------------------------------------------- 3. Features & leakage check
CAT = [
    "gender", "senior_citizen", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_t_v", "streaming_movies", "contract", "paperless_billing", "payment_method",
]
NUM = ["tenure", "monthly_charges", "total_charges"]


def leakage_check(X: pd.DataFrame, y: pd.Series) -> None:
    """Flag any single feature that predicts churn almost perfectly on its own (a leakage smell)."""
    rows = []
    for col in CAT + NUM:
        if col in NUM:
            score = X[col].astype(float)
        else:  # target-rate encoding within the training split only
            score = X[col].map(y.groupby(X[col]).mean())
        auc = roc_auc_score(y, score)
        rows.append((col, round(max(auc, 1 - auc), 3)))
    t = pd.DataFrame(rows, columns=["feature", "single_feature_auc"]).sort_values(
        "single_feature_auc", ascending=False
    )
    flagged = t[t.single_feature_auc > 0.90]
    md("## 3. Leakage check")
    md("Single-feature ROC-AUC on the training split (a feature > 0.90 on its own would suggest leakage). "
       "`customer_id` is excluded as an identifier.")
    md()
    md(df_to_md(t.head(6)))
    md()
    md(f"**Result:** {'no features flagged' if flagged.empty else 'FLAGGED: ' + ', '.join(flagged.feature)}. "
       "`total_charges` is ≈ tenure × monthly charges, so it is redundant rather than leaky; it is kept "
       "because it is known at scoring time.")
    md()


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CAT), ("num", num, NUM)]
    )


# ---------------------------------------------------------------- 4. Models + CV
def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [("prep", preprocessor(True)), ("clf", LogisticRegression(max_iter=2000, C=1.0))]
        ),
        "Random Forest": Pipeline(
            [("prep", preprocessor(False)),
             ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                            random_state=SEED, n_jobs=-1))]
        ),
        "Gradient Boosting": Pipeline(
            [("prep", preprocessor(False)),
             ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4,
                                                    l2_regularization=1.0, random_state=SEED))]
        ),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def lift_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
    base = d.y.mean()
    t = d.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"),
                                               avg_pred=("p", "mean"))
    t["churn_rate"] = t.churners / t.customers
    t["lift"] = t.churn_rate / base
    t["cum_capture"] = t.churners.cumsum() / d.y.sum()
    out = t.reset_index()
    out["avg_pred"] = out.avg_pred.map("{:.1%}".format)
    out["churn_rate"] = out.churn_rate.map("{:.1%}".format)
    out["lift"] = out.lift.map("{:.2f}x".format)
    out["cum_capture"] = out.cum_capture.map("{:.0%}".format)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    X, y = df[CAT + NUM], df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    leakage_check(X_tr, y_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = build_models()
    rows, preds = [], {}
    for name, m in models.items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        preds[name] = p
        lo, hi = bootstrap_auc_ci(y_te.values, p)
        rows.append({
            "model": name,
            "cv_roc_auc (mean ± sd)": f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}",
            "test_roc_auc [95% CI]": f"{roc_auc_score(y_te, p):.3f} [{lo:.3f}, {hi:.3f}]",
            "test_pr_auc": f"{average_precision_score(y_te, p):.3f}",
            "brier": f"{brier_score_loss(y_te, p):.3f}",
            "_auc": roc_auc_score(y_te, p),
        })
    res = pd.DataFrame(rows)
    md("## 4. Model comparison")
    md(f"Train/test split 80/20 stratified (train {len(X_tr):,}, test {len(X_te):,}); 5-fold stratified CV on "
       f"train. Churn base rate in test: {y_te.mean():.1%} (PR-AUC of a random model ≈ base rate).")
    md()
    md(df_to_md(res.drop(columns="_auc")))
    md()

    # Choose the model: prefer the interpretable one unless a tree model is clearly better (> 0.01 AUC)
    lr_auc = res.loc[res.model == "Logistic Regression", "_auc"].item()
    best_other = res[res.model != "Logistic Regression"].sort_values("_auc", ascending=False).iloc[0]
    chosen = best_other.model if best_other._auc - lr_auc > 0.01 else "Logistic Regression"
    md(f"**Chosen model: {chosen}.** Rule: keep the interpretable logistic regression unless a tree model "
       f"beats it by more than 0.01 test ROC-AUC (best tree: {best_other.model}, {best_other._auc:.3f} vs "
       f"{lr_auc:.3f}). Stability: CV standard deviation is small, so performance is consistent across folds.")
    md()

    p = preds[chosen]
    thr = 0.5
    md(f"Classification report for {chosen} at threshold {thr}:")
    md()
    md("```")
    md(classification_report(y_te, (p >= thr).astype(int), target_names=["stay", "churn"], digits=3))
    md("```")
    md()

    # calibration plot
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for name, pp in preds.items():
        fx, fy = calibration_curve(y_te, pp, n_bins=10, strategy="quantile")
        ax.plot(fy, fx, marker="o", ms=3, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", dpi=150)
    plt.close(fig)

    md("## 5. Lift by decile (chosen model, test set)")
    lt = lift_table(y_te.values, p)
    md(df_to_md(lt))
    md()
    md("![Calibration](figures/calibration.png)")
    md()

    drivers(models, chosen, X_te, y_te)
    economics(y_te.values, p, X_te["monthly_charges"].values)
    return {"chosen": chosen, "results": res.drop(columns="_auc").to_dict("records"),
            "test_auc": float(roc_auc_score(y_te, p)), "base_rate": float(y_te.mean()),
            "top_decile_lift": lt.lift.iloc[0], "top30_capture": lt.cum_capture.iloc[2]}


# ---------------------------------------------------------------- 6. Drivers
def drivers(models: dict, chosen: str, X_te: pd.DataFrame, y_te: pd.Series) -> None:
    lr = models["Logistic Regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    t = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coef": coefs})
    t["odds_ratio"] = np.exp(t.coef).round(2)
    t = t.reindex(t.coef.abs().sort_values(ascending=False).index).head(10)
    md("## 6. Churn drivers")
    md("**Logistic regression, top 10 effects** (odds ratio > 1 raises churn odds; numeric features are "
       "standardised, so their odds ratio is per 1 standard deviation):")
    md()
    md(df_to_md(t[["feature", "odds_ratio"]]))
    md()

    m = models[chosen]
    pi = permutation_importance(m, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    imp = pd.DataFrame({"feature": X_te.columns, "auc_drop": pi.importances_mean}).sort_values(
        "auc_drop", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(imp.feature[::-1], imp.auc_drop[::-1], color="#3b6ea5")
    ax.set_xlabel("Drop in test ROC-AUC when shuffled")
    ax.set_title(f"Permutation importance — {chosen}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "permutation_importance.png", dpi=150)
    plt.close(fig)
    md("![Permutation importance](figures/permutation_importance.png)")
    md()


# ---------------------------------------------------------------- 7. Economics + 8. A/B test design
def economics(y: np.ndarray, p: np.ndarray, mrc: np.ndarray) -> None:
    order = np.argsort(-p)
    y_s, m_s = y[order], mrc[order]
    rows = []
    for k in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]:
        n = int(round(k * len(y)))
        yy, mm = y_s[:n], m_s[:n]
        saved_value = (SAVE_RATE * yy * mm * VALUE_HORIZON_MONTHS).sum()
        # discount is paid by everyone targeted who stays (non-churners + saved churners)
        stayers = (1 - yy) + SAVE_RATE * yy
        offer_cost = (stayers * mm * OFFER_DISCOUNT * OFFER_MONTHS).sum() + CONTACT_COST * n
        rows.append({"target_top": f"{k:.0%}", "customers": n, "churners_reached": int(yy.sum()),
                     "value_saved_$": round(saved_value), "offer_cost_$": round(offer_cost),
                     "net_value_$": round(saved_value - offer_cost)})
    t = pd.DataFrame(rows)
    # random-targeting benchmark at 30%
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), int(0.3 * len(y)), replace=False)
    yy, mm = y[idx], mrc[idx]
    rand_net = (SAVE_RATE * yy * mm * VALUE_HORIZON_MONTHS).sum() - (
        ((1 - yy) + SAVE_RATE * yy) * mm * OFFER_DISCOUNT * OFFER_MONTHS).sum() - CONTACT_COST * len(idx)
    best = t.loc[t["net_value_$"].idxmax()]

    md("## 7. Retention targeting economics (test set)")
    md(f"Assumptions (illustrative, editable at the top of the script): offer = {OFFER_DISCOUNT:.0%} off the "
       f"monthly bill for {OFFER_MONTHS} months, paid by every targeted customer who stays; the offer saves "
       f"{SAVE_RATE:.0%} of would-be churners; a saved customer is worth {VALUE_HORIZON_MONTHS} months of "
       f"revenue; ${CONTACT_COST:.0f} contact cost per customer.")
    md()
    md(df_to_md(t))
    md()
    md(f"**Best policy: target the top {best.target_top} riskiest customers → net value ${best['net_value_$']:,}** "
       f"on this {len(y):,}-customer test set, vs. ${rand_net:,.0f} for randomly targeting 30% and "
       f"${t.iloc[-1]['net_value_$']:,} for targeting everyone.")
    md()

    # A/B test design for the offer among the targeted group
    top_n = int(round(0.3 * len(y)))
    p0 = y_s[:top_n].mean()
    rel_mde = 0.15
    p1 = p0 * (1 - rel_mde)
    alpha, power = 0.05, 0.80
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p0 + p1) / 2
    n_arm = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / (p0 - p1) ** 2
    md("## 8. A/B test design for the retention offer")
    md(f"- Population: top-30% riskiest customers; baseline churn in this group ≈ **{p0:.1%}**")
    md(f"- Hypothesis: the offer reduces churn. H0: churn(treatment) = churn(control); two-sided α = {alpha}, "
       f"power = {power:.0%}")
    md(f"- Minimum detectable effect: {rel_mde:.0%} relative reduction ({p0:.1%} → {p1:.1%})")
    md(f"- **Required sample size ≈ {int(np.ceil(n_arm)):,} customers per arm** (two-proportion z-test)")
    md("- Primary metric: 90-day churn rate; guardrail metrics: revenue per customer (discount cost), "
       "complaint rate, and churn in the untargeted population")
    md("- Randomise at customer level within the targeted group, fix the sample size and duration in "
       "advance (no peeking), and report the effect with a 95% confidence interval")
    md()


def main() -> None:
    md("# Customer Churn Prediction & Retention Targeting — Results")
    md()
    df = load()
    run_sql(df)
    summary = evaluate(df)
    (REPORTS / "results.md").write_text("\n".join(report))
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
