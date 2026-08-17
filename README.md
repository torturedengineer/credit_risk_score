# Credit Risk Scoring & Default Prediction

A small but complete consumer-credit default model: EDA → missing-value treatment → interpretable baseline vs.
gradient boosting → calibrated probabilities → a validated credit-score/risk-band framework → and an extension
most portfolio projects skip, **auto-generated, regulator-style adverse action reason codes** for every declined
applicant.

## Why this project

Most "credit risk" portfolio projects stop at a ROC-AUC number and a SHAP plot. This one goes one step further:
in the US, ECOA/Reg B require lenders to tell a declined applicant the **specific reasons** they were denied.
This project turns the same SHAP values used for interpretability into per-applicant reason codes — with an
explicit, documented list of factors (age, marital status, residential type) that are deliberately **excluded**
from those reasons even though the model can see them, because they're legally protected or proxy for
protected characteristics.

## Pipeline

| Step | What's done |
|---|---|
| Data | 4,455 consumer loan applications ([gastonstat/CreditScoring](https://github.com/gastonstat/CreditScoring)) |
| EDA | Target imbalance, distribution checks, good vs. bad payer comparisons |
| Missing values | Sentinel decoding (`99999999`, `0`) → `NaN` → missingness flags → median/mode imputation |
| Feature engineering | Debt-to-income, loan-to-income, installment burden, % of price financed (winsorized on train only) |
| Baseline model | Logistic Regression, `class_weight="balanced"`, coefficient interpretation |
| Main model | XGBoost, `scale_pos_weight` for imbalance |
| Evaluation | ROC-AUC, PR-AUC, confusion matrix — **not** accuracy, given the ~28% default base rate |
| Calibration | Isotonic (5-fold), validated with Brier score + reliability curve |
| Explainability | SHAP (global importance + per-applicant values) |
| Scoring | Log-odds (PDO) score anchored to the *portfolio's own* base rate, not a borrowed prime-book constant |
| Risk bands | Quantile-cut, then **validated** against actual observed default rate per band |
| **Extension** | **Adverse action reason codes**: top-4 SHAP-driven, ECOA-safe reasons per decline, plus a portfolio-level "most common decline reasons" view |

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | ~0.845 | ~0.70 |
| XGBoost | ~0.845 | ~0.69 |

Isotonic calibration reduces the Brier score from ~0.153 to ~0.137. Risk bands validate cleanly: observed
default rate rises monotonically from ~2.7% (Prime) to ~72% (Decline).

## Files

- `credit_risk_scoring.ipynb` — the full, already-executed notebook
- `credit_risk_scoring_preview.html` — static HTML view (open in any browser, no Jupyter needed)
- `data/CreditScoring.csv` — the raw dataset
- `requirements.txt`

## Running it yourself

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_scoring.ipynb
```

## Responsible-use note

This is a portfolio/teaching project, not a production underwriting system. Before anything like this went live,
it would need reject inference (the model only ever sees applicants who were funded), formal fair-lending /
disparate-impact testing on the underlying model — not just the reason-code layer — and ongoing population
stability monitoring. These are called out explicitly in the notebook's conclusion.

## Suggested resume bullet

> Built an end-to-end credit-default model (logistic regression baseline + calibrated XGBoost, ROC-AUC 0.85) on
> consumer-credit data, translating predicted probabilities into a validated risk-band framework and an
> auto-generated, ECOA-compliant adverse-action reason-code system for declined applicants — reducing what would
> normally be a manual compliance step to a reproducible, auditable model output.
