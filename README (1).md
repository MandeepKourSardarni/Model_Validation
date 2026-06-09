# Independent Model Validation — Credit-Risk Classifier

A hands-on **model-risk validation** case study. Rather than building "the best model," this project takes a credit-risk classifier and challenges it as an **independent validator** would inside an Enterprise Model Risk Management function — in the spirit of the U.S. Federal Reserve's **SR 11-7** and Canada's **OSFI Guideline E-23 (2027)**.

> **Core question:** *Can we trust this model, and where does it break?*

## Why this project

Model validation is the "second line of defence": an independent team that challenges models built by developers before and during their use. This notebook reproduces that workflow on a public credit-risk dataset, assessing a model across the dimensions a real AI-validation team uses.

## Dataset

Statlog **German Credit** data — 1,000 loan applicants, 20 features. The target is reframed as `high_risk = 1` (bad credit / likely default), the costly outcome a lender wants to catch. Classes are imbalanced (~30% high-risk).

## Validation dimensions covered

| # | Dimension | What I check | Tooling |
|---|-----------|--------------|---------|
| 1 | Performance & stability | Precision/recall/F1, ROC-AUC, PR-AUC + 5-fold CV variance | scikit-learn |
| 2 | Benchmarking | vs. logistic regression and a majority-class baseline | scikit-learn |
| 3 | Calibration | reliability curve + Brier score | scikit-learn |
| 4 | Robustness | AUC degradation under injected input noise / drift | numpy |
| 5 | Fairness | flag-rate, TPR, FPR by **sex** and **age**; demographic-parity & equal-opportunity gaps | custom |
| 6 | Explainability | global + local feature attribution; conceptual-soundness check | SHAP |

## Key findings

- **Performant and reproducible:** ROC-AUC ≈ 0.80; cross-validated AUC 0.786 ± 0.017 (stable across folds).
- **Benchmarking flag:** the Random Forest (0.804) does **not** meaningfully beat logistic regression (0.807) — added complexity is not justified by performance.
- **Cost-sensitivity:** recall on high-risk applicants is ~0.60, so the model misses ~40% of bad loans; accuracy alone overstates quality on imbalanced data.
- **Fairness concern:** the model flags women as high-risk far more often than men (demographic-parity gap ≈ 0.12; equal-opportunity gap ≈ 0.16) and young applicants more than older — disparate impact that would require mitigation/escalation.
- **Explainability:** SHAP drivers (checking-account status, credit duration, credit history, savings) are conceptually sound for credit risk with no obvious leakage.

## Validator's conclusion

Illustrative outcome: **conditional approval** — fit for ranking credit risk, subject to (1) a decision threshold tied to the bank's cost of false negatives, (2) monitoring/mitigation of the fairness gaps, and (3) drift monitoring. This mirrors how a real review closes: not "good/bad," but **fit-for-purpose with conditions and monitoring**.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook model_validation_credit_risk.ipynb
```

The dataset (`german.csv`) is included. Built with Python, scikit-learn, SHAP, pandas, matplotlib.

## Repository structure

```
.
├── model_validation_credit_risk.ipynb   # the validation report (run top-to-bottom)
├── german.csv                            # dataset
├── requirements.txt
└── README.md
```
