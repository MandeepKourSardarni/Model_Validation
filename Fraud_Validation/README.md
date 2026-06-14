# Fraud Detection — Baseline Model & Validation

A compact, end-to-end fraud-detection notebook on the **PaySim** banking
transactions dataset. The goal isn't just to build a model — it's to **evaluate
and challenge it the way a model-risk validator would**.

## What it does

1. Trains a simple **Logistic Regression** baseline to flag fraudulent transactions.
2. Evaluates it honestly for an imbalanced problem — **recall, precision, ROC-AUC,
   and PR-AUC** — not misleading accuracy.
3. Shows the **decision threshold as a business choice** (catch more fraud vs.
   fewer false alarms).
4. Adds an unsupervised **Isolation Forest** cross-check that needs no labels.
5. Closes with a **validator's lens**: assumptions, limitations, and what to
   check before production.

## Why it's built this way

Fraud is well under 1% of transactions, so a model that predicts "never fraud"
looks ~99% accurate while catching nothing. The project leans on the metrics
that actually matter and demonstrates the model-risk practices a regulated bank
expects:

- **Stratified train/test split** so rare fraud is represented in both sets.
- **Scaling inside a Pipeline** so the scaler is fit on training data only — no
  data leakage.
- `class_weight="balanced"` so the rare fraud class isn't ignored.
- **PR-AUC alongside ROC-AUC**, since ROC-AUC flatters imbalanced problems.
- A written set of **limitations and next checks** (drift, threshold governance,
  fairness, reproducibility).

This mirrors the standards banks use — **SR 11-7** and Canada's **OSFI Guideline
E-23 (Model Risk Management)**, which takes effect **May 1, 2027** and explicitly
covers AI/ML models at federally regulated institutions.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook fraud-detection-improved.ipynb
```

The notebook uses the real PaySim file if it's present, otherwise it generates a
small synthetic sample with the **same columns**, so it runs anywhere.

**To use the real data:** download PaySim from Kaggle (`ealaxi/paysim1`) and set
the path at the top of the load cell, e.g.:

```python
PAYSIM_PATH = r"C:\Users\you\Downloads\paysim.csv"   # raw string avoids the \U path error
```

## Key takeaway

A simple model already catches most fraud, but its precision is low — far too
many false alarms to deploy as-is. The value is in showing that clearly,
explaining the trade-offs, and documenting what would need to change. **Build it,
then challenge it** — that's the model-risk validator mindset.

---

*Data is synthetic/simulated; fraud labels are illustrative. Not for production decisions.*
