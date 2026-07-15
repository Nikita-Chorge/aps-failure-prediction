# APS Failure Prediction — Scania Trucks

A cost-sensitive machine learning project for predicting Air Pressure System (APS) failures in Scania trucks, using real-world industrial sensor data released for the 2016 IDA Industrial Challenge.

## Problem

Scania trucks report component failures via 171 anonymized sensor features. The dataset's positive class (APS-related failures) makes up only **1.67%** of the 60,000 training examples, making this a severe class-imbalance problem. Misclassification costs are asymmetric and explicitly defined by Scania:

| | Predicted Failure | Predicted OK |
|---|---|---|
| **Actually Failure** | — | Cost: 500 (missed breakdown) |
| **Actually OK** | Cost: 10 (unnecessary check) | — |

The goal is minimizing total cost, not maximizing accuracy — a model predicting "no failure" every time would score ~98% accuracy while being useless in practice.

## Dataset

- **Source:** [UCI Machine Learning Repository — APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)
- **Size:** 60,000 training rows, 16,000 test rows, 171 anonymized features
- **Missing data:** present across nearly all columns; feature names are anonymized for proprietary reasons
- **Feature structure:** 100 single-value sensor counters + 7 histogram-style sensor groups (10 bins each)

## Methodology

1. **Data inspection** — auto-detected header offset, confirmed class distribution (59,000 neg / 1,000 pos) matched documentation.
2. **Missingness handling** — dropped 7 columns with >70% missing values (all confirmed single counters, not histogram bins). Empirically tested whether additionally dropping mid-tier missing columns (38–66% missing) helped or hurt, using validation cost as the metric — **kept them**, since removing them increased cost (16,950 vs. 14,430 with a single default-threshold model), confirming XGBoost's native missing-value handling was extracting real signal from partially-missing columns rather than noise.
3. **Imbalance handling** — built and compared two approaches:
   - A single XGBoost classifier with `scale_pos_weight` to counter the 59:1 imbalance.
   - A homogeneous 6-model ensemble (EasyEnsemble-style): the full positive class paired with 6 disjoint subsets of the negative class, predictions combined via probability averaging.

   **Finding:** after threshold tuning, both approaches performed nearly identically (validation cost 5,530 vs. 5,520) — the added architectural complexity of the ensemble did not meaningfully outperform a single well-tuned model. Threshold calibration, not model architecture, was the actual lever for improvement.
4. **Threshold selection** — used Scania's published cost matrix to sweep thresholds against real cost rather than accuracy or F1. Validated the choice via 5-fold stratified cross-validation (not a single train/val split) to avoid overfitting the threshold to one split's quirks. Reduced validation cost from 9,270 (default 0.5 threshold) to 5,530.
5. **Explainability** — used SHAP (TreeExplainer) for both global feature importance and individual (local) prediction explanations. Given anonymized feature names, physical meaning of top features cannot be determined, but relative importance and directional effect per prediction are reported.

## Results

| Metric | Value |
|---|---|
| Validation cost (default 0.5 threshold) | 9,270 |
| Validation cost (CV-tuned threshold, 0.01) | 5,530 |
| **Final test set cost** | **14,920** |
| 2016 Challenge — 1st place | 9,920 |
| 2016 Challenge — 2nd place | 10,900 |
| 2016 Challenge — 3rd place | 11,480 |

**On the test-set gap:** the held-out test set has a **2.34% positive rate versus 1.67% in training** — roughly 40% relatively more actual failures than the training distribution. Since false negatives are penalized at 50x the rate of false positives, this base-rate shift alone accounts for a substantial share of the cost gap independent of model quality. Feature-level distribution checks on the top SHAP-ranked features showed only minor shifts between train and test, suggesting the gap is driven primarily by label prevalence rather than covariate drift.

**What likely closes the remaining gap** (not attempted here due to time constraints): feature engineering on the 7 histogram groups (e.g., weighted bin statistics instead of raw counts), which published top submissions for this challenge are known to have used.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Jupyter

## Project Structure

```
├── data/           # raw CSVs (not committed — see .gitignore)
├── notebooks/      # analysis notebook(s)
├── outputs/        # saved models, plots, results
├── src/            # reusable scripts (if any)
├── requirements.txt
└── README.md
```

## Limitations

- Feature anonymization prevents physically grounded interpretation of SHAP results.
- Ensemble approach did not outperform a single tuned model — included here as a tested comparison, not a recommended architecture.
- Final test-set cost trails published 2016 benchmarks; the base-rate explanation is a hypothesis backed by direct measurement, not a fully closed investigation.
