# Credit Default Risk Prediction and Scoring API

> **Status: in progress.** The scope and approach below are the build plan. Results and run instructions will be added as each stage is completed.

An end-to-end credit default prediction pipeline on a public lending dataset, served through a FastAPI scoring endpoint.

## Problem

Given a borrower's financial profile, estimate the probability that they will experience serious delinquency (90+ days past due) within two years. Lenders use this kind of estimate for approval decisions, limit setting, and collections prioritisation.

The target is heavily imbalanced (defaults are the minority class), so accuracy is a misleading metric here. The evaluation is built around ranking quality and an explicit decision threshold instead.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle): about 150,000 borrower records with 10 features such as revolving utilisation, age, debt ratio, monthly income, and counts of past-due events. Target: `SeriousDlqin2yrs`.

The dataset is not committed to this repository. Download it from Kaggle and place `cs-training.csv` in `data/`.

## Approach

1. **Data cleaning and EDA**: inspect distributions, handle missing values (monthly income, dependents), cap or treat implausible outliers, check feature-target relationships.
2. **Feature engineering**: derived ratios and aggregate delinquency features, missing-value indicators.
3. **Class imbalance**: stratified splits, class weighting, and comparison against resampling.
4. **Models**: Logistic Regression as an interpretable baseline, LightGBM as the main model, both trained on the same split and cross-validated.
5. **Evaluation**: AUC-ROC, KS statistic, precision-recall, and a decision threshold chosen from the precision/recall trade-off rather than the default 0.5.
6. **Explainability**: SHAP for global feature importance and per-applicant explanations.
7. **Serving**: FastAPI endpoint with Pydantic request validation, containerised with Docker.

## Planned project structure

```
credit-default-risk-api/
├── data/                  # dataset goes here (gitignored)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── features.py        # cleaning and feature engineering
│   ├── train.py           # model training
│   ├── evaluate.py        # metrics and threshold selection
│   └── explain.py         # SHAP analysis
├── app/
│   ├── main.py            # FastAPI app
│   └── schemas.py         # Pydantic request/response models
├── models/                # saved model artifacts
├── Dockerfile
├── requirements.txt
└── README.md
```

## Planned API

`POST /predict` accepts a borrower's feature values, validates them with Pydantic, and returns a default probability and a decision at the chosen threshold.

## Roadmap

- [x] Define problem, dataset, and evaluation plan
- [ ] Data cleaning and EDA
- [ ] Feature engineering
- [ ] Baseline (Logistic Regression) and LightGBM training
- [ ] Evaluation: AUC, KS, threshold selection
- [ ] SHAP explainability
- [ ] FastAPI scoring endpoint
- [ ] Dockerfile
- [ ] Results and usage sections in this README

## Results

To be added after evaluation.

## Tech stack

Python, Pandas, NumPy, scikit-learn, LightGBM, SHAP, FastAPI, Pydantic, Docker
