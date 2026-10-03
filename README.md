# Customer Churn Predictor

Predicts whether a telecom customer will cancel their service, using the IBM Telco Customer Churn dataset. Built as a baseline-first ML project: every model is compared against a naive baseline, and judged on recall as well as accuracy.

> **Status:** baseline complete. Cross-validation comparison and the `v1.0` tag are still to come.

## Problem

Churn means a customer stops doing business with a company. Winning back a lost customer is expensive, so the goal is to flag likely churners early enough for the company to act (for example, with a retention offer).

This is a binary classification problem: `Churn = Yes` (1) or `No` (0).

## Dataset

- **Source:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn`. About 26.5% of customers churned, so the classes are imbalanced.
- **Not included in this repo.** Download the CSV and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Approach

1. **Cleaning:** `TotalCharges` has blank values for customers with `tenure = 0` (new customers who haven't been billed yet). These were set to 0.
2. **EDA:** churn by contract type, tenure distribution, monthly charges by churn status.
3. **Baselines:** a majority-class `DummyClassifier`, then Logistic Regression on 4 numeric columns.
4. **Encoding:** one-hot encoding of all categorical columns (52 features in total).
5. **Feature engineering:** `tenure_bucket` (6 groups) and `ContractxCharges` (contract length in months × monthly charge).
6. **Models compared:** Logistic Regression vs. XGBoost (default settings), on the same 80/20 split (`random_state=42`).
7. **Error analysis:** confusion matrices for every model.

## Results

Evaluated on a held-out test set of 1,409 customers (373 churners). Single train/test split.

| Model | Accuracy | Recall | Precision |
|---|---|---|---|
| Dummy (always "No churn") | 73.5% | 0.0% | n/a |
| Logistic Regression (4 numeric columns) | 80.6% | 47.5% | 69.7% |
| Logistic Regression (all columns + engineered features) | 81.3% | 56.8% | 67.3% |
| XGBoost (all columns + engineered features) | 80.2% | 55.0% | 64.9% |

Confusion matrices (rows = real, columns = predicted; order is No, Yes):

| Model | Correct "No" | False alarms | Missed churners | Caught churners |
|---|---|---|---|---|
| Logistic Regression | 933 | 103 | 161 | 212 |
| XGBoost | 925 | 111 | 168 | 205 |

## Key findings

- **Accuracy is misleading here.** The dummy model scores 73.5% accuracy while catching zero churners. Recall is the metric that matters.
- **Contract type is the strongest signal.** Churn is about 43% for month-to-month contracts, 11% for one-year, and 3% for two-year.
- **New customers churn most.** Tenure is bimodal: a large group of very new customers and a large group of long-term ones.
- **More features helped; engineered features did not.** Adding all encoded columns raised recall from 47.5% to 59.8%. The two engineered features slightly lowered it (to 56.8%).
- **XGBoost did not beat Logistic Regression** with default settings. The gap is about 7 customers, within the noise of a single split.
- **The model still misses 43% of churners** (161 of 373).

## Project structure

```
.
├── preprocess.py     # cleans the raw CSV, writes data/ProcessedData.csv
├── eda.py            # the three EDA plots
├── main.py           # features, training, evaluation, model saving, Predict()
├── model/
│   ├── logistic_model.pkl   # trained Logistic Regression
│   └── columns.json         # the 52 feature names, in training order
├── decisions.md      # engineering log: what was tried, what failed, why
└── README.md
```

## How to run

```bash
pip install pandas scikit-learn xgboost joblib matplotlib

# 1. Put the raw CSV at data/WA_Fn-UseC_-Telco-Customer-Churn.csv
python preprocess.py   # creates data/ProcessedData.csv
python eda.py          # optional: shows the plots
python main.py         # trains, prints metrics, saves the model to model/
```

To score customers with the saved model, call `Predict(customers)` in `main.py`. Input must have the same 52 columns, in the same order, as listed in `model/columns.json`.

## Limitations and next steps

- Results come from a single split. **5-fold cross-validation** is next, to compare models fairly.
- Scaling is not applied. Logistic Regression converges anyway (`max_iter=10000`), but a scaling step inside a pipeline is the correct fix.
- XGBoost was not tuned.
- `Predict()` expects already-encoded input. Accepting a raw customer record needs a preprocessing step and a fixed set of tenure-bucket edges. This will be handled when the model is served as an API.
- The model is a baseline. Recall of 57% means many churners are still missed.
