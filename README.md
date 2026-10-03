# Customer Churn Predictor

Predicts whether a telecom customer will cancel their service, using the IBM Telco Customer Churn dataset. Built as a baseline-first ML project: every model is compared against a naive baseline, judged on recall as well as accuracy, and compared with 5-fold cross-validation.

**Final model:** Logistic Regression on 45 features. About 55% recall and 80.5% accuracy (5-fold CV).

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
4. **Encoding:** one-hot encoding of all categorical columns (45 features in total).
5. **Feature engineering (tested, then removed):** `tenure_bucket` and `ContractxCharges` did not improve recall under cross-validation, so the final model leaves them out.
6. **Model comparison:** Logistic Regression vs. XGBoost (default settings) with stratified 5-fold cross-validation.
7. **Error analysis:** confusion matrices on a held-out test set.

## Results

### 5-fold cross-validation (final 45 features)

Mean ± standard deviation across 5 folds.

| Model | Accuracy | Recall | Precision |
|---|---|---|---|
| **Logistic Regression** | **80.5% ± 1.0** | **55.4% ± 2.9** | **65.8% ± 1.9** |
| XGBoost | 78.5% ± 0.7 | 52.0% ± 1.3 | 61.1% ± 1.8 |

Logistic Regression is better on all three metrics.

### Single 80/20 split (1,409 test customers, 373 churners)

| Model | Accuracy | Recall | Precision |
|---|---|---|---|
| Dummy (always "No churn") | 73.5% | 0.0% | n/a |
| Logistic Regression, 4 numeric columns | 80.6% | 47.5% | 69.7% |
| Logistic Regression, 45 features (final) | 82.0% | 59.8% | 68.4% |

Confusion matrix of the final model (rows = real, columns = predicted; order is No, Yes):

| | Predicted No | Predicted Yes |
|---|---|---|
| **Real No** | 933 | 103 |
| **Real Yes** | 150 | 223 |

A single split overestimates recall. Cross-validation gives the more honest figure, about 55%.

## Key findings

- **Accuracy is misleading here.** The dummy model scores 73.5% accuracy while catching zero churners. Recall is the metric that matters.
- **Contract type is the strongest signal.** Churn is about 43% for month-to-month contracts, 11% for one-year, and 3% for two-year.
- **New customers churn most.** Tenure is bimodal: a large group of very new customers and a large group of long-term ones.
- **Encoding the categorical columns helped most.** Recall rose from 47.5% to 59.8% on the single split.
- **Engineered features did not help.** Under cross-validation, the version without them had higher recall (55.4% vs. 52.9%).
- **XGBoost did not beat Logistic Regression** with default settings.
- **The model still misses about 40% of churners** (150 of 373 on the test split).

## Project structure

```
.
├── preprocess.py     # cleans the raw CSV, writes data/ProcessedData.csv
├── eda.py            # the three EDA plots
├── main.py           # features, training, evaluation, cross-validation, model saving, Predict()
├── model/
│   ├── logistic_model.pkl   # trained Logistic Regression
│   └── columns.json         # the 45 feature names, in training order
├── decisions.md      # engineering log: what was tried, what failed, why
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1. Put the raw CSV at data/WA_Fn-UseC_-Telco-Customer-Churn.csv
python preprocess.py   # creates data/ProcessedData.csv
python eda.py          # optional: shows the plots
python main.py         # trains, prints metrics and CV scores, saves the model to model/
```

To score customers with the saved model, call `Predict(customers)` in `main.py`. Input must have the same 45 columns, in the same order, as listed in `model/columns.json`.

## Limitations and next steps

- Recall of about 55% means many churners are still missed. A lower decision threshold would trade precision for recall.
- Scaling is not applied. Logistic Regression converges anyway (`max_iter=10000`), but a scaling step inside a pipeline is the correct approach.
- XGBoost was not tuned.
- `Predict()` expects already-encoded input. Accepting a raw customer record needs a preprocessing step. This will be handled when the model is served as an API.
