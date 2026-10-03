# Decisions Log: Customer Churn Predictor

Format for each entry: what I tried, what happened, and what I chose.

## Day 1: Setup and data loading

- **Churn** means a customer stops doing business with the company. The dataset has 7,043 customers (rows), 21 columns, and the target is `Churn`.
- **Problem:** `TotalCharges` loaded as text because some values are blank.
  - **Finding:** the blank rows are customers with `tenure = 0`. They haven't been billed yet.
  - **Decision:** convert with `pd.to_numeric(errors="coerce")`, then set the blanks to 0. A customer who hasn't used the service has paid nothing.
- **Decision:** never overwrite the raw CSV. Cleaned data goes to a separate `ProcessedData.csv`, produced by `preprocess.py`.

## Day 2: EDA

Three takeaways:

1. **Contract type drives churn.** Churn is about 43% on month-to-month contracts, 11% on one-year, and 3% on two-year. Customer loyalty rises with contract length. Contract will matter in any model.
2. **Tenure is bimodal.** One large spike at 0-4 months and another at about 70 months. The left spike is new customers, who are the most likely to leave.
3. **Churners pay more.** The boxplot shows the median monthly charge is about 80 for churners vs. about 65 for non-churners. (A boxplot shows the median as the line in the box, the middle 50% of values as the box, and the range as the whiskers.) Higher bills may push customers away.

- **Decision:** a bar chart would suit the contract plot better than a line, since contract types are categories with no order in between. Left as a line for now.

## Day 3: Baseline

- **Tried:** majority-class `DummyClassifier` and Logistic Regression on 4 numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`), 80/20 split, `random_state=42`.
- **Result:**

| Model | Accuracy | Recall | Precision |
|---|---|---|---|
| Dummy | 73.5% | 0.0% | n/a |
| Logistic | 80.6% | 47.5% | 69.7% |

- **Lesson:** accuracy is misleading on imbalanced data. The dummy gets 73.5% by never predicting churn. Recall exposed the real weakness: the model caught fewer than half of the churners.
- **Problem:** `ConvergenceWarning` from Logistic Regression. **Decision:** `max_iter=10000`. This is a workaround; scaling inside a pipeline is the proper fix.
- **Problem:** the dummy's precision was undefined because it never predicts "Yes". That is expected and harmless.
- **Problem:** `metrics.json` failed with `TypeError: unhashable type: 'dict'`. I had put two dictionaries inside a set instead of giving each one a key.

## Day 4: Encoding, features, and a second model

### One-hot encoding

- **Tried:** `pd.get_dummies` on all categorical columns, dropping `customerID`, `Churn`, and `Churn_Numeric` first. Dropping the target columns matters: leaving them in would leak the answer and give a near-perfect score.
- **Result:** 45 features. Recall rose from 47.5% to 59.8% (accuracy 82.0%, precision 68.4%) on the single split. More columns gave the model more signal.

### Feature engineering

- **Tried:** `tenure_bucket` (6 groups with `pd.cut`) and `ContractxCharges` (contract months × monthly charge).
- **First attempt:** recall dropped to 53.4%.
  - **Cause:** I overwrote the original `tenure` column with the buckets, so the model lost exact tenure.
  - **Lesson:** add engineered features alongside the originals. Check each change against the previous score.
- **After the fix:** recall 56.8% on the single split, still below the 59.8% without them. Cross-validation later confirmed this (see below).
- **`pd.cut` notes:** use `include_lowest=True` so `tenure = 0` isn't dropped to NaN. Use `labels=` for plain names, because XGBoost rejects feature names containing `[`, `]`, or `<`.

### Normalization

- **Tried:** z-score normalization of `tenure`, `MonthlyCharges`, and `TotalCharges`.
- **Result:** accuracy moved by about 0.2 points. Recall did not change.
- **Why:** normalization mainly helps training converge; it does not make the model more capable. With `max_iter=10000` it was already converging.
- **Two mistakes to avoid:**
  1. I normalized `MonthlyCharges` before building `ContractxCharges`, which made that feature meaningless. Build features from raw columns first, then scale.
  2. I computed the mean and standard deviation on the full dataset, leaking a little test information into training. Scaling statistics should come from the training set only. A pipeline solves this.
- **Decision:** left scaling out of the final version.

### XGBoost

- **Expected:** XGBoost to beat Logistic Regression.
- **Result (single split, 52 features):** it did not. Accuracy 80.2%, recall 55.0%, precision 64.9%.
- **Likely cause:** default settings overfit a small, mostly weak-signal dataset.
- **Mistake:** I passed `max_iter=10000` to XGBoost. It isn't an XGBoost parameter and was ignored, with a warning.
- **Lesson:** a fancier model is not automatically better. On simple tabular data, Logistic Regression is hard to beat without tuning.

### Confusion matrices (single split, 52 features)

| Model | Correct "No" | False alarms | Missed churners | Caught churners |
|---|---|---|---|---|
| Logistic | 933 | 103 | 161 | 212 |
| XGBoost | 925 | 111 | 168 | 205 |
| Dummy | 1036 | 0 | 373 | 0 |

- **Bug:** I multiplied the confusion matrix by 100 (copied from the percentage metrics). Counts are not percentages.
- For churn, a missed customer usually costs more than a false alarm.

## Day 5: Saving the model

- **Decision:** train once, save with `joblib`, and load in `Predict()`. Retraining on every call is slow and not how real systems work.
- **Bug:** I first saved `data.columns` to `columns.json`. The model is trained on `X`, not `data`. Fixed to `X.columns`.
- **Note for the API project (1.7):** a single raw customer, once one-hot encoded, only has columns for its own values. Rebuild all feature columns from `columns.json`, filling missing ones with 0.

## Day 6: Cross-validation and final model

- **Problem:** single-split scores were noisy. The 2-point gaps between models were about 7 customers, within chance.
- **Tried:** stratified 5-fold cross-validation (`shuffle=True`, `random_state=42`), using the same folds for every model.

| Model | Features | Accuracy | Recall | Precision |
|---|---|---|---|---|
| Logistic | 52 (with engineered) | 80.4% ± 0.8 | 52.9% ± 2.0 | 66.3% ± 1.8 |
| XGBoost | 52 (with engineered) | 78.7% ± 1.1 | 52.8% ± 2.1 | 61.4% ± 2.7 |
| Logistic | 45 (plain) | 80.5% ± 1.0 | 55.4% ± 2.9 | 65.8% ± 1.9 |
| XGBoost | 45 (plain) | 78.5% ± 0.7 | 52.0% ± 1.3 | 61.1% ± 1.8 |

- **Findings:**
  - The single-split recall of 56.8% (and 59.8%) was lucky. The honest estimate is about 55%.
  - Logistic Regression beats XGBoost on accuracy and precision in every setup. On recall it ties with the 52-feature set and wins with the 45-feature set.
  - The engineered features lowered Logistic recall under CV (55.4% → 52.9%). They likely repeat information already in `tenure`, `Contract`, and `MonthlyCharges`.
- **Decision:** the final model is Logistic Regression on the 45 plain features. `model/logistic_model.pkl` and `model/columns.json` were rebuilt to match.
- **Check:** the rebuilt model reproduces the earlier 45-feature single-split result (82.0% accuracy, 59.8% recall, 68.4% precision). Confusion matrix: 933 / 103 / 150 / 223.
- **Lesson:** compare models on the same folds, and test each feature idea with CV before keeping it.

## Open items (optional)

- [ ] Error analysis of the 150 missed churners: compare them with the caught churners (contract, tenure, monthly charges).
- [ ] Try a lower decision threshold to trade precision for recall.
- [ ] Add a scaling step inside a pipeline.
- [ ] Tune XGBoost.
