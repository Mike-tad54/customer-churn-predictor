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
2. **Tenure is bimodal.** One large spike at 0-4 months and another at about 70 months. The left spike is new customers, who are the most likely to leave. The model may need to treat new customers differently.
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
- **Result:** recall rose from 47.5% to 59.8% (accuracy 82.0%, precision 68.4%). More columns gave the model more signal.

### Feature engineering

- **Tried:** `tenure_bucket` (6 groups with `pd.cut`) and `ContractxCharges` (contract months × monthly charge).
- **First attempt:** recall dropped to 53.4%.
  - **Cause:** I overwrote the original `tenure` column with the buckets, so the model lost exact tenure.
  - **Lesson:** add engineered features alongside the originals. Check each change against the previous score.
- **After the fix:** recall 56.8%, accuracy 81.3%, precision 67.3%. Still below the 59.8% without engineered features.
  - **Conclusion:** the two engineered features did not help Logistic Regression. Not every idea works.
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
- **Result:** it did not. Accuracy 80.2%, recall 55.0%, precision 64.9%.
- **Likely cause:** default settings overfit a small, mostly weak-signal dataset. The gap is about 7 customers, so the two models are effectively tied on this split.
- **Lesson:** a fancier model is not automatically better. On simple tabular data, Logistic Regression is hard to beat without tuning.

### Confusion matrices

| Model | Correct "No" | False alarms | Missed churners | Caught churners |
|---|---|---|---|---|
| Logistic | 933 | 103 | 161 | 212 |
| XGBoost | 925 | 111 | 168 | 205 |
| Dummy | 1036 | 0 | 373 | 0 |

- **Bug:** I multiplied the confusion matrix by 100 (copied from the percentage metrics). Counts are not percentages.
- The model misses 161 of 373 churners (43%). For churn, a missed customer usually costs more than a false alarm.

## Day 5: Saving the model

- **Decision:** train once, save with `joblib`, and load in `Predict()`. Retraining on every call is slow and not how real systems work.
- **Bug:** I first saved `data.columns` to `columns.json`. The model is trained on `X`, not `data`. Fixed to `X.columns`. The file now lists the 52 training features.
- **Note for the API project (1.7):**
  - A single raw customer, once one-hot encoded, only has columns for its own values. Rebuild all 52 columns from `columns.json`, filling missing ones with 0.
  - `pd.cut(tenure, 6)` computes edges from the data. A single customer needs fixed edges.

## Open items

- [ ] **5-fold cross-validation** for Logistic Regression and XGBoost, so the comparison doesn't rest on one split.
- [ ] **Model choice to revisit:** the saved model includes the engineered features, but the version without them scored slightly higher on this split (recall 59.8% vs. 56.8%). Decide after cross-validation.
- [ ] **Error analysis on the 161 missed churners:** compare them with the caught churners (contract, tenure, monthly charges).
- [ ] Add a scaling step inside a pipeline.
- [ ] Tag `v1.0` after the above.
