import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_rows', None)

data = pd.read_csv("data/ProcessedData.csv")
X = data.drop(columns=["Churn", "Churn_Numeric", "customerID"])
X = pd.get_dummies(X, dtype=float)
Y = data["Churn_Numeric"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train)
y_logistic_pred = logistic_model.predict(X_test)
logistic_model_accuracy = accuracy_score(y_test, y_logistic_pred) * 100
logistic_model_recall = recall_score(y_test, y_logistic_pred) * 100
logistic_model_precision = precision_score(y_test, y_logistic_pred) * 100
logistic_model_confusion = confusion_matrix(y_test, y_logistic_pred)
print(f"Logistic Model Accuracy: {logistic_model_accuracy}%")
print(f"Logistic Model Recall: {logistic_model_recall}%  Logistic Model Precision: {logistic_model_precision}%")
print(f"Logistic Model Confusion: {logistic_model_confusion}")
joblib.dump(logistic_model, "model/logistic_model.pkl")
with open("model/columns.json", "w") as file:
    json.dump(X.columns.tolist(), file, indent=4)

xgboost_model = XGBClassifier(random_state=42)
xgboost_model.fit(X_train, y_train)
y_xgboost_pred = xgboost_model.predict(X_test)
xgboost_model_accuracy = accuracy_score(y_test, y_xgboost_pred) * 100
xgboost_model_recall = recall_score(y_test, y_xgboost_pred) * 100
xgboost_precision = precision_score(y_test, y_xgboost_pred) * 100
xgboost_model_confusion = confusion_matrix(y_test, y_xgboost_pred)
print(f"XGBoost Model Accuracy: {xgboost_model_accuracy}%")
print(f"XGBoost Model Recall: {xgboost_model_recall}%  XGBoost Precision: {xgboost_precision}%")
print(f"XGBoost Model Confusion: {xgboost_model_confusion}")

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_accuracy = accuracy_score(y_test, dummy_pred) * 100
dummy_recall = recall_score(y_test, dummy_pred) * 100
dummy_precision = precision_score(y_test, dummy_pred) * 100
dummy_confusion = confusion_matrix(y_test, dummy_pred)
print(f"My Dummy Accuracy: {dummy_accuracy}%")
print(f"Dummy Recall: {dummy_recall}%  Dummy Precision: {dummy_precision}%")
print(f"Dummy Confusion: {dummy_confusion}")

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, cv_estimator in [("Logistic Regression", LogisticRegression(max_iter=10000)),
                           ("XGBoost", XGBClassifier(random_state=42))]:
    results = cross_validate(cv_estimator, X, Y, cv=splitter, scoring=["accuracy", "recall", "precision"])
    print(f"{name} 5-fold CV")
    for metric in ["accuracy", "recall", "precision"]:
        scores = results[f"test_{metric}"]
        print(f"  {metric}: {scores.mean() * 100:.1f}% +/- {scores.std() * 100:.1f}")


def Predict(customers):
    model = joblib.load("model/logistic_model.pkl")
    pred = model.predict(customers)
    return pred


print(f"Model Pred: {Predict(X_test.iloc[[0]])}, Y Pred: {y_test.iloc[[0]]}")
