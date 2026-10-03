import pandas as pd

data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
h = pd.to_numeric(data.TotalCharges, errors='coerce')
c = h.isnull()
h.loc[c] = 0
data.TotalCharges = h
Churn_Numeric = (data.Churn == "Yes")
Churn_Numeric = Churn_Numeric.astype(float)
data["Churn_Numeric"] = Churn_Numeric
data.to_csv("data/ProcessedData.csv", index=False)
