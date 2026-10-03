import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/ProcessedData.csv")

churn_by_contract = data.groupby(["Contract"])["Churn_Numeric"].mean()
plt.figure(figsize=(8, 10))
plt.title("Churn by Contract")
plt.plot(churn_by_contract)
plt.ylabel("Churn")
plt.xlabel("Contract")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 10))
plt.title("Tenure Distribution")
plt.hist(data.tenure, bins=20, edgecolor="black")
plt.ylabel("Frequency")
plt.xlabel("Tenure")
plt.show()

monthly_non_churned = data["MonthlyCharges"][data["Churn"] == "No"]
monthly_churned = data["MonthlyCharges"][data["Churn"] == "Yes"]
data_to_plot = [monthly_non_churned, monthly_churned]
plt.figure(figsize=(8, 10))
plt.title("Monthly Charges by Churn")
plt.boxplot(data_to_plot, vert=True)
plt.xticks([1, 2], ["No", "Yes"])
plt.ylabel("Monthly Charges")
plt.xlabel("Churn")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
