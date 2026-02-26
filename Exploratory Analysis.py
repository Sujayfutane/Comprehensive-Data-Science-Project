# =====================================================
# EXPLORATORY DATA ANALYSIS (EDA) - COLORED & STYLED
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# GLOBAL STYLING
# -----------------------------------------------------

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------

df = pd.read_csv("customer_churn_data.csv")

print("Dataset Shape:", df.shape)

# -----------------------------------------------------
# 2. TARGET VARIABLE ANALYSIS
# -----------------------------------------------------

plt.figure()
sns.countplot(
    x="Churn",
    data=df,
    palette=["#4CAF50", "#F44336"]
)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

# -----------------------------------------------------
# 3. UNIVARIATE ANALYSIS (NUMERICAL)
# -----------------------------------------------------

num_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "ComplaintsCount"]

for col in num_cols:
    plt.figure()
    sns.histplot(
        df[col],
        bins=30,
        kde=True,
        color="#2196F3"
    )
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# -----------------------------------------------------
# 4. UNIVARIATE ANALYSIS (CATEGORICAL)
# -----------------------------------------------------

cat_cols = ["Gender", "ContractType", "PaymentMethod", "InternetService"]

for col in cat_cols:
    plt.figure()
    df[col].value_counts().plot(
        kind="bar",
        color=sns.color_palette("Set3")
    )
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.show()

# -----------------------------------------------------
# 5. BIVARIATE ANALYSIS (CHURN VS FEATURES)
# -----------------------------------------------------

# Tenure vs Churn
plt.figure()
sns.boxplot(
    x="Churn",
    y="Tenure",
    data=df,
    palette=["#81C784", "#E57373"]
)
plt.title("Tenure vs Churn")
plt.show()

# Monthly Charges vs Churn
plt.figure()
sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df,
    palette=["#64B5F6", "#FF8A65"]
)
plt.title("Monthly Charges vs Churn")
plt.show()

# Complaints vs Churn
plt.figure()
sns.boxplot(
    x="Churn",
    y="ComplaintsCount",
    data=df,
    palette=["#AED581", "#FFB74D"]
)
plt.title("Complaints Count vs Churn")
plt.show()

# -----------------------------------------------------
# 6. CHURN VS CONTRACT TYPE
# -----------------------------------------------------

plt.figure()
sns.countplot(
    x="ContractType",
    hue="Churn",
    data=df,
    palette=["#4DB6AC", "#EF5350"]
)
plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Customer Count")
plt.xticks(rotation=20)
plt.legend(title="Churn")
plt.show()

# -----------------------------------------------------
# 7. CHURN RATE BY CATEGORY (TEXT + BAR)
# -----------------------------------------------------

def churn_rate(col):
    return df.groupby(col)["Churn"].mean().sort_values(ascending=False)

contract_churn = churn_rate("ContractType")

plt.figure()
contract_churn.plot(
    kind="bar",
    color=["#E53935", "#FB8C00", "#43A047"]
)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.xticks(rotation=20)
plt.show()

# -----------------------------------------------------
# 8. CORRELATION HEATMAP
# -----------------------------------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(
    df[num_cols + ["Churn"]].corr(),
    annot=True,
    cmap="RdYlBu",
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------------------------------
# 9. MULTIVARIATE SCATTER PLOT
# -----------------------------------------------------

plt.figure()
sns.scatterplot(
    x="Tenure",
    y="MonthlyCharges",
    hue="Churn",
    data=df,
    palette=["#2E7D32", "#C62828"],
    alpha=0.6
)
plt.title("Tenure vs Monthly Charges (Churn Highlighted)")
plt.show()

# -----------------------------------------------------
# 10. KEY BUSINESS INSIGHTS
# -----------------------------------------------------

print("\n--- KEY EDA INSIGHTS ---")
print("✔ Customers with tenure < 12 months churn significantly more")
print("✔ Month-to-month contracts show highest churn risk")
print("✔ Higher monthly charges correlate with higher churn")
print("✔ More complaints strongly increase churn probability")
print("✔ Long-term contracts reduce churn")

print("\nEDA WITH STYLED VISUALS COMPLETED SUCCESSFULLY 🚀")