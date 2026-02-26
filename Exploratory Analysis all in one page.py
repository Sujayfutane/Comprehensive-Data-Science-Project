# =====================================================
# SINGLE PAGE EXPLORATORY DATA ANALYSIS DASHBOARD
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# STYLING
# -----------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (20, 18)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("customer_churn_data.csv")

# -----------------------------------------------------
# CREATE SUBPLOTS GRID
# -----------------------------------------------------
fig, axes = plt.subplots(4, 3)
fig.suptitle("Customer Churn – Exploratory Data Analysis Dashboard", fontsize=18, fontweight="bold")

# -----------------------------------------------------
# ROW 1
# -----------------------------------------------------

# Churn Distribution
sns.countplot(x="Churn", data=df, palette=["#4CAF50", "#F44336"], ax=axes[0, 0])
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].set_xlabel("Churn (0 = No, 1 = Yes)")

# Tenure Distribution
sns.histplot(df["Tenure"], bins=30, kde=True, color="#2196F3", ax=axes[0, 1])
axes[0, 1].set_title("Tenure Distribution")

# Monthly Charges Distribution
sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color="#9C27B0", ax=axes[0, 2])
axes[0, 2].set_title("Monthly Charges Distribution")

# -----------------------------------------------------
# ROW 2
# -----------------------------------------------------

# Total Charges Distribution
sns.histplot(df["TotalCharges"], bins=30, kde=True, color="#FF9800", ax=axes[1, 0])
axes[1, 0].set_title("Total Charges Distribution")

# Complaints Distribution
sns.histplot(df["ComplaintsCount"], bins=20, kde=False, color="#607D8B", ax=axes[1, 1])
axes[1, 1].set_title("Complaints Count Distribution")

# Tenure vs Churn
sns.boxplot(x="Churn", y="Tenure", data=df, palette=["#81C784", "#E57373"], ax=axes[1, 2])
axes[1, 2].set_title("Tenure vs Churn")

# -----------------------------------------------------
# ROW 3
# -----------------------------------------------------

# Monthly Charges vs Churn
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=["#64B5F6", "#FF8A65"], ax=axes[2, 0])
axes[2, 0].set_title("Monthly Charges vs Churn")

# Contract Type vs Churn
sns.countplot(
    x="ContractType",
    hue="Churn",
    data=df,
    palette=["#4DB6AC", "#EF5350"],
    ax=axes[2, 1]
)
axes[2, 1].set_title("Contract Type vs Churn")
axes[2, 1].tick_params(axis="x", rotation=15)
axes[2, 1].legend(title="Churn")

# Correlation Heatmap
num_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "ComplaintsCount", "Churn"]
sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap="RdYlBu",
    linewidths=0.4,
    ax=axes[2, 2]
)
axes[2, 2].set_title("Correlation Heatmap")

# -----------------------------------------------------
# ROW 4
# -----------------------------------------------------

# Tenure vs Monthly Charges (Churn Highlighted)
sns.scatterplot(
    x="Tenure",
    y="MonthlyCharges",
    hue="Churn",
    data=df,
    palette=["#2E7D32", "#C62828"],
    alpha=0.6,
    ax=axes[3, 0]
)
axes[3, 0].set_title("Tenure vs Monthly Charges (Churn)")

# Remove empty plots
axes[3, 1].axis("off")
axes[3, 2].axis("off")

# -----------------------------------------------------
# LAYOUT ADJUSTMENT
# -----------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Single-page EDA dashboard generated successfully 🚀")