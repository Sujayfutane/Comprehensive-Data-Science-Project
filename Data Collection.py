# =====================================================
# DATA COLLECTION PIPELINE - SINGLE SCRIPT
# =====================================================

import pandas as pd
import numpy as np

# -----------------------------------------------------
# 1. SIMULATE CUSTOMER CHURN DATA
# -----------------------------------------------------

def simulate_customer_data(n_customers=8000, random_state=42):
    np.random.seed(random_state)

    df = pd.DataFrame({
        "CustomerID": [f"CUST_{i+1}" for i in range(n_customers)],
        "Gender": np.random.choice(["Male", "Female"], n_customers),
        "SeniorCitizen": np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
        "Tenure": np.random.randint(0, 72, n_customers),
        "MonthlyCharges": np.round(np.random.uniform(20, 120, n_customers), 2),
        "ContractType": np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            n_customers,
            p=[0.6, 0.25, 0.15]
        ),
        "PaymentMethod": np.random.choice(
            ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
            n_customers
        ),
        "InternetService": np.random.choice(
            ["DSL", "Fiber optic", "None"],
            n_customers,
            p=[0.35, 0.45, 0.20]
        ),
        "TechSupport": np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        "StreamingServices": np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        "ComplaintsCount": np.random.poisson(lam=1.2, size=n_customers)
    })

    # Total charges
    df["TotalCharges"] = np.round(df["Tenure"] * df["MonthlyCharges"], 2)

    # Churn logic (business-inspired)
    churn_score = (
        0.30 * (df["ContractType"] == "Month-to-month").astype(int) +
        0.25 * (df["Tenure"] < 12).astype(int) +
        0.20 * (df["ComplaintsCount"] > 2).astype(int) +
        np.random.rand(n_customers) * 0.2
    )

    df["Churn"] = np.where(churn_score > 0.5, 1, 0)

    return df


# -----------------------------------------------------
# 2. CREATE DATASET
# -----------------------------------------------------

df = simulate_customer_data()

print("Dataset Shape:", df.shape)
print(df.head())


# -----------------------------------------------------
# 3. DATA DICTIONARY
# -----------------------------------------------------

data_dictionary = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.values,
    "Description": [
        "Unique customer identifier",
        "Customer gender",
        "Senior citizen indicator (1 = Yes, 0 = No)",
        "Number of months customer has stayed",
        "Monthly subscription charges",
        "Type of contract",
        "Payment method used",
        "Internet service type",
        "Availability of technical support",
        "Use of streaming services",
        "Number of customer complaints",
        "Total amount billed to customer",
        "Churn flag (Target Variable)"
    ]
})

print("\nData Dictionary:")
print(data_dictionary)


# -----------------------------------------------------
# 4. DATA QUALITY VALIDATION
# -----------------------------------------------------

print("\n--- DATA QUALITY CHECKS ---")

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

# Data type info
print("\nData Types:")
print(df.info())

# Logical checks
assert (df["MonthlyCharges"] > 0).all(), "Invalid MonthlyCharges detected"
assert (df["Tenure"] >= 0).all(), "Invalid Tenure detected"
assert (df["TotalCharges"] >= 0).all(), "Invalid TotalCharges detected"

print("\nLogical validation passed ✔")


# -----------------------------------------------------
# 5. OUTLIER DETECTION (IQR METHOD)
# -----------------------------------------------------

def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((series < lower) | (series > upper)).sum()

print("\nOutliers Count:")
print("MonthlyCharges:", count_outliers(df["MonthlyCharges"]))
print("TotalCharges:", count_outliers(df["TotalCharges"]))


# -----------------------------------------------------
# 6. TARGET VARIABLE VALIDATION
# -----------------------------------------------------

churn_distribution = df["Churn"].value_counts(normalize=True) * 100
print("\nChurn Distribution (%):")
print(churn_distribution)


# -----------------------------------------------------
# 7. SAVE OUTPUT FILES
# -----------------------------------------------------

df.to_csv("customer_churn_data.csv", index=False)
data_dictionary.to_csv("data_dictionary.csv", index=False)

print("\nFiles Saved:")
print("✔ customer_churn_data.csv")
print("✔ data_dictionary.csv")

print("\nDATA COLLECTION PHASE COMPLETED SUCCESSFULLY 🚀")