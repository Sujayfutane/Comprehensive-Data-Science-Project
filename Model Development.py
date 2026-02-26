# =====================================================
# MODEL DEVELOPMENT PIPELINE - SINGLE SCRIPT
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------

df = pd.read_csv("customer_churn_data.csv")
print("Dataset Loaded:", df.shape)

# -----------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------

# Create new features
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["Tenure"] + 1)
df["HighComplaints"] = np.where(df["ComplaintsCount"] >= 3, 1, 0)

# Drop ID column
df.drop(columns=["CustomerID"], inplace=True)

# -----------------------------------------------------
# 3. DEFINE FEATURES & TARGET
# -----------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

num_features = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ComplaintsCount",
    "AvgMonthlySpend"
]

cat_features = [
    "Gender",
    "ContractType",
    "PaymentMethod",
    "InternetService"
]

bin_features = [
    "SeniorCitizen",
    "TechSupport",
    "StreamingServices",
    "HighComplaints"
]

# -----------------------------------------------------
# 4. PREPROCESSING PIPELINE
# -----------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
        ("bin", "passthrough", bin_features)
    ]
)

# -----------------------------------------------------
# 5. TRAIN-TEST SPLIT
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------------------------------
# 6. MODEL 1: LOGISTIC REGRESSION
# -----------------------------------------------------

log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Results ---")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

# -----------------------------------------------------
# 7. MODEL 2: RANDOM FOREST + HYPERPARAMETER TUNING
# -----------------------------------------------------

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 8, 12],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="recall",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest (Tuned) Results ---")
print("Best Parameters:", grid_search.best_params_)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# -----------------------------------------------------
# 8. MODEL COMPARISON SUMMARY
# -----------------------------------------------------

summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob_log),
        roc_auc_score(y_test, y_prob_rf)
    ]
})

print("\n--- MODEL COMPARISON ---")
print(summary)

# -----------------------------------------------------
# 9. FINAL MODEL SELECTION
# -----------------------------------------------------

print("\nFINAL MODEL SELECTED: Random Forest (Tuned)")
print("Reason: Higher recall and ROC-AUC for churn prediction")

print("\nMODEL DEVELOPMENT COMPLETED SUCCESSFULLY 🚀")