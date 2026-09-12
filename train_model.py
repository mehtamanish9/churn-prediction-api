"""
Trains a Logistic Regression churn model on the REAL Telco Customer Churn
dataset (loaded from telco.db, a SQLite database with a `customers` table)
and saves the model for the API to load.

This mirrors the original Customer Churn Prediction project: data cleaning,
feature engineering, one-hot encoding, class-imbalance handling, and
Logistic Regression, evaluated with ROC-AUC.
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- 1. Load real data from telco.db ----
conn = sqlite3.connect("telco.db")
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from telco.db")

# ---- 2. Clean + encode target ----
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Keep the same feature set the API's app.py expects: tenure, MonthlyCharges,
# Contract, InternetService, TechSupport, PaperlessBilling (+ engineered TotalCharges)
FEATURES_RAW = ["tenure", "MonthlyCharges", "Contract", "InternetService", "TechSupport", "PaperlessBilling"]

model_df = df[FEATURES_RAW + ["Churn"]].copy()

# ---- 3. Feature engineering (same style as original project) ----
model_df["TotalCharges"] = model_df["tenure"] * model_df["MonthlyCharges"]
model_df = pd.get_dummies(
    model_df,
    columns=["Contract", "InternetService", "TechSupport", "PaperlessBilling"],
    drop_first=True,
)

FEATURE_COLUMNS = [c for c in model_df.columns if c != "Churn"]

X = model_df[FEATURE_COLUMNS]
y = model_df["Churn"]

churn_rate = y.mean()
print(f"Churn rate: {churn_rate:.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"Validation ROC-AUC: {auc:.3f}")

# ---- 4. Save model + scaler + column order for the API ----
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

print("Saved model.pkl, scaler.pkl, feature_columns.pkl")
print("Feature columns:", FEATURE_COLUMNS)
