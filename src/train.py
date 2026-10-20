import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import numpy as np

# Sample training dataset (replace with your real dataset)
df = pd.read_csv("sample_data/heart_disease_raw.csv")

# Numeric columns → fill missing with median
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# If any non-numeric missing remain → drop rows
df.dropna(inplace=True)

# -----------------------------
# 3. Encode target (binary classification)
# -----------------------------
# UCI 'num' column: 0 = no disease, 1–4 = disease
if "num" in df.columns:
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)

# -----------------------------
# 5. Final feature/target split
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "target" in numeric_cols:
    numeric_cols.remove("target")   # exclude target

# Identify all other columns (the one-hot encoded boolean/binary columns)
other_cols = [col for col in df.columns if col != "target" and col not in numeric_cols]

# Define the preprocessor explicitly mapping every feature to avoid internal remainder helpers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("passthrough", "passthrough", other_cols)
    ],
    remainder="drop"
)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ))
])

rf_cv = cross_validate(
    rf_clf, X, y,
    cv=5,
    scoring=["accuracy", "precision", "recall", "roc_auc", "f1"],
    return_train_score=False
)

# Fit model
rf_clf.fit(X_train, y_train)

joblib.dump(rf_clf, "models/champion_model.pkl")
