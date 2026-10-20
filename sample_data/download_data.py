from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Extract combined layout
df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)

#
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
#

# -----------------------------
# 4. Encode categorical features
# -----------------------------
# These columns are categorical in UCI Heart dataset
categorical_cols = ["cp", "restecg", "thal", "slope"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.to_csv("sample_data/heart_disease_raw.csv", index=False)
