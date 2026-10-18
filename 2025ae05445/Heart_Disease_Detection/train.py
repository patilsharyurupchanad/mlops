import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Sample training dataset (replace with your real dataset)
data = pd.read_csv("sample_data/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/champion_model.pkl")

print("Model trained and saved to models/champion_model.pkl")
