from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "champion_model.pkl"
)

model = joblib.load(MODEL_PATH)

# The exact 18-column feature schema expected by the trained model pipeline
MODEL_INPUT_FEATURES = [
    "age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak",
    "ca", "cp_2", "cp_3", "cp_4", "restecg_1", "restecg_2", "thal_6.0",
    "thal_7.0", "slope_2", "slope_3"
]

app = FastAPI(title="Heart Disease Detection")

@app.get("/health")


def health():
    return {"status": "ok", "message": "API is healthy"}

class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")


def predict(data: InputData):
    # 1. Convert input JSON data to a pandas DataFrame
    raw_df = pd.DataFrame([data.dict()])
    
    # 2. Re-apply the exact same get_dummies logic
    # as done in the notebook training
    categorical_cols = ["cp", "restecg", "thal", "slope"]
    processed_df = pd.get_dummies(raw_df, columns=categorical_cols,
                                  drop_first=False)
    
    # In python pandas dummy names would look like 'cp_1', 'cp_2',
    # or 'thal_6.0' We need to guarantee 'thal' columns keep float
    # representations in string matching
    processed_df.columns = [
            col.replace(".0", ".0") for col in processed_df.columns]

    # 3. Align Columns: Reindex to the 18 expected model features,
    # setting absent columns to False
    for col in MODEL_INPUT_FEATURES:
        if col not in processed_df.columns:
            processed_df[col] = False
            
    # Ensure features are in the exact structural order expected
    # by the model pipeline
    aligned_df = processed_df[MODEL_INPUT_FEATURES]
    
    # 4. Predict and return prediction
    prediction = model.predict(aligned_df)
    return {"prediction": int(prediction[0])}
