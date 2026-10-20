from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import logging
from prometheus_fastapi_instrumentator import Instrumentator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("heart-api")


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
    trestbps: int
    chol: int
    fbs: int
    thalach: int
    exang: int
    oldpeak: float
    ca: int
    cp_2: int
    cp_3: int
    cp_4: int
    restecg_1: int
    restecg_2: int
    thal_6_0: float = Field(alias="thal_6.0")
    thal_7_0: float = Field(alias="thal_7.0")
    slope_2: int
    slope_3: int


@app.post("/predict")
def predict(data: InputData):

    logger.info(f"Incoming /predict request: {data.dict()}")

    # 1. Convert input JSON data to a pandas DataFrame
    raw_df = pd.DataFrame([data.model_dump()], columns=MODEL_INPUT_FEATURES)
    # 4. Predict and return prediction
    prediction = model.predict(raw_df)[0]

    # Get confidence score of the predicted class
    probabilities = model.predict_proba(raw_df)[0]
    confidence = float(probabilities[prediction])

    logger.info(
        f"Prediction result: prediction={int(prediction)}, \
            confidence={confidence}"
    )

    return {
        "prediction": int(prediction),
        "confidence": confidence
    }


Instrumentator().instrument(app).expose(app, endpoint="/metrics")
