import joblib
import pandas as pd


def test_model_prediction():
    model = joblib.load("models/champion_model.pkl")
    sample = pd.DataFrame([{
        "age": 63,
        "sex": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "ca": 0,
        "cp_2": 0,
        "cp_3": 0,
        "cp_4": 0,
        "restecg_1": 0,
        "restecg_2": 1,
        "thal_6.0": 1,
        "thal_7.0": 0,
        "slope_2": 0,
        "slope_3": 1
    }])
    pred = model.predict(sample)
    assert pred is not None
