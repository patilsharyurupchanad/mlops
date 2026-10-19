import joblib
import pandas as pd

#age  sex  trestbps  chol  fbs  thalach  exang  oldpeak   ca  target   cp_2  \
#63    1       145   233    1      150      0      2.3  0.0       0  False
#cp_3   cp_4  restecg_1  restecg_2  thal_6.0  thal_7.0  slope_2  slope_3
#False  False      False       True      True     False    False     True 

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
        "thal_6.0": 1
        "thal_7.0": 0
        "slope_2": 0,
        "slope_3": 1,
    }])
    pred = model.predict(sample)
    assert pred is not None
