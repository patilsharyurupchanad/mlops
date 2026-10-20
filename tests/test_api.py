from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    sample = {
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
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
