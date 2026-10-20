Heart Disease Prediction – End‑to‑End MLOps Pipeline
GitHub Repository: https://github.com/patilsharyurupchanad/mlops

A complete MLOps workflow for Heart Disease Prediction, covering EDA, model training, MLflow experiment tracking, FastAPI model serving, Docker containerization, Kubernetes deployment, monitoring (Prometheus + Grafana), CI/CD automation, and production‑readiness.

📁 Project Structure
Code
mlops/
│
├── api/
│   └── main.py
│
├── docker/
│   └── Dockerfile
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── models/
│   └── champion_model.pkl
│
├── notebooks/
│   └── assign1_heart_disease.ipynb
│
├── src/
│   └── train.py
│
├── test/
│   ├── test_api.py
│   ├── test_model.py
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── REPORT.pdf
⚙️ Installation & Setup (Step‑Wise with Brief Explanation)
Below is the complete setup workflow required to run the project from a clean environment.

1. Clone the Repository
Download the project to your local machine.

bash
git clone https://github.com/patilsharyurupchanad/mlops
cd mlops
Explanation:  
This pulls all code, notebooks, Dockerfiles, Kubernetes manifests, and CI/CD workflows.

2. Create a Python Environment
Use any environment tool (venv, conda, pyenv).

bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Explanation:  
Keeps dependencies isolated and avoids conflicts with system Python.

3. Install Dependencies
Install all required packages.

bash
pip install -r requirements.txt
Explanation:  
This installs FastAPI, Uvicorn, MLflow, scikit‑learn, pandas, Prometheus instrumentation, and all tools used in the pipeline.

4. Run the Notebook (EDA + Training + MLflow Tracking)
Open:

Code
notebooks/assign1_heart_disease.ipynb
Run all cells.

✔ The notebook will automatically:
Perform EDA

Train multiple models

Track experiments in MLflow

Select the champion model

Download champion_model.pkl

Download MLflow_export.zip

⚠️ Important:  
Enable multiple file downloads in your browser.

Explanation:  
This notebook generates the trained model and MLflow artifacts required for serving.

5. Run FastAPI Locally
Start the model inference API.

bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
Test endpoints:

Code
http://localhost:8000/health
http://localhost:8000/predict
Explanation:  
This serves the trained model via REST API.

6. Build Docker Image
Containerize the API.

bash
docker build -t heart-api:latest -f docker/Dockerfile .
Run container:

bash
docker run -p 8000:8000 heart-api:latest
Explanation:  
Ensures the model runs in an isolated, reproducible environment.

7. Deploy to Kubernetes (Docker Desktop Kubernetes)
Apply manifests:

bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
Port‑forward:

bash
kubectl port-forward svc/heart-api-service 8001:8000
Test:

Code
http://localhost:8001/health
http://localhost:8001/predict
Explanation:  
This deploys the API to a production‑like environment.

8. Enable Monitoring (Prometheus + Grafana)
Prometheus metrics:

Code
http://localhost:8001/metrics
Grafana dashboard:

Code
http://localhost:3000
Explanation:  
Provides real‑time monitoring of API performance and traffic.

9. CI/CD Pipeline (GitHub Actions)
Workflow file:

Code
.github/workflows/ci.yml
Pipeline includes:

Linting

Unit tests

Model training

Artifact upload

Build validation

Explanation:  
Ensures automated testing and reproducible builds.

📊 Notebook: EDA + Training + MLflow
The notebook performs:

Data cleaning

Exploratory Data Analysis

Feature engineering

Model training

MLflow experiment tracking

Champion model selection

Automatic export of model + MLflow artifacts

🚀 API Endpoints
GET /health
Health check endpoint.

POST /predict
Accepts JSON input and returns:

json
{
  "prediction": 0,
  "confidence": 0.87
}
🐳 Docker
Build:

bash
docker build -t heart-api:latest .
Run:

bash
docker run -p 8000:8000 heart-api:latest
☸️ Kubernetes
Deployment:

bash
kubectl apply -f k8s/deployment.yaml
Service:

bash
kubectl apply -f k8s/service.yaml
📈 Monitoring
Prometheus metrics:

Code
/metrics
Grafana dashboard includes:

Request count

Latency

/predict traffic

API performance

📸 Screenshots
All screenshots are stored in:

Code
screenshots/
Includes:

MLflow

CI/CD

Docker

Kubernetes

Prometheus

Grafana

API testing

📄 Final Report
A 10‑page professional report is included as:

Code
REPORT.pdf
