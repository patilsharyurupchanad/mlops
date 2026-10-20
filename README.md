# Heart Disease Prediction – End‑to‑End MLOps Pipeline

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/patilsharyurupchanad/mlops)

A complete MLOps workflow for Heart Disease Prediction, covering EDA, model training, MLflow experiment tracking, FastAPI model serving, Docker containerization, Kubernetes deployment, monitoring (Prometheus + Grafana), CI/CD automation, and production‑readiness.

---

## 📁 Project Structure

```text
mlops/
│
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD Pipeline
├── api/
│   └── main.py                 # FastAPI Application with Prometheus Tracking
├── docker/
│   └── Dockerfile              # Containerization Definition
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment Manifest
│   └── service.yaml            # Kubernetes Service Manifest
├── models/
│   └── champion_model.pkl      # Serialized Champion Model Artifact
├── notebooks/
│   └── assign1_heart_disease.ipynb  # Jupyter Notebook for EDA, Training & MLflow Tracking
├── src/
│   └── train.py                # Model Training Script
├── test/
│   ├── test_api.py             # API Unit Tests
│   └── test_model.py           # Model Framework Tests
├── screenshots/                # Pipeline Telemetry & UI Verification Images
├── requirements.txt            # Python Dependencies Spectrum
├── README.md                   # Project Documentation Index
└── REPORT.pdf                  # Comprehensive 10-Page Engineering Report
```

---

## ⚙️ Installation & Setup

Below is the step-by-step pipeline workflow required to bootstrap, deploy, and monitor the project from a clean machine environment.

### 1. Clone the Repository
Download the project source to your local workstation.
```bash
git clone https://github.com/patilsharyurupchanad/mlops.git
cd mlops
```
* **Explanation:** This pulls down all code files, local notebooks, configuration setups, Kubernetes manifests, and GitHub Action automation workflows.

### 2. Create a Python Environment
Initialize an isolated working environment using your preferred manager (`venv`, `conda`, or `pyenv`).
```bash
# Initialize the virtual environment
python -m venv venv

# Activate the workspace environment
source venv/bin/activate   # Mac / Linux
# OR
venv\Scripts\activate      # Windows
```
* **Explanation:** This keeps your pipeline dependencies self-contained and avoids system-wide Python package version conflicts.

### 3. Install Dependencies
Populate the active environment with the exact project requirements.
```bash
pip install -r requirements.txt
```
* **Explanation:** Installs core engineering tools including FastAPI, Uvicorn, MLflow, Scikit-Learn, Pandas, and the Prometheus FastAPI Instrumentator engine.

### 4. Run the Experimentation & Training Pipeline (EDA + Training + MLflow)

Launch your Jupyter notebook interface and open:
```text
notebooks/assign1_heart_disease.ipynb
```
Execute all cells sequentially. The notebook runtime will automatically complete detailed data preprocessing, candidate model testing, and evaluation metrics logging. 

Once complete, the notebook will trigger automated browser downloads for your top-performing `champion_model.pkl` file and an `MLflow_export.zip` archival tracking package.

> ⚠️ **Important Browser Configuration:** Ensure your browser permissions allow **multiple concurrent file downloads** so both files exit the runtime environment successfully.

#### Launching the MLflow Tracking UI Locally

To inspect your tracked experiment parameters, runs, loss metrics, and performance charts locally, follow these steps in a new WSL terminal window:

1. Navigate to the local directory where your browser saved the downloaded training files:
   ```bash
   cd Downloads
   ```
   *(Adjust this path if your Windows user profile downloads folders are mapped differently).*

2. Extract the downloaded MLflow run archive:
   ```bash
   unzip MLflow_export.zip -d MLflow_export
   ```

3. Move inside the extracted tracking directory:
   ```bash
   cd MLflow_export
   ```

4. Spin up the MLflow tracking dashboard server bound to all local network adapters:
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

5. Review your pipeline iterations, feature coefficients, and metrics charts directly within your host web browser at:
   ```text
   http://localhost:5000
   ```

**Explanation:** 
This pipeline stage isolates your high-performing machine learning artifacts and starts a local registry interface to systematically cross-reference your training runs before exporting files directly down into production container environments.

### 5. Launch FastAPI Locally
Spin up a local ASGI application worker instance to serve your model for real-time predictions:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Verify the operational health status and router endpoints using your browser or terminal utility:
* **Health Endpoint:** `http://localhost:8000/health`
* **Prediction Documentation UI:** `http://localhost:8000/docs`

### 6. Build and Verify the Docker Container Image
Containerize your prediction serving application layer to build an immutable environment artifact:
```bash
# Build the Docker image mapping from the root execution context
docker build -t heart-api:latest -f docker/Dockerfile .

# Instantiate the container locally mapping standard app ports
docker run -p 8001:8000 heart-api:latest
```
* **Explanation:** Isolates system dependencies, configurations, and core Python binaries inside a reproducible container wrapper layer.

### 7. Orchestrate Deployment inside Kubernetes
Apply the cloud-native deployment declarations to a working cluster (such as Docker Desktop Kubernetes, Minikube, or Kind):
```bash
# Deploy the container runtime specification pods
kubectl apply -f k8s/deployment.yaml

# Bind and expose the routing Service resource
kubectl apply -f k8s/service.yaml

# Tunnel traffic safely from your local machine adapter loopback into the service network
kubectl port-forward svc/heart-api-service 8002:8000 --address 0.0.0.0
```
Test your live, cluster-managed endpoints via your desktop interface or terminal shell:
* **Cluster Application Health:** `http://127.0.0.1:8002/health`
* **Interactive OpenAPI Swagger Docs UI:** `http://127.0.0.1:8002/docs`

### 8. Access Integrated Infrastructure Monitoring
Verify that runtime metrics engine integration captures cluster transaction activity:
* **Raw Prometheus Text Stream Metrics:** `http://127.0.0.1:8002/metrics`
* **Visual Telemetry Dashboard Experience:** Access Grafana at `http://localhost:3000`

The system collects, aggregates, and visualizes live traffic signals including **Total Request Volumes**, **Latency Quantiles**, **/predict Endpoint Hit Rates**, and **Worker Node Engine Performance**.

### 9. Automated CI/CD Pipeline Tracking
Continuous Integration and Delivery flows run automatically via the configuration rule file located at `.github/workflows/ci.yml`. Every code push triggers a structured sequence:
* Code linting verification checks.
* Python unit and contract tests execution (`pytest`).
* Automated validation test iterations across the data model layers.
* Staged artifact uploads and container engine verification checks.

---

## 🚀 API Endpoint Schema

### `GET /health`
* **Description:** Basic service availability confirmation routing.
* **Response Body (JSON):**
  ```json
  {
    "status": "ok",
    "message": "API is healthy"
  }
  ```

### `POST /predict`
* **Description:** Evaluates 18 input features to predict heart disease risk metrics.
* **Request Body Example (JSON containing engineered features):**
  ```json
  {
    "age": 63, "sex": 1, "trestbps": 145, "chol": 233, "fbs": 1, "thalach": 150, 
    "exang": 0, "oldpeak": 2.3, "ca": 0, "cp_2": 0, "cp_3": 0, "cp_4": 0, 
    "restecg_1": 0, "restecg_2": 1, "thal_6.0": 0, "thal_7.0": 0, "slope_2": 0, "slope_3": 1
  }
  ```
* **Response Body Example (JSON):**
  ```json
  {
    "prediction": 1,
    "confidence": 0.87
  }
  ```

---

## 📸 Architectural Telemetry and Reports
* **Pipeline Artifact Screenshots:** All visual proofs, covering MLflow experiment runs, active GitHub Actions, active Docker run status, active Kubernetes components, and live Prometheus/Grafana monitors are cataloged inside the `/screenshots` directory.
* **Technical Design Documentation Report:** A publication-grade, detailed 10-page structural analysis breakdown is compiled as **`REPORT.pdf`**.
