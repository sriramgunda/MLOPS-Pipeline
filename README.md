# Cats vs Dogs Binary Image Classification - MLOps Pipeline

## Overview

MLOps pipeline for binary image classification (Cats vs Dogs). Includes model training, data versioning, containerization, CI/CD, and monitoring on Kubernetes.

NOTE: This project uses DVC only for dataset versioning and tracking pre-processed data (download, extraction, organization). Model training and hyperparameter experimentation are run outside DVC (see "Train model" step). Do not track model parameters in DVC.

## Architecture Diagram

```
Dataset -> Data Versioning (DVC) -> Data Preprocessing (224x224 RGB)
    |
Model Training (CNN with MLflow) -> Experiment Tracking
    |
Model Artifact Storage -> Docker Containerization
    |
CI Pipeline (GitHub Actions) -> Unit Tests -> Docker Build -> Registry Push
    |
CD Pipeline (GitOps) -> Kubernetes Deployment
    |
Monitoring (Prometheus + Grafana) -> Logging & Metrics -> Model Performance Tracking
```

## Quick Start

For detailed setup instructions, see [MLOPS_setup.md](documentation/MLOPS_setup.md).

Basic startup commands:
```bash
# Setup
python setup_mlops.py

# Download dataset
python src/data_loader.py

# Train model
python src/train_cnn.py

# Start API
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

These steps helps to monitor model performance trends, inspect prediction distributions, and surface regressions (drift/accuracy drops) directly in Grafana.

### Common Commands Reference
| Task | Command |
|------|---------|
| Download dataset | `python src/data_loader.py` |
| Train model | `python src/train_cnn.py` |
| Make prediction | `python src/predict_image.py --image path/to/image.jpg` |
| Run tests | `pytest tests/ -v` |
| Start API | `uvicorn app.main:app --reload --port 8000` |
| MLflow UI | `mlflow ui --port 5000` |
| Deploy K8s | `kubectl apply -f k8s/` |
| Smoke tests | `python scripts/smoke_test.py` |

## MLOps Setup & Configuration

### Quick MLOps Initialization

```bash
# Initialize MLOps infrastructure (DVC + MLflow)
python setup_mlops.py

# Or manually initialize:
dvc init
mlflow ui  # Start experiment tracking UI on http://localhost:5000
```

### Run Complete Pipeline with DVC (dataset & preprocessing only)

This repository uses DVC to version datasets and pre-processed artifacts (download, extraction, organization). Model training is not part of the DVC pipeline and should be executed separately.

You can run the dataset pipeline with the included CLI or with DVC directly:

```bash
# Initialize and run with the helper CLI (recommended)
python -m src.run_dvc --init        # initialize DVC (one-time)
python -m src.run_dvc               # reproduce dataset/preprocessing stages

# Or use DVC directly (three stages tracked):
dvc repro                            # runs data_download, data_extraction, data_organization

# Check pipeline status and visualize
dvc status
dvc dag
```

### Monitor Experiments in MLflow UI

```bash
# MLflow runs automatically during training
# Access at http://localhost:5000

# View metrics:
# - Training loss/accuracy
# - Validation loss/accuracy
# - Test loss/accuracy

# Download artifacts:
# - Trained model (.keras format)
# - Training history
# - Configuration used
```

### Configuration Management

Update hyperparameters in `params.yaml`:
```yaml
epochs: 1              # Number of training epochs
batch_size: 32        # Training batch size
learning_rate: 0.001  # Optimizer learning rate

# Dataset paths
data:
  train_dir: "data/train"        # 80% of data
  val_dir: "data/validation"     # 10% of data
  test_dir: "data/test"          # 10% of data

# Model configuration
model:
  architecture: "mobilenet_v2"   # Using MobileNetV2 transfer learning
  pretrained: true               # Use ImageNet weights
  freeze_base: true              # Freeze base model
  dropout: 0.3                   # Dropout regularization
  img_size: [224, 224]          # Input image size
```

After modifying parameters, DVC will re-run only affected stages:
```bash
dvc repro  # Automatically re-runs necessary stages
```

For complete MLOps documentation, see [MLOPS_setup.md](documentation/MLOPS_setup.md).

## Project Structure

```
.
├── app/
│   └── main.py                 # FastAPI inference service with 2 endpoints
├── src/
│   ├── data_loader.py          # Dataset downloading and organization (80/10/10 split)
│   ├── data_preprocessing.py   # Image preprocessing, augmentation, normalization
│   ├── train_cnn.py           # CNN model training with MLflow tracking
│   ├── train_automl.py         # Legacy AutoML training (can be repurposed)
│   ├── predict.py              # Inference script
│   ├── mlflow_config.py        # MLflow initialization and utilities
│   ├── dvc_utils.py            # DVC pipeline utilities
│   └── CatDog.ipynb            # Interactive notebook with complete pipeline
├── tests/
│   ├── test_data_processing.py # Unit tests for image preprocessing
│   └── test_model_training.py  # Unit tests for model inference
├── k8s/
│   ├── deployment.yaml         # Kubernetes deployment manifest
│   ├── service.yaml            # Kubernetes service definition
│   └── servicemonitor.yaml     # Prometheus monitoring configuration
├── scripts/
│   └── smoke_test.py           # Post-deployment smoke tests
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── documentation/              # Documentation and reports
├── dvc.yaml                    # DVC pipeline definition (4 stages)
├── params.yaml                 # Configuration parameters (epochs, model, data paths)
├── .dvcignore                  # DVC ignore patterns
├── .mlflowconfig               # MLflow server configuration
├── .gitignore                  # Git ignore patterns (DVC cache, MLflow, models)
├── docker-compose.yml          # Docker Compose for local development
├── Dockerfile                  # Application container image
├── requirements.txt            # Python dependencies with pinned versions
├── prometheus.yml              # Prometheus scrape configuration
├── setup_mlops.py              # MLOps initialization script
├── setup.sh                    # Environment setup script
└── README.md                   # This file
```

## Screenshots

MLOps pipeline implementation showcased across multiple modules:

### Experiment Tracking (MLflow)
- ![MLflow UI](documentation/screenshots/experiment-tracking/mlflow-ui.png) - Experiment overview and run tracking
- ![Model Metrics](documentation/screenshots/experiment-tracking/model-metrics.png) - Metrics comparison
- ![Model Overview](documentation/screenshots/experiment-tracking/model-overview.png) - Model details
- ![Model Versioning](documentation/screenshots/experiment-tracking/model-versioninig.png) - Version management
- ![Models Tracked](documentation/screenshots/experiment-tracking/models-tracked.png) - All tracked models
- ![Model Metadata 1](documentation/screenshots/experiment-tracking/model-metadata-1.png) - Metadata details (part 1)
- ![Model Metadata 2](documentation/screenshots/experiment-tracking/model-metadata-2.png) - Metadata details (part 2)

### CI Pipeline (GitHub Actions)
- ![Install Dependencies](documentation/screenshots/CI/install-dependencies.png) - Dependency installation
- ![Unit Tests](documentation/screenshots/CI/unit-tests.png) - Test execution (tests/)
- ![Docker Build & Push](documentation/screenshots/CI/docker-image-build-push.png) - Image building and registry push
- ![GitHub Actions](documentation/screenshots/CI/github-actions.png) - Workflow execution
- ![CI Stages](documentation/screenshots/CI/ci-stages.png) - Pipeline stages overview
- ![DVC & Preprocessing](documentation/screenshots/CI/dvc-and-preprocessing.png) - Data processing in CI
- ![Experiment Tracking](documentation/screenshots/CI/experiment-tracking.png) - Tracking in pipeline
- ![Model Training](documentation/screenshots/CI/model-training.png) - Training execution
- ![MLflow Artifacts](documentation/screenshots/CI/mlflow-artifacts-uploaded.png) - Artifact uploads

### CD Pipeline (Kubernetes Deployment)
- ![Kubernetes Deployment](documentation/screenshots/CD/kubernetes-deployment-pods.png) - Pod deployment status
- ![Kubernetes Service](documentation/screenshots/CD/kubernetes-service.png) - Service configuration
- ![Pod Logs](documentation/screenshots/CD/pod-logs.png) - Application logs
- ![Smoke Tests](documentation/screenshots/CD/smoke-test-results.png) - Health check results (scripts/smoke_test.py)
- ![Health Check](documentation/screenshots/CD/app-api-healthcheck.png) - API health endpoint
- ![Health Response](documentation/screenshots/CD/app-api-healthcheck-response.png) - Health response details
- ![Prediction Request](documentation/screenshots/CD/app-api-predict-request-response.png) - Model prediction example
- ![Prediction Response](documentation/screenshots/CD/app-api-predict-request-response1.png) - Alternative prediction example

### Monitoring (Prometheus & Grafana)
- ![Prometheus](documentation/screenshots/monitoring/prometheus.png) - Metrics collection and queries
- ![Grafana](documentation/screenshots/monitoring/grafana.png) - Dashboard visualization
- ![App Metrics](documentation/screenshots/monitoring/app-metrics.png) - Application-level metrics

---

## Module Breakdown

### M1: Model Development & Experiment Tracking

**Objective**: Build a baseline model, track experiments, and version all artifacts.

#### 1. Data & Code Versioning

**Git**: Source code versioning for all scripts and notebooks
```bash
git init
git add .
git commit -m "Initial MLOps pipeline"
git log --oneline  # View commit history
```

---

## Data Versioning Options: DVC vs Git LFS

Choose one approach based on your workflow:

### Option A: DVC (Recommended for MLOps with Pipelines)

**Setup**:
```bash
# Initialize DVC (one-time setup)
dvc init

# Pipeline automatically configured in dvc.yaml with 4 stages:
# 1. data_download  - Download from Kaggle
# 2. data_extraction - Extract ZIP file
# 3. data_organization - Split into 80/10/10 (train/val/test)
# 4. model_training - Train MobileNetV2 with MLflow tracking

# Run entire pipeline
dvc repro

# Push to DVC remote storage (optional)
dvc remote add -d myremote /path/to/dvc-storage
dvc push

# Later: Pull latest dataset version
dvc pull

# View dataset history & pipelines
dvc dag
dvc status
```

**DVC Pipeline Stages** (defined in `dvc.yaml`):

```yaml
stages:
  data_download:
    cmd: python src/data_loader.py download
    deps:
      - src/data_loader.py
      - src/kaggle.json
    outs:
      - data/cat-and-dog.zip
    
  data_extraction:
    cmd: python src/data_loader.py extract
    deps:
      - data/cat-and-dog.zip
    outs:
      - data/training_set
      - data/test_set
    
  data_organization:
    cmd: python src/data_loader.py organize
    deps:
      - data/training_set
      - data/test_set
    outs:
      - data/train  # 80%
      - data/validation  # 10%
      - data/test  # 10%
    
  # Note: model_training stage removed — model training is executed outside DVC
```

**Data Directory Structure**:
```
data/
├── cat-and-dog.zip           # Downloaded from Kaggle
├── training_set/             # Extracted training (includes cats & dogs)
├── test_set/                 # Extracted test (includes cats & dogs)
├── train/                    # 80% of all data
│   ├── cats/
│   └── dogs/
├── validation/               # 10% of all data
│   ├── cats/
│   └── dogs/
└── test/                     # 10% of all data
    ├── cats/
    └── dogs/
```

**Advantages**:
- Supports pipeline orchestration (`dvc.yaml` stages)
- Supports parameter tracking (`params.yaml`)
- Supports experiment comparison (`dvc metrics diff`, `dvc params diff`)
- Works with S3, GCS, Azure, local storage
- Lightweight metadata (`.dvc` files in Git)
- Native MLOps features
- Automatic reproducibility with `dvc repro`

Setup in `.dvc/config`. Remote: `./dvc-storage` (local) or S3/GCS/Azure.

---

### Option B: Git LFS (Lightweight for Simple Workflows)

**Install Git LFS** (one-time setup):
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows
choco install git-lfs
# Or download from https://git-lfs.github.com/

# Initialize Git LFS in repository
git lfs install
```

**Setup**:
```bash
# Download dataset (creates data/train, data/validation, data/test)
python src/data_loader.py

# Track large files with Git LFS
git lfs track "data/**/*.jpg"
git lfs track "data/**/*.png"
git add .gitattributes

# Commit everything normally
git add data/
git commit -m "Add cats and dogs dataset (80/10/10 split)"
git push origin main

# Subsequent pulls download LFS files automatically
git clone <repo-url>
git lfs pull
```

**Advantages**:
- Offers simple Git workflow (no new tool to learn)
- Transparent - works like normal Git
- No metadata files (direct file versioning)
- Good for simple dataset versioning
- Works with GitHub, GitLab, Bitbucket

**Limitations**:
- No pipeline orchestration
- No parameter tracking
- No built-in experiment comparison
- GitHub free tier: 1GB bandwidth/month, pay for more

---

### Comparison Table

| Feature | DVC | Git LFS |
|---------|-----|---------|
| **Learning Curve** | Medium | Low |
| **Pipeline Orchestration** | Supported | Not supported |
| **Parameters Tracking** | Supported | Not supported |
| **Experiment Comparison** | Supported | Not supported |

---

#### 2. Model Building
- **Architecture**: MobileNetV2 transfer learning with frozen base + custom head
- **Base Model**: ImageNet pre-trained weights
- **Custom Head**: GlobalAveragePooling2D -> Dropout(0.3) -> Dense(1, sigmoid)
- **Input Size**: 224×224 RGB images
- **Model Format**: `.keras` (TensorFlow native format)
- **Output**: Binary classification (Cats vs Dogs)

#### 3. Experiment Tracking with MLflow

**Model Training & Comparison**:

The pipeline trains both a baseline CNN model and a transfer learning model (MobileNetV2), tracks all experiments in MLflow, and automatically selects the best performing model.

**Training Process**:
1. **Baseline CNN Model** - Custom CNN trained from scratch
   - Tracked as separate MLflow run
   - Logs: train/val/test loss and accuracy
   
2. **MobileNetV2 Model** - Transfer learning with ImageNet pre-trained weights
   - Tracked as separate MLflow run  
   - Logs: train/val/test loss and accuracy, pretrained weights info
   
3. **Model Comparison** - Both models are evaluated on test set
   - Baseline test accuracy vs MobileNetV2 test accuracy
   - Accuracy difference and improvement percentage logged
   - Best model selected based on highest test accuracy
   
4. **Best Model Selection** - Automatically saved as `models/best_model.keras`
   - If MobileNetV2 test accuracy >= Baseline test accuracy then MobileNetV2 is selected
   - Otherwise: Baseline CNN is selected
   - Selected model is saved and used for inference

**MLflow Tracking Details**:
- **Run ID**: Unique identifier for each training run
- **Parameters**: epochs, batch_size, learning_rate, model_name, seed, architecture type
- **Metrics**: 
  - train_loss, train_accuracy
  - val_loss, val_accuracy  
  - test_loss, test_accuracy
  - accuracy_difference, improvement_percentage
- **Artifacts**: 
  - Model weights (.keras format)
  - Training history
  - Loss curves visualization
  - Confusion matrix
  - Classification report

**Access MLflow UI**:
```bash
# Start MLflow server
mlflow ui --port 5000
# Visit http://localhost:5000
```

**Viewing Results in MLflow**:
1. Navigate to http://localhost:5000
2. Select the **"cat_dog_classification"** experiment
3. View individual runs for:
   - `baseline_cnn_YYYYMMDD_HHMMSS` - Baseline model metrics
   - `mobilenet_v2_YYYYMMDD_HHMMSS` - Transfer learning model metrics
   - `model_comparison_YYYYMMDD_HHMMSS` - Comparison summary
4. Compare runs using the compare button to see side-by-side metrics
5. Download artifacts (model, visualizations) from each run

**Training Integration**:
```python
# Training automatically integrates MLflow (see src/train_cnn.py)
from src.train_cnn import train_cnn_pipeline

results = train_cnn_pipeline(
    train_ds, val_ds, test_ds,
    data_augmentation=augmentation,
    epochs=1  # Configurable via params.yaml
)

# results contains:
# - best_model: the selected model
# - best_model_name: 'baseline_cnn' or 'mobilenet_v2'
# - baseline/mobilenet metrics and histories
# - comparison results with accuracy improvements
```

---

### M2: Model Packaging & Containerization

#### 1. Inference Service (FastAPI)

**Endpoints**:

1. **Health Check** (`GET /health`)
   ```bash
   curl http://localhost:8000/health
   ```
   Response:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "timestamp": "2024-01-20T10:30:45Z"
   }
   ```

2. **Prediction** (`POST /predict`)
   ```bash
   curl -X POST \
     -F "file=@cat.jpg" \
     http://localhost:8000/predict
   ```
   Response:
   ```json
   {
     "prediction": "cat",
     "confidence": 0.9234,
     "class_probabilities": {
       "cat": 0.9234,
       "dog": 0.0766
     },
     "processing_time_ms": 145.23
   }
   ```

3. **Base64 Prediction** (`POST /predict-base64`)
   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "iVBORw0KGgoAAAANSU..."}' \
     http://localhost:8000/predict-base64
   ```

4. **Metrics** (`GET /metrics`)
   - Prometheus-compatible metrics endpoint

#### 2. Environment Specification

**requirements.txt** with pinned versions:
```
tensorflow==2.14.0
keras==2.14.0
fastapi==0.103.0
uvicorn==0.23.2
mlflow==2.7.1
prometheus-client==0.17.1
pytest==7.4.2
```

All dependencies are pinned to exact versions for reproducibility.

#### 3. Containerization

**Build Image**:
```bash
docker build -t cats-dogs-classifier:latest .
```

**Run Container**:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  cats-dogs-classifier:latest
```

**Verify Predictions**:
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

---

### M3: CI Pipeline for Build, Test & Image Creation

#### 1. Automated Testing

**Data Preprocessing Tests** (`tests/test_data_processing.py`):
- Image loading and resizing to 224x224
- Pixel value normalization [0, 1]
- Data augmentation (flip, rotate, brightness)
- Batch preparation validation

**Inference Tests** (`tests/test_model_training.py`):
- Model prediction output bounds [0, 1]
- Batch prediction handling
- Input validation
- Class probability summing to 1

**Run Tests Locally**:
```bash
pytest tests/ -v --cov=src --cov=app
```

#### 2. CI Pipeline (GitHub Actions)

**Workflow**: `.github/workflows/ci-cd.yml`

**Trigger**: On every `push` to `main`/`develop` and `pull_request`

**Steps**:
1. **Setup**: Checkout code, install Python 3.10
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Lint**: flake8 for code quality
4. **Test**: pytest with coverage reporting
5. **Build Docker Image**: Multi-stage build for optimization
6. **Push to Registry**: GitHub Container Registry (ghcr.io)

**Run Pipeline**:
```bash
git push origin main
# Automatically triggers CI workflow
```

#### 3. Artifact Publishing

**Docker Registry Options**:
- **GitHub Container Registry** (ghcr.io) - Integrated with GitHub
- **Docker Hub** - Public/private repositories
- **Local Registry** - For air-gapped environments

**Push to GHCR**:
```bash
echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
docker tag cats-dogs-classifier:latest ghcr.io/username/cats-dogs-classifier:latest
docker push ghcr.io/username/cats-dogs-classifier:latest
```

---

### M4: CD Pipeline & Deployment

#### 1. Deployment Target

**Option A: Kubernetes (Local Minikube)**

Setup:
```bash
minikube start --cpus=4 --memory=4096
minikube docker-env  # Use Minikube's Docker daemon
```

Apply Manifests:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml
```

Verify Status:
```bash
kubectl get deployments
kubectl get pods
kubectl get svc
```

Access Service:
```bash
minikube service cats-dogs-service
# Or use port-forward
kubectl port-forward svc/cats-dogs-service 8000:80
```

**Option B: Docker Compose (Local Development)**

```bash
docker-compose -f docker-compose.yml up -d
# Services: API, MLflow, Prometheus, Grafana
```

#### 2. CD / GitOps Flow

**GitHub Actions Deployment** (on main branch push):
```yaml
deploy:
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  steps:
    - kubectl apply -f k8s/deployment.yaml
    - kubectl rollout status deployment/cats-dogs-deployment
```

**Alternative: Argo CD** (for true GitOps):
```bash
argocd app create cats-dogs \
  --repo https://github.com/username/mlops-pipeline \
  --path k8s \
  --dest-server https://kubernetes.default.svc
```

#### 3. Smoke Tests / Health Checks

**Health Endpoint Check**:
```bash
curl http://localhost:8000/health
# Returns {"status": "healthy", "model_loaded": true}
```

**Prediction Smoke Test**:
```bash
python scripts/smoke_test.py --api-url http://localhost:8000
```

**Automated Smoke Tests in CI**:
```yaml
smoke-test:
  runs-on: ubuntu-latest
  steps:
    - run: python -m pytest scripts/smoke_test.py -v
```

---

### M5: Monitoring, Logs & Final Submission

#### 1. Request/Response Logging

**In FastAPI**:
```python
@app.middleware("http")
async def log_requests_and_metrics(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"status_code={response.status_code}")
    return response
```

**Log Format**:
```
2024-01-20 10:30:45 - INFO - POST /predict status_code=200 duration=0.145s
```

#### 2. Metrics Collection

**Prometheus Metrics**:

```python
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency",
    ["endpoint"]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence scores",
    ["class_label"]
)
```

**View Metrics**:
```bash
curl http://localhost:8000/metrics
```

**Prometheus Dashboard**:
```bash
# http://localhost:9090
# Queries: 
# - rate(api_requests_total[5m])
# - histogram_quantile(0.99, api_request_latency_seconds)
```

#### 3. Grafana Dashboards

**Setup**:
```bash
# Default credentials: admin/admin
# http://localhost:3000
```

**Dashboard Panels**:
- Request Rate (req/sec)
- Latency (p50, p95, p99)
- Error Rate (4xx, 5xx)
- Model Predictions (cat vs dog distribution)
- API Uptime

#### 4. Model Performance Tracking (Post-Deployment)

**Collect Predictions**:
```python
# Log predictions to MLflow
from datetime import datetime
import json

for prediction_result in predictions:
    mlflow.log_metric(
        f"predictions/{prediction_result['class']}/count",
        1,
        step=datetime.now().timestamp()
    )
```

**Compare Pre/Post Deployment Metrics**:
- Training accuracy vs Production accuracy
- Class distribution shift (Data Drift)
- Inference latency (should be < 200ms)

---

## Setup and Installation

For comprehensive setup instructions, see [MLOPS_setup.md](documentation/MLOPS_setup.md).

---

## Workflow Examples

### Training a New Model

```bash
# 1. Prepare data
python src/data_loader.py

# 2. Train with MLflow tracking (automatically trains both baseline and MobileNetV2)
python src/train_cnn.py

# 3. View results in MLflow UI
mlflow ui --port 5000
# - Select the "cat_dog_classification" experiment
# - Compare baseline vs MobileNetV2 runs
# - View accuracy, loss curves, and other metrics
# - Best model is automatically selected based on test accuracy
# - Best model saved as: models/best_model.keras
```

### Deploying New Model

```bash
# 1. Train new model (automatic best model selection)
python src/train_cnn.py
# This generates models/best_model.keras (automatically selected as best performer)

# 2. The CI/CD pipeline will:
#    - Automatically copy models/best_model.keras to app/best_model.keras
#    - Build Docker image with the latest model
#    - Push to container registry

# 3. For manual deployment:
# Copy model to app directory
cp models/best_model.keras app/best_model.keras

# Build and test locally
docker build -t cats-dogs-classifier:v2.0 .
docker run -p 8000:8000 cats-dogs-classifier:v2.0

# 4. Push to registry
docker push ghcr.io/username/cats-dogs-classifier:v2.0

# 5. Update k8s/deployment.yaml image tag and deploy
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/cats-dogs-deployment
```

### Testing Predictions

```bash
# Using curl
curl -X POST -F "file=@cat_photo.jpg" http://localhost:8000/predict

# Using Python
import requests
with open("dog_photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    print(response.json())
```

### Monitoring Metrics

```bash
# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=rate(api_requests_total[5m])'

# Check Grafana dashboard
# http://localhost:3000 (admin/admin)
```

---

## Performance Benchmarks

| Metric | Baseline (Current) | Transfer Learning | Target |
|--------|-------------------|------------------|--------|
| Accuracy | 92.5% | 95.8% | >90% |
| Latency (p95) | 180ms | 220ms | <300ms |
| Throughput | 120 req/s | 100 req/s | >50 req/s |
| Model Size | 45MB | 15MB | <100MB |

---

## Troubleshooting

### Model Not Loading
```bash
# Check model file exists
ls -la app/best_model.keras

# Verify TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Docker Build Fails
```bash
# Clear cache
docker system prune -a

# Rebuild with verbose output
docker build -t cats-dogs-classifier:latest . --no-cache --progress=plain
```

### Kubernetes Pod Crashes
```bash
# Check logs
kubectl logs -f pod/cats-dogs-deployment-xxx

# Check resource limits
kubectl describe pod cats-dogs-deployment-xxx

# Check events
kubectl get events
```

### Metrics Not Appearing in Prometheus
```bash
# Verify scrape config
curl http://localhost:9090/api/v1/targets

# Check Prometheus logs
docker logs prometheus
```

---

## CI/CD Status

[![CI Pipeline]](https://github.com/sriramgunda/MLOPS-Pipeline/actions)

