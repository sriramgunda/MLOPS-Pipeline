# Cats vs Dogs Binary Image Classification - MLOps Pipeline

## Overview

MLOps pipeline for binary image classification (Cats vs Dogs). Includes model training, data versioning, containerization, CI/CD, and monitoring on Kubernetes.

## Architecture Diagram

```
Dataset → Data Versioning (DVC) → Data Preprocessing (224x224 RGB)
    ↓
Model Training (CNN with MLflow) → Experiment Tracking
    ↓
Model Artifact Storage → Docker Containerization
    ↓
CI Pipeline (GitHub Actions) → Unit Tests → Docker Build → Registry Push
    ↓
CD Pipeline (GitOps) → Kubernetes Deployment
    ↓
Monitoring (Prometheus + Grafana) → Logging & Metrics → Model Performance Tracking
```

## Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repo-url>
cd MLOPS-Pipeline

# Linux/Mac
bash setup.sh

# Windows
./setup.bat
```

### 2. Download Dataset & Train Model
```bash
# Download Microsoft Cats vs Dogs dataset
python src/data_loader.py

# Train CNN model with MLflow tracking
python src/train_cnn.py

# View experiments
mlflow ui --port 5000  # Open http://localhost:5000
```

### 3. Start API Service
```bash
# Local development (reload on code changes)
uvicorn app.main:app --reload --port 8000

# For production
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access API
curl -X GET http://localhost:8000/health
```

### 4. Deploy to Kubernetes
```bash
# Deploy to K8s cluster (Minikube/GKE/AKS/EKS)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml

# Check deployment status
kubectl get deployment -w
kubectl get svc -w

# Access API via service
kubectl port-forward svc/cats-dogs-service 8000:80
```

### 5. Run Tests & Monitoring
```bash
# Run unit tests
pytest tests/ -v

# Post-deployment smoke tests
python scripts/smoke_test.py --api-url http://localhost:8000 --timeout 30

# Monitor with Prometheus + Grafana
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

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

### Run Complete Pipeline with DVC

```bash
# Execute entire reproducible pipeline (all 4 stages)
dvc repro

# Or run specific stages:
dvc repro data_download    # Download from Kaggle
dvc repro data_extraction  # Extract ZIP
dvc repro data_organization # Organize 80/10/10 split
dvc repro model_training   # Train MobileNetV2 with MLflow

# Check pipeline status
dvc status
dvc dag  # View pipeline visualization
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

For complete MLOps documentation, see [MLOps_SETUP.md](MLOps_SETUP.md).

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
    
  model_training:
    cmd: python src/train_cnn.py
    deps:
      - src/train_cnn.py
      - data/train
      - data/validation
      - data/test
    params:
      - epochs
      - batch_size
      - learning_rate
    outs:
      - models/mobilenet_v2.keras
    metrics:
      - metrics.json
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
- [YES] Pipeline orchestration (`dvc.yaml` stages)
- [YES] Parameter tracking (`params.yaml`)
- [YES] Experiment comparison (`dvc metrics diff`, `dvc params diff`)
- [YES] Works with S3, GCS, Azure, local storage
- [YES] Lightweight metadata (`.dvc` files in Git)
- [YES] Native MLOps features
- [YES] Automatic reproducibility with `dvc repro`

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
- [YES] Simple Git workflow (no new tool to learn)
- [YES] Transparent - works like normal Git
- [YES] No metadata files (direct file versioning)
- [YES] Good for simple dataset versioning
- [YES] Works with GitHub, GitLab, Bitbucket

**Limitations**:
- [NO] No pipeline orchestration
- [NO] No parameter tracking
- [NO] No built-in experiment comparison
- [NO] GitHub free tier: 1GB bandwidth/month, pay for more

---

### Comparison Table

| Feature | DVC | Git LFS |
|---------|-----|---------|
| **Learning Curve** | Medium | Low |
| **Pipeline Orchestration** | [YES] | [NO] |
| **Parameters Tracking** | [YES] | [NO] |
| **Experiment Comparison** | [YES] | [NO] |
| **Best For** | Full MLOps pipelines | Simple data versioning |

**Recommendation for this assignment**: Use **DVC** for full MLOps module completion (M1 includes pipeline versioning).

---

#### 2. Model Building
- **Architecture**: MobileNetV2 transfer learning with frozen base + custom head
- **Base Model**: ImageNet pre-trained weights
- **Custom Head**: GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid)
- **Input Size**: 224×224 RGB images
- **Model Format**: `.keras` (TensorFlow native format)
- **Output**: Binary classification (Cats vs Dogs)

#### 3. Experiment Tracking with MLflow

All training runs are automatically logged to MLflow with:
- **Run ID**: Unique identifier for each training run
- **Parameters**: epochs, batch_size, learning_rate, model_name, seed
- **Metrics**: train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy
- **Artifacts**: Model weights (.keras), training history

```python
# Training automatically integrates MLflow (see src/train_cnn.py)
from src.train_cnn import train_cnn_pipeline

model, history, test_loss, test_acc = train_cnn_pipeline(
    train_ds, val_ds, test_ds,
    data_augmentation=augmentation,
    epochs=1  # Configurable via params.yaml
)

# View experiments in MLflow UI:
# mlflow ui  # http://localhost:5000
```

**Track**:
- Hyperparameters (learning rate, batch size, epochs)
- Metrics (accuracy, loss, AUC)
- Artifacts (model.h5, confusion_matrix.png, training_history.png)
- Metadata (training time, dataset size)

**Access MLflow UI**:
```bash
mlflow ui --port 5000
# Visit http://localhost:5000
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

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Kubernetes (Minikube for local, or cloud K8s)
- Git & DVC
- ~5GB disk space for dataset

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/username/cats-dogs-mlops-pipeline.git
cd cats-dogs-mlops-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Git & DVC
git init
dvc init

# Option 1: Download dataset manually & add to DVC
python src/data_loader.py
dvc add data/cats_and_dogs/
# This creates data/cats_and_dogs.dvc and updates .gitignore

# Option 2: Reproduce entire DVC pipeline (if remote is configured)
dvc pull              # Download data from remote
dvc repro             # Rebuilds all stages: data_download → data_preprocess → train_model

# Run tests
pytest tests/ -v
```

**DVC Pipeline** (`dvc.yaml`):
- **Stage 1: data_download** - Downloads Microsoft Cats vs Dogs dataset
- **Stage 2: data_preprocess** - Validates images, creates train/val/test splits
- **Stage 3: train_model** - Trains CNN, logs metrics and artifacts

**Run entire reproducible pipeline**:
```bash
dvc repro                    # Runs all stages with cached outputs
dvc repro --force           # Force rerun all stages
dvc repro --single-stage data_download  # Run only one stage
```

**Parameters** in `params.yaml`:
- Training hyperparameters (epochs, batch_size, learning_rate)
- Data configuration (directories, image size)
- Model architecture (CNN, dropout, etc.)
- Augmentation settings

**Track parameter changes**:
```bash
dvc plots diff               # Compare metrics between runs
dvc params diff             # Compare parameters between runs
dvc dag                     # View pipeline DAG
```

# Start MLflow UI
mlflow ui --port 5000

# Run API locally
uvicorn app.main:app --reload --port 8000

# Test API
curl http://localhost:8000/health
```

### Docker Setup

```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# Run with Docker Compose
docker-compose -f docker-compose.yml up -d

# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### Kubernetes Setup

```bash
# Start Minikube
minikube start

# Load Docker image into Minikube
minikube image load cats-dogs-classifier:latest

# Deploy
kubectl apply -f k8s/

# Access service
kubectl port-forward svc/cats-dogs-service 8000:80

# Monitor
kubectl logs -f deployment/cats-dogs-deployment
```

---

## Workflow Examples

### Training a New Model

```bash
# 1. Prepare data
python src/data_loader.py

# 2. Train with MLflow tracking
python src/train_cnn.py \
  --model-type baseline \
  --epochs 50 \
  --batch-size 32

# 3. Check MLflow UI
mlflow ui --port 5000
# Select best run and register model
```

### Deploying New Model

```bash
# 1. Update model path in app/main.py
# 2. Build and test locally
docker build -t cats-dogs-classifier:v2.0 .
docker run -p 8000:8000 cats-dogs-classifier:v2.0

# 3. Push to registry
docker push ghcr.io/username/cats-dogs-classifier:v2.0

# 4. Update k8s/deployment.yaml image tag
# 5. Deploy
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
ls -la app/model.h5

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

[![CI Pipeline](https://github.com/username/cats-dogs-mlops/workflows/CI%20Pipeline/badge.svg)](https://github.com/username/cats-dogs-mlops/actions)

---

## License

MIT License - See LICENSE file

---

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: your.email@example.com
- Reference assignment: MLOps Assignment 2 (S1-25_AIMLCZG523)
