# MLOps Setup and Installation Guide

## Prerequisites

- Python 3.10+
- Git & DVC
- Docker (optional, for containerization)
- Kubernetes/Minikube (optional, for deployment)
- ~5GB disk space

## 1. Environment Setup

### Clone and Create Virtual Environment
```bash
git clone <repo-url>
cd MLOPS-Pipeline

# Create virtual environment
python -m venv venv

# Activate (Windows: venv\Scripts\activate)
source venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Initialize Version Control
```bash
git init
dvc init
dvc config core.autostage true
```

## 2. Data Preparation

### Download and Organize Dataset
```bash
# downloads from Kaggle and creates train/validation/test splits (80/10/10)
python src/data_loader.py

# Expected structure after execution:
# data/
#   ├── train/{cats,dogs}/
#   ├── validation/{cats,dogs}/
#   └── test/{cats,dogs}/
```

### Version Data with DVC
```bash
dvc add data/training_set data/test_set
git add data/training_set.dvc data/test_set.dvc .gitignore
git commit -m "Add dataset versioning"

# Optional: push to remote storage
dvc remote add -d storage /path/to/storage
dvc push
```

## 3. Model Training

### Review Configuration
```bash
# Check training parameters in params.yaml
cat params.yaml

# Key settings:
# - epochs: Training epochs
# - batch_size: Batch size for training
# - learning_rate: Optimizer learning rate
# - model.architecture: mobilenet_v2 (transfer learning)
# - img_size: [224, 224] (input image size)
```

### Train Model with MLflow Tracking
```bash
# Trains baseline CNN and MobileNetV2 models
# Automatically logs metrics to MLflow
# Saves best model to models/best_model.keras
python src/train_cnn.py

# View experiment tracking UI
mlflow ui --port 5000
# Visit http://localhost:5000
# - Select "cat_dog_classification" experiment
# - Compare baseline vs MobileNetV2 runs
# - Download model artifacts
```

## 4. API Service

### Verify Model File
```bash
# Copy trained model to app directory
cp models/best_model.keras app/best_model.keras
```

### Start API Service
```bash
# Development mode (auto-reload on code changes)
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test API
curl http://localhost:8000/health
curl -X POST -F "file=@test.jpg" http://localhost:8000/predict
```

### API Endpoints
- `GET /health` - Service health check
- `POST /predict` - Make predictions on uploaded image
- `POST /feedback` - Submit feedback for model improvement
- `GET /metrics` - Prometheus metrics

## 5. Testing

### Run Unit Tests
```bash
# All tests in tests/
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov=app --cov-report=html
```

### Run Smoke Tests
```bash
# Post-deployment health checks
python scripts/smoke_test.py --api-url http://localhost:8000 --timeout 30
```

## 6. Containerization

### Build Docker Image
```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  cats-dogs-classifier:latest

# Test container
curl http://localhost:8000/health

# Push to registry
docker tag cats-dogs-classifier:latest <registry>/cats-dogs-classifier:latest
docker push <registry>/cats-dogs-classifier:latest
```

## 7. Kubernetes Deployment

### Local Testing with Minikube
```bash
# Start Minikube
minikube start --cpus=4 --memory=4096

# Build and load image
docker build -t cats-dogs-classifier:latest .
minikube image load cats-dogs-classifier:latest
```

### Deploy to Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml

# Check deployment status
kubectl get pods -w
kubectl get svc

# Access service
kubectl port-forward svc/cats-dogs-service 8000:80
curl http://localhost:8000/health

# View logs
kubectl logs -f deployment/cats-dogs-deployment
```

## 8. Monitoring

### Prometheus Metrics
```bash
# Metrics exposed at /metrics endpoint
curl http://localhost:8000/metrics

# Available metrics:
# - api_requests_total - Total API requests
# - api_request_latency_seconds - Request latency
# - prediction_confidence - Prediction confidence distribution
# - model_predictions_total - Total predictions by class
```

### Grafana Dashboard
```bash
# Access Grafana at http://localhost:3000
# Default credentials: admin/admin

# Add Prometheus as data source:
# URL: http://localhost:9090

# Create dashboard panels with:
# - Prediction rate: sum(rate(model_predictions_total[1m]))
# - Latency P95: histogram_quantile(0.95, api_request_latency_seconds)
# - Accuracy: sum(model_correct_predictions_total[5m]) / sum(model_predictions_total[5m])
```

## 9. Docker Compose (All-in-One)

### Start All Services
```bash
# Start: API, MLflow, Prometheus, Grafana
docker-compose up -d

# Access services:
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000

# Stop services
docker-compose down
```

## 10. Common Workflows

### Complete Training Pipeline
```bash
# 1. Download and organize dataset
python src/data_loader.py

# 2. Train model (trains both baseline and MobileNetV2)
python src/train_cnn.py

# 3. View results in MLflow UI
mlflow ui --port 5000

# 4. Copy best model to app
cp models/best_model.keras app/best_model.keras

# 5. Run tests
pytest tests/ -v

# 6. Start API
uvicorn app.main:app --reload --port 8000
```

### DVC Pipeline
```bash
# Run complete DVC pipeline (data versioning only)
dvc repro

# View pipeline status
dvc status
dvc dag

# Track parameter changes
dvc params diff
dvc plots diff
```

### CI/CD Workflow (GitHub Actions)
```bash
# Push to trigger CI pipeline (.github/workflows/ci-cd.yml):
# 1. Install dependencies
# 2. Run tests (tests/)
# 3. Build Docker image
# 4. Push to registry

# For CD: Update k8s/deployment.yaml with new image tag
git push origin main
```

## Troubleshooting

### Dependencies Issues
```bash
# Verify Python version
python --version  # Should be 3.10+

# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Data Issues
```bash
# Verify data structure
ls -la data/train/cats/ data/train/dogs/
find data/train -name "*.jpg" | wc -l

# Check Kaggle credentials
# Required: src/kaggle.json with API key
```

### Model Training Issues
```bash
# Monitor training logs
tail -f logs/training.log

# Clear MLflow runs if needed
rm -rf mlruns/

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### API Service Issues
```bash
# Verify model exists
ls -la app/best_model.keras

# Check port availability
netstat -an | grep 8000

# Test locally
curl -v http://localhost:8000/health
```

### Kubernetes Issues
```bash
# Check pod status
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# View cluster events
kubectl get events

# Restart deployment
kubectl rollout restart deployment/cats-dogs-deployment
```

## Key Project Files

- `setup_mlops.py` - Initializes DVC and MLflow
- `setup.sh` - Linux/Mac setup script
- `src/data_loader.py` - Downloads and organizes dataset from Kaggle
- `src/data_preprocessing.py` - Image preprocessing (224x224, normalization)
- `src/train_cnn.py` - Trains CNN and MobileNetV2 models with MLflow tracking
- `src/predict.py` - Inference script
- `app/main.py` - FastAPI service with prediction endpoints
- `tests/` - Unit tests for preprocessing and predictions
- `k8s/` - Kubernetes deployment manifests
- `scripts/smoke_test.py` - Post-deployment verification
- `params.yaml` - Training configuration (epochs, batch size, learning rate)
- `dvc.yaml` - DVC pipeline definition
- `requirements.txt` - Python dependencies

## Verification Checklist

- Virtual environment created and activated
- Dependencies installed: `pip list | grep tensorflow`
- Git and DVC initialized: `.git/` and `.dvc/` directories exist
- Dataset downloaded: `data/train/`, `data/validation/`, `data/test/` exist
- Model trained: `models/best_model.keras` exists
- MLflow UI accessible: `http://localhost:5000`
- API running: `curl http://localhost:8000/health` returns healthy status
- Tests passing: `pytest tests/ -v`
- Docker image built: `docker images | grep cats-dogs-classifier`

## Next Steps

1. Complete Quick Start commands from [README.md](../README.md)
2. Review model performance in MLflow UI
3. Deploy to Kubernetes or run with Docker Compose
4. Set up Grafana dashboards for monitoring
5. Integrate with CI/CD pipeline for automated deployments
