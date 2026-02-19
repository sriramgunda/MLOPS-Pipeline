#!/bin/bash
# Setup script for Cats vs Dogs MLOps Pipeline
# Automates initial project setup

set -e

echo "=================================="
echo "Cats vs Dogs MLOps Pipeline Setup"
echo "=================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_status() {
    echo -e "${GREEN}[OK] $1${NC}"
}

function print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# 1. Check prerequisites
print_info "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi
print_status "Python found: $(python3 --version)"

if ! command -v git &> /dev/null; then
    print_error "Git not found. Please install Git"
    exit 1
fi
print_status "Git found"

if ! command -v kubectl &> /dev/null; then
    print_info "kubectl not found. Install for Kubernetes deployment"
else
    print_status "kubectl found: $(kubectl version --client --short)"
fi

# 2. Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
print_status "Virtual environment activated"

# 3. Install dependencies
print_info "Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_status "Dependencies installed"

# 4. Create necessary directories
print_info "Creating project directories..."
mkdir -p data/{train,validation,test}/{cats,dogs}
mkdir -p data/training_set
mkdir -p data/test_set
mkdir -p models artifacts logs mlruns plots
print_status "Directories created"

# 5. Setup Git
print_info "Initializing Git repository..."
if [ ! -d ".git" ]; then
    git init
    git config user.email "mlops@example.com"
    git config user.name "MLOps Pipeline"
    print_status "Git repository initialized"
else
    print_info "Git repository already exists"
fi

# 6. Setup Data Versioning - Choose DVC or Git LFS
print_info "Setting up data versioning..."
echo "Options:"
echo "  1. DVC (recommended for MLOps pipelines)"
echo "  2. Git LFS (simpler for basic versioning)"
read -p "Choose versioning method (1/2) [default=1]: " versioning_choice
versioning_choice=${versioning_choice:-1}

if [ "$versioning_choice" = "2" ]; then
    # Git LFS option
    print_info "Setting up Git LFS..."
    if ! command -v git-lfs &> /dev/null; then
        echo "Git LFS not installed. Please install from https://git-lfs.github.com/"
        echo "  macOS: brew install git-lfs"
        echo "  Linux: sudo apt-get install git-lfs"
        echo "  Windows: choco install git-lfs"
    else
        git lfs install
        print_status "Git LFS initialized"
    fi
else
    # DVC option (default)
    print_info "Initializing DVC..."
    if ! command -v dvc &> /dev/null; then
        pip install dvc > /dev/null 2>&1
    fi

    if [ ! -d ".dvc" ]; then
        dvc init
        print_status "DVC initialized"
    else
        print_info "DVC already initialized"
    fi
fi

# 7. Create .env from example
print_info "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_status ".env file created (update with your values)"
else
    print_info ".env file already exists"
fi

# 8. Create startup scripts
print_info "Creating startup scripts..."

cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "Starting Cats vs Dogs MLOps Pipeline (Local Development)"
source venv/bin/activate
mlflow ui --port 5000 &
uvicorn app.main:app --reload --port 8000
EOF
chmod +x start_dev.sh
print_status "start_dev.sh created"

cat > deploy_k8s.sh << 'EOF'
#!/bin/bash
echo "Deploying Cats vs Dogs MLOps Pipeline to Kubernetes"
echo ""
echo "Prerequisites:"
echo "  - Minikube/K8s cluster running"
echo "  - kubectl configured"
echo "  - Docker image pushed to registry"
echo ""
echo "Deploying..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml
echo ""
echo "Checking rollout status..."
kubectl rollout status deployment/cats-dogs-deployment --timeout=5m
echo ""
echo "Services deployed:"
kubectl get svc
EOF
chmod +x deploy_k8s.sh
print_status "deploy_k8s.sh created"

# 9. Initialize MLOps Infrastructure
print_info "Initializing MLOps (DVC + MLflow)..."
if python -c "from src import dvc_utils; dvc_utils.initialize_dvc()" 2>/dev/null; then
    print_status "DVC initialized"
else
    print_info "DVC initialization handled separately (run: dvc init)"
fi

# 10. Run tests
print_info "Running tests..."
pytest tests/ -v --tb=short 2>&1 | tail -20
print_status "Tests completed"

# 11. Print summary
echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
print_info "Next steps:"
echo ""
echo "1. Initialize MLOps:"
echo "   python setup_mlops.py"
echo ""
echo "2. Download dataset and train:"
echo "   dvc repro"
echo ""
echo "3. Or run step-by-step:"
echo "   python src/data_loader.py (to download)"
echo "   python src/train_cnn.py (to train)"
echo ""
echo "4. Monitor experiments:"
echo "   mlflow ui  # Open http://localhost:5000"
echo ""
echo "5. Start local API service:"
echo "   uvicorn app.main:app --reload --port 8000"
echo ""
echo "6. Deploy to Kubernetes:"
echo "   ./deploy_k8s.sh"
echo ""
echo "Documentation:"
echo "  - README.md: Project overview"
echo "  - MLOps_SETUP.md: MLOps infrastructure guide"
echo ""

