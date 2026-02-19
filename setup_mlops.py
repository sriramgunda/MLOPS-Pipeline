#!/usr/bin/env python
"""
MLOps Initialization Script
Sets up DVC and MLflow for the project
Run this once to initialize MLOps infrastructure
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dvc_utils import initialize_dvc, check_dvc_initialized, run_dvc_pipeline
from mlflow_config import initialize_mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_dvc():
    """Initialize and configure DVC"""
    logger.info("=" * 60)
    logger.info("Setting up DVC (Data Version Control)")
    logger.info("=" * 60)
    
    if initialize_dvc():
        logger.info("[OK] DVC initialized successfully")
        
        # Configure DVC caching
        try:
            subprocess.run(
                ["dvc", "config", "core.autostage", "true"],
                check=True,
                capture_output=True
            )
            logger.info("[OK] DVC autostage enabled")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not enable autostage: {e}")
        
        return True
    else:
        logger.error("[FAIL] Failed to initialize DVC")
        return False


def setup_mlflow():
    """Initialize MLflow"""
    logger.info("=" * 60)
    logger.info("Setting up MLflow (Experiment Tracking)")
    logger.info("=" * 60)
    
    try:
        initialize_mlflow()
        logger.info("[OK] MLflow configured successfully")
        logger.info("  URI: http://localhost:5000")
        logger.info("  Start MLflow UI with: mlflow ui")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize MLflow: {e}")
        return False


def create_required_directories():
    """Create required directories"""
    logger.info("=" * 60)
    logger.info("Creating required directories")
    logger.info("=" * 60)
    
    dirs = [
        "data",
        "models",
        "logs",
        "mlruns"
    ]
    
    for dir_name in dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"[OK] Created/verified directory: {dir_name}")
        except Exception as e:
            logger.warning(f"Could not create {dir_name}: {e}")
    
    return True


def verify_dependencies():
    """Verify required packages are installed"""
    logger.info("=" * 60)
    logger.info("Verifying dependencies")
    logger.info("=" * 60)
    
    required_packages = [
        ("tensorflow", "TensorFlow"),
        ("dvc", "DVC"),
        ("mlflow", "MLflow"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn")
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {name} installed")
        except ImportError:
            logger.error(f"[FAIL] {name} NOT installed")
            missing.append(name)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Run full MLOps setup"""
    logger.info("\n" + "=" * 60)
    logger.info("MLOps Pipeline Initialization")
    logger.info("=" * 60 + "\n")
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    logger.info(f"Working directory: {os.getcwd()}\n")
    
    # Step 1: Verify dependencies
    if not verify_dependencies():
        logger.error("\nPlease install missing dependencies:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info()
    
    # Step 2: Create directories
    if not create_required_directories():
        logger.warning("Some directories could not be created")
    
    logger.info()
    
    # Step 3: Setup DVC
    dvc_ok = setup_dvc()
    
    logger.info()
    
    # Step 4: Setup MLflow
    mlflow_ok = setup_mlflow()
    
    logger.info()
    
    # Step 5: Summary
    logger.info("=" * 60)
    logger.info("MLOps Setup Summary")
    logger.info("=" * 60)
    
    if dvc_ok and mlflow_ok:
        logger.info("[OK] MLOps infrastructure initialized successfully!\n")
        
        logger.info("Next steps:")
        logger.info("  1. Download dataset: python src/data_loader.py")
        logger.info("  2. Or run full pipeline: dvc repro")
        logger.info("  3. Monitor experiments: mlflow ui")
        logger.info("\nFor detailed documentation, see: MLOps_SETUP.md")
        
        return 0
    else:
        logger.error("[FAIL] Some components failed to initialize")
        logger.info("Please check the logs above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
