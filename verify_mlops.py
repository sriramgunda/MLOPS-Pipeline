#!/usr/bin/env python
"""
MLOps Configuration Verification Script
Checks all MLOps components are properly configured
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Tuple

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.RESET}\n")

def print_pass(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")

def print_fail(text):
    print(f"{Colors.RED}[FAIL] {text}{Colors.RESET}")

def print_warn(text):
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

def check_file_exists(filepath: str) -> Tuple[bool, str]:
    """Check if file exists"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        return True, f"({size} bytes)"
    return False, "NOT FOUND"

def check_package_installed(package_name: str) -> bool:
    """Check if Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_file_content(filepath: str, search_text: str) -> bool:
    """Check if file contains specific text"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            return search_text in content
    except:
        return False

def check_yaml_structure(filepath: str, keys: list) -> Tuple[bool, list]:
    """Check if YAML file contains required keys"""
    try:
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        missing = []
        for key in keys:
            if key not in str(data):
                missing.append(key)
        
        return len(missing) == 0, missing
    except:
        return False, keys

def main():
    """Run verification checks"""
    
    print_header("MLOps Configuration Verification")
    
    passed = 0
    failed = 0
    warnings = 0
    
    # ===== 1. Configuration Files =====
    print_header("1. Configuration Files")
    
    # Check dvc.yaml
    exists, info = check_file_exists("dvc.yaml")
    if exists:
        print_pass(f"dvc.yaml exists {info}")
        valid, missing = check_yaml_structure("dvc.yaml", ["stages"])
        if valid:
            print_pass("dvc.yaml has valid structure with stages")
            passed += 1
        else:
            print_fail(f"dvc.yaml missing required keys: {missing}")
            failed += 1
    else:
        print_fail(f"dvc.yaml {info}")
        failed += 1
    
    # Check params.yaml
    exists, info = check_file_exists("params.yaml")
    if exists:
        print_pass(f"params.yaml exists {info}")
        required_keys = ["epochs", "batch_size", "learning_rate", "data", "model"]
        valid, missing = check_yaml_structure("params.yaml", required_keys)
        if valid:
            print_pass("params.yaml has all required configuration keys")
            passed += 1
        else:
            print_warn(f"params.yaml missing some keys: {missing}")
            warnings += 1
    else:
        print_fail(f"params.yaml {info}")
        failed += 1
    
    # Check .dvcignore
    exists, info = check_file_exists(".dvcignore")
    if exists:
        print_pass(f".dvcignore exists {info}")
        passed += 1
    else:
        print_warn(f".dvcignore {info}")
        warnings += 1
    
    # Check .mlflowconfig
    exists, info = check_file_exists(".mlflowconfig")
    if exists:
        print_pass(f".mlflowconfig exists {info}")
        passed += 1
    else:
        print_warn(f".mlflowconfig {info}")
        warnings += 1
    
    # ===== 2. Source Code Modules =====
    print_header("2. MLOps Utility Modules")
    
    # Check mlflow_config.py
    exists, info = check_file_exists("src/mlflow_config.py")
    if exists:
        print_pass(f"mlflow_config.py exists {info}")
        required_funcs = ["initialize_mlflow", "start_mlflow_run", "log_model_metrics"]
        all_funcs = True
        for func in required_funcs:
            if not check_file_content("src/mlflow_config.py", f"def {func}"):
                all_funcs = False
                print_warn(f"  Missing function: {func}")
        if all_funcs:
            print_pass("mlflow_config.py has all required functions")
            passed += 1
        else:
            warnings += 1
    else:
        print_fail(f"mlflow_config.py {info}")
        failed += 1
    
    # Check dvc_utils.py
    exists, info = check_file_exists("src/dvc_utils.py")
    if exists:
        print_pass(f"dvc_utils.py exists {info}")
        required_funcs = ["initialize_dvc", "run_dvc_pipeline"]
        all_funcs = True
        for func in required_funcs:
            if not check_file_content("src/dvc_utils.py", f"def {func}"):
                all_funcs = False
                print_warn(f"  Missing function: {func}")
        if all_funcs:
            print_pass("dvc_utils.py has required functions")
            passed += 1
        else:
            warnings += 1
    else:
        print_fail(f"dvc_utils.py {info}")
        failed += 1
    
    # ===== 3. Setup Scripts =====
    print_header("3. Setup & Initialization Scripts")
    
    # Check setup_mlops.py
    exists, info = check_file_exists("setup_mlops.py")
    if exists:
        print_pass(f"setup_mlops.py exists {info}")
        passed += 1
    else:
        print_fail(f"setup_mlops.py {info}")
        failed += 1
    
    # ===== 4. Git Configuration =====
    print_header("4. Git & Version Control")
    
    # Check .gitignore
    exists, info = check_file_exists(".gitignore")
    if exists:
        print_pass(f".gitignore exists {info}")
        has_dvc_cache = check_file_content(".gitignore", ".dvc/cache")
        has_mlflow = check_file_content(".gitignore", "mlruns/")
        has_models = check_file_content(".gitignore", "models/")
        
        if has_dvc_cache and has_mlflow and has_models:
            print_pass(".gitignore properly excludes DVC/MLflow artifacts")
            passed += 1
        else:
            print_warn(".gitignore may be missing some artifact exclusions")
            warnings += 1
    else:
        print_fail(f".gitignore {info}")
        failed += 1
    
    # Check .git directory
    if os.path.isdir(".git"):
        print_pass(".git directory exists")
        passed += 1
    else:
        print_warn(".git directory not found (run: git init)")
        warnings += 1
    
    # ===== 5. Documentation =====
    print_header("5. Documentation Files")
    
    docs = [
        ("MLOps_SETUP.md", "Comprehensive MLOps guide"),
        ("MLOPS_CHECKLIST.md", "Implementation verification"),
        ("README.md", "Project README with MLOps section"),
    ]
    
    for doc_file, description in docs:
        exists, info = check_file_exists(doc_file)
        if exists:
            print_pass(f"{doc_file} exists {info} - {description}")
            passed += 1
        else:
            print_warn(f"{doc_file} {info}")
            warnings += 1
    
    # ===== 6. Python Dependencies =====
    print_header("6. Python Package Dependencies")
    
    packages = [
        ("tensorflow", "TensorFlow"),
        ("dvc", "DVC"),
        ("mlflow", "MLflow"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
    ]
    
    missing_packages = []
    for package, name in packages:
        if check_package_installed(package):
            print_pass(f"{name} installed")
            passed += 1
        else:
            print_fail(f"{name} NOT installed")
            missing_packages.append(package)
            failed += 1
    
    if missing_packages:
        print_info(f"Install missing packages: pip install {' '.join(missing_packages)}")
    
    # ===== 7. Directory Structure =====
    print_header("7. Required Directories")
    
    directories = [
        ("data", "Dataset directory"),
        ("models", "Model artifacts"),
        ("src", "Source code"),
        ("tests", "Test files"),
        ("logs", "Log files"),
    ]
    
    for dir_name, description in directories:
        if os.path.isdir(dir_name):
            print_pass(f"{dir_name}/ exists - {description}")
            passed += 1
        else:
            print_warn(f"{dir_name}/ NOT found")
            warnings += 1
    
    # ===== 8. DVC Status =====
    print_header("8. DVC Status")
    
    if os.path.isdir(".dvc"):
        print_pass(".dvc directory exists (DVC initialized)")
        passed += 1
        
        # Check if dvc.lock exists
        if os.path.exists("dvc.lock"):
            print_info("dvc.lock found (pipeline has been run)")
        else:
            print_info("dvc.lock not found (pipeline hasn't been run yet)")
    else:
        print_warn(".dvc directory not found - Run: dvc init")
        warnings += 1
    
    # ===== 9. Model Artifacts =====
    print_header("9. Model Artifacts")
    
    model_dir = Path("models")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
        if model_files:
            print_pass(f"Model artifacts found: {len(model_files)} file(s)")
            for model_file in model_files:
                size = model_file.stat().st_size / (1024*1024)  # Convert to MB
                print_info(f"  - {model_file.name} ({size:.2f} MB)")
            passed += 1
        else:
            print_info("No model artifacts yet (run training to generate)")
    else:
        print_warn("models/ directory not found")
        warnings += 1
    
    # ===== Summary =====
    print_header("Verification Summary")
    
    total = passed + failed
    percentage = (passed / total * 100) if total > 0 else 0
    
    print_pass(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    if failed > 0:
        print_fail(f"Failed: {failed}/{total}")
    if warnings > 0:
        print_warn(f"Warnings: {warnings}")
    
    print()
    
    if failed == 0:
        print_pass("[OK] All critical MLOps components verified!")
        if warnings == 0:
            print_pass("[OK] No issues found - Ready to use!")
        else:
            print_info("[WARNING] Some optional components may need attention")
        return 0
    else:
        print_fail("[FAIL] Some critical components are missing!")
        print_info("\nRun the following to fix:")
        print_info("  python setup_mlops.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
