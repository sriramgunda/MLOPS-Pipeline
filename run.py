#!/usr/bin/env python
"""
Direct Pipeline Runner - Execute pipeline locally with DVC
No arguments needed - just run: python run.py
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, desc=""):
    """Run command and print status"""
    if desc:
        print(f"\n... {desc}")
        print("=" * 60)
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] {cmd}")
        sys.exit(1)
    print(f"[OK] Done\n")

def main():
    print("\n" + "=" * 60)
    print("  MLOps Pipeline Direct Runner")
    print("=" * 60)
    
    # 1. Check DVC installed
    print("\n[1/5] Checking DVC installation...")
    result = subprocess.run("dvc --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("[INFO] DVC not installed. Installing...")
        run_cmd("pip install dvc dvc[gs]", "Installing DVC")
    else:
        print(f"[OK] DVC installed: {result.stdout.decode().strip()}")
    
    # 2. Initialize DVC
    print("\n[2/5] Initializing DVC...")
    if not Path(".dvc").exists():
        run_cmd("dvc init", "Initializing DVC")
        run_cmd("dvc config core.autostage true", "Enabling autostage")
    else:
        print("[OK] DVC already initialized")
    
    # 3. Check dvc.yaml exists
    print("\n[3/5] Verifying pipeline configuration...")
    if not Path("dvc.yaml").exists():
        print("[FAIL] dvc.yaml not found!")
        sys.exit(1)
    print("[OK] dvc.yaml found")
    
    # 4. View pipeline
    print("\n[4/5] Pipeline Structure:")
    print("-" * 60)
    run_cmd("dvc dag", "")
    
    # 5. Run pipeline
    print("\n[5/5] Running pipeline (all 4 stages)...")
    print("-" * 60)
    run_cmd("dvc repro", "Executing DVC pipeline")
    
    # Summary
    print("\n" + "=" * 60)
    print("  [OK] Pipeline Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - View metrics:     dvc metrics show")
    print("  - Check status:     dvc status")
    print("  - Monitor MLflow:   mlflow ui")
    print()

if __name__ == "__main__":
    main()
