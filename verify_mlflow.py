#!/usr/bin/env python
"""
Quick MLflow verification script
Checks MLflow setup and configuration
"""

import mlflow
import json
import os
from pathlib import Path

def verify_mlflow():
    """Verify MLflow setup"""
    
    print("\n" + "="*70)
    print("MLflow Verification")
    print("="*70 + "\n")
    
    # Check MLflow tracking URI
    tracking_uri = mlflow.get_tracking_uri()
    print(f"[OK] Tracking URI: {tracking_uri}")
    
    # Check MLflow artifacts location
    artifact_root = mlflow.get_artifact_uri()
    print(f"[OK] Artifact Root: {artifact_root}")
    
    # Check MLflow home
    mlflow_home = os.getenv("MLFLOW_HOME", "~/.mlflow")
    print(f"[OK] MLflow Home: {mlflow_home}")
    
    # List experiments
    print("\n" + "-"*70)
    print("Experiments:")
    print("-"*70)
    
    experiments = mlflow.search_experiments()
    if experiments:
        for exp in experiments:
            print(f"  • {exp.name} (ID: {exp.experiment_id})")
    else:
        print("  No experiments found")
    
    # Try to create a test experiment
    print("\n" + "-"*70)
    print("Test Run:")
    print("-"*70)
    
    try:
        mlflow.set_experiment("test_verification")
        with mlflow.start_run(run_name="verify_test"):
            mlflow.log_param("test", "verification")
            mlflow.log_metric("test_metric", 0.5)
            print("[OK] Successfully created test run")
            print(f"[OK] Run ID: {mlflow.active_run().info.run_id}")
    except Exception as e:
        print(f"[FAIL] Error creating test run: {e}")
    
    # List runs in experiment
    print("\n" + "-"*70)
    print("Recent Runs:")
    print("-"*70)
    
    try:
        runs = mlflow.search_runs(experiment_names=["test_verification"])
        if runs is not None and len(runs) > 0:
            for idx, run in enumerate(runs.head(5).itertuples()):
                print(f"  {idx+1}. {run.tags.get('mlflow.runName', 'unnamed')} (Status: {run.status})")
        else:
            print("  No runs found")
    except Exception as e:
        print(f"[FAIL] Error listing runs: {e}")
    
    # Check MLflow UI URL
    print("\n" + "-"*70)
    print("To view MLflow UI:")
    print("-"*70)
    print(f"  mlflow ui --host localhost --port 5000")
    print(f"  Then open: http://localhost:5000")
    
    print("\n" + "="*70)
    print("Verification Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    verify_mlflow()
