"""
Simple CLI to run the project's DVC pipeline programmatically.
Usage:
    python -m src.run_dvc --init --stages prepare train

This script uses functions in `src/dvc_utils.py` to initialize DVC and reproduce stages.
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml

from dvc_utils import initialize_dvc, run_dvc_pipeline, check_dvc_initialized, verify_pipeline_artifacts

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run DVC pipeline programmatically")
    parser.add_argument("--init", action="store_true", help="Initialize DVC if not present")
    parser.add_argument("--stages", nargs="*", help="Specific DVC stages to run (space-separated)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cwd = Path.cwd()
    logger.info("Starting DVC pipeline runner")
    logger.info(f"Working directory: {cwd}")

    # Step 1: Initialization
    print("\nStep 1/5: Initialization")
    if args.init:
        logger.info("Initializing DVC (requested by --init)")
        ok = initialize_dvc()
        if not ok:
            logger.error("Failed to initialize DVC")
            sys.exit(2)
    else:
        logger.info("Checking if DVC is initialized")
        if not check_dvc_initialized():
            logger.error("DVC is not initialized. Use --init to initialize.")
            sys.exit(2)
    print("[OK] DVC initialized check passed")

    # Ensure dvc.yaml exists
    dvc_file = cwd / "dvc.yaml"
    if not dvc_file.exists():
        logger.error("dvc.yaml not found in working directory")
        sys.exit(3)

    # Step 2: Pipeline overview
    print("\nStep 2/5: Pipeline overview")
    logger.info("Displaying pipeline DAG (dvc dag)")
    print("\n--- DVC DAG ---")
    import subprocess
    try:
        subprocess.run(["dvc", "dag"], check=False)
    except Exception as e:
        logger.debug(f"Failed to run `dvc dag`: {e}")

    # List stages from dvc.yaml for visibility
    try:
        with open("dvc.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        stages = list(cfg.get("stages", {}).keys())
        print("\nPipeline stages:")
        for i, s in enumerate(stages, start=1):
            print(f"  {i}. {s}")
    except Exception:
        logger.debug("Could not list stages from dvc.yaml")

    # Step 3: Verify existing artifacts
    print("\nStep 3/5: Verify existing artifacts")
    ok_artifacts_before = verify_pipeline_artifacts()
    if not ok_artifacts_before:
        logger.warning("Some expected artifacts are missing before reproduction; repro will attempt to generate them.")

    # Step 4: Run pipeline
    print("\nStep 4/5: Running pipeline")
    if args.stages:
        logger.info(f"Running specific stages: {args.stages}")
    else:
        logger.info("Running full pipeline (all stages)")

    success = run_dvc_pipeline(stages=args.stages if args.stages else None)
    if not success:
        logger.error("DVC pipeline failed")
        sys.exit(4)

    logger.info("DVC pipeline completed successfully")

    # Step 5: Verify artifacts after run
    print("\nStep 5/5: Verify artifacts after run")
    ok_artifacts_after = verify_pipeline_artifacts()
    if ok_artifacts_after:
        print("[OK] All declared artifacts exist")
    else:
        print("[WARN] Some artifacts are still missing; check logs above for details")

    print("\nNext steps:")
    print("  - Check status:     dvc status")
    print("  - Monitor MLflow:   mlflow ui")
    print("  - Train models locally (outside DVC): python src/train_cnn.py --config params.yaml")


if __name__ == "__main__":
    main()
