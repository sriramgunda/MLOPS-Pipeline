"""
MLflow configuration and utilities
Handles experiment tracking setup and logging
"""

import mlflow
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_ARTIFACT_ROOT = "mlruns"
MLFLOW_EXPERIMENT_NAME = "cat_dog_classification"


def load_experiment_config():
    """Load experiment configuration from params.yaml"""
    try:
        config_path = Path(__file__).parent.parent / "params.yaml"
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def initialize_mlflow(experiment_name=None, tracking_uri=None):
    """
    Initialize MLflow tracking server
    
    Args:
        experiment_name: Name of experiment (default: from params.yaml)
        tracking_uri: MLflow tracking server URI (default: local)
    """
    try:
        # Determine tracking URI from config or defaults
        config = load_experiment_config()
        if tracking_uri is None:
            tracking_uri = config.get('artifacts', {}).get('mlflow_uri', MLFLOW_TRACKING_URI)

        # Try a light-weight TCP check for remote MLflow before using the MLflow client.
        # This prevents the MLflow client (requests/urllib3) from performing multiple retries
        # when the remote server is unreachable.
        try:
            from urllib.parse import urlparse
            parsed = urlparse(tracking_uri)
            is_http = parsed.scheme in ("http", "https")
        except Exception:
            is_http = False

        if is_http:
            try:
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                import socket
                sock = socket.create_connection((host, port), timeout=1)
                sock.close()
            except Exception as e:
                logger.warning(f"MLflow server {tracking_uri} appears unreachable (quick TCP check failed): {e}")
                logger.warning("Skipping remote MLflow initialization and falling back to local file store.")
            else:
                try:
                    mlflow.set_tracking_uri(tracking_uri)
                    if experiment_name is None:
                        experiment_name = config.get('artifacts', {}).get('experiment_name', MLFLOW_EXPERIMENT_NAME)
                    mlflow.set_experiment(experiment_name)
                    logger.info(f"MLflow initialized - URI: {tracking_uri}, Experiment: {experiment_name}")
                    return
                except Exception as e:
                    logger.warning(f"Could not initialize remote MLflow at {tracking_uri}: {e}")
        else:
            # Non-http URIs (e.g., file://) — try directly
            try:
                mlflow.set_tracking_uri(tracking_uri)
                if experiment_name is None:
                    experiment_name = config.get('artifacts', {}).get('experiment_name', MLFLOW_EXPERIMENT_NAME)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow initialized - URI: {tracking_uri}, Experiment: {experiment_name}")
                return
            except Exception as e:
                logger.warning(f"Could not initialize MLflow at {tracking_uri}: {e}")

        # Fallback to local file-based tracking (mlruns directory)
        try:
            local_dir = Path(__file__).parent.parent.joinpath(MLFLOW_ARTIFACT_ROOT).absolute()
            local_uri = local_dir.as_uri()
            mlflow.set_tracking_uri(local_uri)
            if experiment_name is None:
                experiment_name = config.get('artifacts', {}).get('experiment_name', MLFLOW_EXPERIMENT_NAME)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow fallback initialized - URI: {local_uri}, Experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow even with local fallback: {e}")
            logger.warning("Proceeding without MLflow tracking.")
            # Do not raise; allow code to continue without MLflow
            return
    except Exception as e:
        logger.error(f"Unexpected error initializing MLflow: {e}")
        logger.warning("Proceeding without MLflow tracking.")
        return


def start_mlflow_run(run_name=None, tags=None, params=None):
    """
    Start MLflow run with configuration
    
    Args:
        run_name: Name of the run
        tags: Dictionary of tags
        params: Dictionary of parameters to log
        
    Returns:
        MLflow run object
    """
    try:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(run_name=run_name)
        
        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        logger.info(f"MLflow run started: {run_name}")
        return run
        
    except Exception as e:
        logger.error(f"Error starting MLflow run: {e}")
        raise


def log_model_metrics(metrics_dict):
    """
    Log model metrics to MLflow
    
    Args:
        metrics_dict: Dictionary of metrics to log
    """
    try:
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value)
        logger.info(f"Logged {len(metrics_dict)} metrics to MLflow")
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")
        raise


def end_mlflow_run(status="FINISHED"):
    """
    End current MLflow run
    
    Args:
        status: Status of the run (FINISHED, FAILED)
    """
    try:
        mlflow.end_run()
        logger.info(f"MLflow run ended with status: {status}")
    except Exception as e:
        logger.error(f"Error ending MLflow run: {e}")


def get_mlflow_tracking_uri():
    """Get current MLflow tracking URI"""
    return mlflow.get_tracking_uri()


def get_active_run():
    """Get active MLflow run"""
    return mlflow.active_run()


def log_model_summary(model, framework="tensorflow"):
    """
    Log model summary and architecture to MLflow
    
    Args:
        model: Trained model object
        framework: Framework type (tensorflow, pytorch, sklearn)
    """
    try:
        if framework == "tensorflow":
            # Log model summary as artifact
            summary_path = "model_summary.txt"
            with open(summary_path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            mlflow.log_artifact(summary_path)
            os.remove(summary_path)
            logger.info("Model summary logged to MLflow")
    except Exception as e:
        logger.error(f"Error logging model summary: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("MLflow configuration module initialized.")
