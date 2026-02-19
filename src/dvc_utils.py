"""
DVC pipeline utilities for data versioning and tracking
Handles dataset download, extraction, and organization stages
"""

import os
import yaml
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# DVC Configuration
DVC_CACHE_DIR = ".dvc/cache"
DVC_REMOTE_NAME = "myremote"  # For future configuration


def load_dvc_config():
    """Load DVC configuration from params.yaml"""
    try:
        config_path = Path(__file__).parent.parent / "params.yaml"
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('dvc', {})
    except Exception as e:
        logger.error(f"Error loading DVC config: {e}")
        return {}


def check_dvc_initialized():
    """Check if DVC is initialized in the project"""
    return os.path.exists(".dvc") and os.path.isdir(".dvc")


def initialize_dvc():
    """Initialize DVC in the project"""
    try:
        if check_dvc_initialized():
            logger.info("DVC already initialized")
            return True
        
        result = subprocess.run(
            ["dvc", "init"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("DVC initialized successfully")
            # Enable autostage
            subprocess.run(
                ["dvc", "config", "core.autostage", "true"],
                cwd=os.getcwd(),
                capture_output=True
            )
            return True
        else:
            logger.error(f"DVC initialization failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing DVC: {e}")
        return False


def run_dvc_pipeline(stages=None):
    """
    Run DVC pipeline
    
    Args:
        stages: List of specific stages to run (None = all)
    """
    try:
        if not check_dvc_initialized():
            logger.error("DVC not initialized. Run initialize_dvc() first.")
            return False
        
        cmd = ["dvc", "repro"]
        if stages:
            for stage in stages:
                cmd.extend(["--single-item", stage])
        
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"DVC pipeline executed successfully")
            logger.debug(result.stdout)
            return True
        else:
            logger.error(f"DVC pipeline execution failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running DVC pipeline: {e}")
        return False


def add_dvc_remote(remote_name=None, remote_url=None):
    """
    Add DVC remote storage (for future cloud storage)
    
    Args:
        remote_name: Name of the remote
        remote_url: URL or path of the remote storage
    """
    try:
        if remote_name is None:
            remote_name = DVC_REMOTE_NAME
        
        cmd = ["dvc", "remote", "add", "-d", remote_name, remote_url or "data/remote"]
        
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"DVC remote '{remote_name}' added successfully")
            return True
        else:
            logger.warning(f"Could not add DVC remote: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error adding DVC remote: {e}")
        return False


def get_dvc_status():
    """Get DVC pipeline status"""
    try:
        result = subprocess.run(
            ["dvc", "status"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            status = result.stdout.strip()
            if not status:
                logger.info("Pipeline is up to date")
                return "up_to_date"
            else:
                logger.info(f"Pipeline status:\n{status}")
                return "needs_update"
        else:
            logger.error(f"Error checking DVC status: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting DVC status: {e}")
        return None


def create_dvc_artifacts_lock():
    """Create/Update DVC lock file for reproducibility"""
    try:
        result = subprocess.run(
            ["dvc", "lock"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("DVC lock file created/updated")
            return True
        else:
            logger.warning(f"Could not create DVC lock: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating DVC lock: {e}")
        return False


def get_data_stage_path(stage_name):
    """Get output path for a specific DVC stage"""
    try:
        config_path = Path(__file__).parent.parent / "dvc.yaml"
        if not config_path.exists():
            logger.warning("dvc.yaml not found")
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('stages', {})
        if stage_name in stages:
            outputs = stages[stage_name].get('outs', [])
            if outputs:
                return outputs[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting stage path: {e}")
        return None


def verify_pipeline_artifacts():
    """Verify all expected DVC artifacts exist"""
    try:
        config_path = Path(__file__).parent.parent / "dvc.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('stages', {})
        missing = []
        
        for stage_name, stage_config in stages.items():
            outputs = stage_config.get('outs', [])
            for output in outputs:
                if not os.path.exists(output):
                    missing.append(f"{stage_name}: {output}")
        
        if missing:
            logger.warning(f"Missing DVC artifacts:\n" + "\n".join(missing))
            return False
        else:
            logger.info("All pipeline artifacts verified")
            return True
            
    except Exception as e:
        logger.error(f"Error verifying artifacts: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("DVC utilities module initialized.")
