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


def _run_and_stream(cmd, cwd=None):
    """Run a command and stream stdout/stderr to logger and stdout.

    Returns tuple (returncode, combined_output_str).
    """
    try:
        if cwd is None:
            cwd = os.getcwd()

        if isinstance(cmd, (list, tuple)):
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        else:
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        collected = []
        # Stream lines as they arrive
        for line in proc.stdout:
            text = line.rstrip("\n")
            collected.append(text)
            logger.info(text)
            print(text)

        proc.wait()
        return proc.returncode, "\n".join(collected)

    except Exception as e:
        logger.exception(f"Failed to run command {cmd}: {e}")
        return 1, str(e)

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
        
        rc, out = _run_and_stream(["dvc", "init"], cwd=os.getcwd())

        if rc == 0:
            logger.info("DVC initialized successfully")
            # Enable autostage
            _run_and_stream(["dvc", "config", "core.autostage", "true"], cwd=os.getcwd())
            return True
        else:
            logger.error(f"DVC initialization failed")
            logger.debug(out)
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
        # If specific stages are provided, pass them as targets to dvc repro
        # e.g. `dvc repro stage_name` or multiple stage names
        if stages:
            if isinstance(stages, (list, tuple)):
                cmd.extend(list(stages))
            else:
                cmd.append(str(stages))
        
        rc, out = _run_and_stream(cmd, cwd=os.getcwd())
        if rc == 0:
            logger.info("DVC pipeline executed successfully")
            logger.debug(out)
            return True
        else:
            logger.error("DVC pipeline execution failed")
            logger.debug(out)
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
        
        rc, out = _run_and_stream(cmd, cwd=os.getcwd())
        if rc == 0:
            logger.info(f"DVC remote '{remote_name}' added successfully")
            return True
        else:
            logger.warning(f"Could not add DVC remote")
            logger.debug(out)
            return False
            
    except Exception as e:
        logger.error(f"Error adding DVC remote: {e}")
        return False


def get_dvc_status():
    """Get DVC pipeline status"""
    try:
        rc, out = _run_and_stream(["dvc", "status"], cwd=os.getcwd())
        if rc == 0:
            status = out.strip()
            if not status:
                logger.info("Pipeline is up to date")
                return "up_to_date"
            else:
                logger.info(f"Pipeline status:\n{status}")
                return "needs_update"
        else:
            logger.error("Error checking DVC status")
            logger.debug(out)
            return None
            
    except Exception as e:
        logger.error(f"Error getting DVC status: {e}")
        return None


def create_dvc_artifacts_lock():
    """Create/Update DVC lock file for reproducibility"""
    try:
        rc, out = _run_and_stream(["dvc", "lock"], cwd=os.getcwd())
        if rc == 0:
            logger.info("DVC lock file created/updated")
            return True
        else:
            logger.warning("Could not create DVC lock")
            logger.debug(out)
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
