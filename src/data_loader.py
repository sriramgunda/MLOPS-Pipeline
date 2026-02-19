"""
Dataset downloading and organization
Handles Kaggle dataset download and 80/10/10 train/validation/test split
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset configuration
KAGGLE_DATASET = "tongpython/cat-and-dog"
SEED = 42
IMG_SIZE = (224, 224)


def configure_kaggle_api():
    """Configure Kaggle API from kaggle.json"""
    try:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Check if kaggle.json exists in current directory or src directory
        kaggle_json_path = None
        if os.path.exists("kaggle.json"):
            kaggle_json_path = "kaggle.json"
        elif os.path.exists("src/kaggle.json"):
            kaggle_json_path = "src/kaggle.json"
        
        if kaggle_json_path:
            shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            logger.info("Kaggle API configured successfully.")
        else:
            logger.warning("kaggle.json not found. Please upload kaggle.json first.")
    except Exception as e:
        logger.error(f"Error configuring Kaggle API: {e}")
        raise


def download_dataset(download_dir: str = "data"):
    """Download Kaggle dataset to specified directory"""
    try:
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Downloading dataset to {download_dir}: {KAGGLE_DATASET}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", download_dir],
            check=True
        )
        logger.info("Dataset downloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def extract_dataset(download_dir: str = "data"):
    """Extract downloaded dataset"""
    try:
        zip_path = os.path.join(download_dir, "cat-and-dog.zip")
        logger.info(f"Extracting dataset from {zip_path} to {download_dir}...")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(download_dir)
            logger.info("Dataset extracted successfully.")
        except zipfile.BadZipFile as e:
            logger.error(f"Bad zip file: {e}")
            raise
    except Exception as e:
        logger.error(f"Error extracting dataset: {e}")
        raise


def organize_dataset_80_10_10(source_path: str = "data", output_dir: str = "data"):
    """
    Organize dataset into 80/10/10 split for training/validation/test
    
    Args:
        source_path: Path to the extracted dataset (where training_set and test_set are located)
        output_dir: Output directory for organized data (train/validation/test splits)
    """
    try:
        logger.info(f"Organizing dataset with 80/10/10 split...")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "validation")
        test_dir = os.path.join(output_dir, "test")
        
        # Create class subdirectories
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, "cats"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "dogs"), exist_ok=True)
        
        # Process training and test sets from original layout
        # Combine training_set and test_set into one pool for 80/10/10 split
        all_images = {"cats": [], "dogs": []}
        
        # Collect from training_set
        if os.path.exists(os.path.join(source_path, "training_set", "training_set")):
            base_path = os.path.join(source_path, "training_set", "training_set")
            for class_name in ["cats", "dogs"]:
                class_path = os.path.join(base_path, class_name)
                if os.path.exists(class_path):
                    for img in os.listdir(class_path):
                        all_images[class_name].append(os.path.join(class_path, img))
        
        # Collect from test_set
        if os.path.exists(os.path.join(source_path, "test_set", "test_set")):
            base_path = os.path.join(source_path, "test_set", "test_set")
            for class_name in ["cats", "dogs"]:
                class_path = os.path.join(base_path, class_name)
                if os.path.exists(class_path):
                    for img in os.listdir(class_path):
                        all_images[class_name].append(os.path.join(class_path, img))
        
        # Split 80/10/10
        for class_name in ["cats", "dogs"]:
            images = all_images[class_name]
            logger.info(f"Processing {class_name}: {len(images)} total images")
            
            # 80/20 split
            train_imgs, temp_imgs = train_test_split(
                images, test_size=0.2, random_state=SEED
            )
            
            # 50/50 split of remaining 20% into validation and test
            val_imgs, test_imgs = train_test_split(
                temp_imgs, test_size=0.5, random_state=SEED
            )
            
            # Copy files to respective directories
            for img_path in train_imgs:
                try:
                    dst_path = os.path.join(train_dir, class_name, os.path.basename(img_path))
                    shutil.copy2(img_path, dst_path)
                except Exception as e:
                    logger.warning(f"Could not copy {img_path}: {e}")
            
            for img_path in val_imgs:
                try:
                    dst_path = os.path.join(val_dir, class_name, os.path.basename(img_path))
                    shutil.copy2(img_path, dst_path)
                except Exception as e:
                    logger.warning(f"Could not copy {img_path}: {e}")
            
            for img_path in test_imgs:
                try:
                    dst_path = os.path.join(test_dir, class_name, os.path.basename(img_path))
                    shutil.copy2(img_path, dst_path)
                except Exception as e:
                    logger.warning(f"Could not copy {img_path}: {e}")
            
            logger.info(f"{class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
        
        logger.info(f"Dataset organized successfully in {output_dir}")
        return train_dir, val_dir, test_dir
        
    except Exception as e:
        logger.error(f"Error organizing dataset: {e}")
        raise


def main():
    """Main function to prepare dataset"""
    data_dir = "data"
    
    # Configure Kaggle API
    configure_kaggle_api()
    
    # Download and extract dataset to data directory
    zip_path = os.path.join(data_dir, "cat-and-dog.zip")
    if not os.path.exists(zip_path):
        download_dataset(download_dir=data_dir)
    
    training_set_path = os.path.join(data_dir, "training_set")
    test_set_path = os.path.join(data_dir, "test_set")
    
    if not os.path.exists(training_set_path) or not os.path.exists(test_set_path):
        extract_dataset(download_dir=data_dir)
    
    # Organize into 80/10/10 split within data directory
    train_path, val_path, test_path = organize_dataset_80_10_10(source_path=data_dir, output_dir=data_dir)
    
    # Save paths for reference
    paths_info = {
        "train": train_path,
        "validation": val_path,
        "test": test_path,
        "img_size": IMG_SIZE,
        "seed": SEED
    }
    
    logger.info(f"Dataset paths: {paths_info}")
    return paths_info


if __name__ == "__main__":
    main()
