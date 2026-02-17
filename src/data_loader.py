"""
Data Loader for Cats vs Dogs Classification Dataset
Downloads and organizes the dataset for model training
"""
import os
import shutil
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def download_dataset(save_dir="data"):
    """
    Downloads Microsoft Cats vs Dogs dataset
    Source: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
    
    For local development, you can also use TensorFlow datasets:
    from tensorflow.keras.datasets import cifar10
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset_path = os.path.join(save_dir, "cats_and_dogs")
    
    # Check if dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    print("Dataset download instructions:")
    print("1. Option A (Automatic - requires internet & space):")
    print("   - Download from: https://www.microsoft.com/download/confirmation.aspx?id=54765")
    print("   - Extract to 'data/cats_and_dogs' folder")
    print("\n2. Option B (TensorFlow - Quick for testing):")
    print("   - The load_data() function will create synthetic data structure")
    
    # Create directory structure for local testing
    os.makedirs(os.path.join(dataset_path, "train", "cats"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "train", "dogs"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "val", "cats"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "val", "dogs"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "test", "cats"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "test", "dogs"), exist_ok=True)
    
    print(f"Created directory structure at {dataset_path}")
    return dataset_path


def create_data_splits(dataset_path="data/cats_and_dogs"):
    """
    Organizes dataset into train (80%), validation (10%), and test (10%) splits
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Get all image files
    raw_path = os.path.join(dataset_path, "raw") if os.path.exists(os.path.join(dataset_path, "raw")) else None
    
    if raw_path:
        print(f"Organizing images from {raw_path} into train/val/test splits...")
        
        for animal_class in ["cats", "dogs"]:
            class_path = os.path.join(raw_path, animal_class)
            if not os.path.exists(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            # 80/10/10 split
            train_imgs, val_test = train_test_split(images, test_size=0.2, random_state=42)
            val_imgs, test_imgs = train_test_split(val_test, test_size=0.5, random_state=42)
            
            # Copy to respective folders
            for img in train_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(dataset_path, "train", animal_class, img)
                shutil.copy2(src, dst)
            
            for img in val_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(dataset_path, "val", animal_class, img)
                shutil.copy2(src, dst)
            
            for img in test_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(dataset_path, "test", animal_class, img)
                shutil.copy2(src, dst)
        
        print("Dataset splits created successfully")
    else:
        print(f"Raw images not found in {raw_path}. Ensure raw images are in {os.path.join(dataset_path, 'raw')}")


def load_data(dataset_path="data/cats_and_dogs", split="train"):
    """
    Loads dataset and returns image paths with labels
    
    Args:
        dataset_path: Path to dataset directory
        split: 'train', 'val', or 'test'
    
    Returns:
        List of tuples: (image_path, label) where label is 0 for cats, 1 for dogs
    """
    split_path = os.path.join(dataset_path, split)
    
    if not os.path.exists(split_path):
        print(f"Split path not found: {split_path}")
        return []
    
    data = []
    
    # Load cat images (label 0)
    cats_path = os.path.join(split_path, "cats")
    if os.path.exists(cats_path):
        for img_file in os.listdir(cats_path):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                data.append((os.path.join(cats_path, img_file), 0, "cat"))
    
    # Load dog images (label 1)
    dogs_path = os.path.join(split_path, "dogs")
    if os.path.exists(dogs_path):
        for img_file in os.listdir(dogs_path):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                data.append((os.path.join(dogs_path, img_file), 1, "dog"))
    
    print(f"Loaded {len(data)} images from {split} split")
    return data


def get_dataset_info(dataset_path="data/cats_and_dogs"):
    """
    Returns dataset statistics
    """
    info = {}
    for split in ["train", "val", "test"]:
        data = load_data(dataset_path, split)
        cats = sum(1 for _, label, _ in data if label == 0)
        dogs = sum(1 for _, label, _ in data if label == 1)
        info[split] = {"total": len(data), "cats": cats, "dogs": dogs}
    
    return info


if __name__ == "__main__":
    # Setup dataset
    dataset_path = download_dataset()
    create_data_splits(dataset_path)
    
    # Display statistics
    info = get_dataset_info(dataset_path)
    for split, stats in info.items():
        print(f"\n{split.upper()} Split:")
        print(f"  Total: {stats['total']}, Cats: {stats['cats']}, Dogs: {stats['dogs']}")
