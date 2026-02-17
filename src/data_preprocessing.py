"""
Image Preprocessing Module for Cats vs Dogs Classification
Handles image loading, resizing, augmentation, and normalization
"""
import cv2
import numpy as np
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard image size for model input
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3


def load_and_preprocess_image(image_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """
    Load and preprocess a single image
    - Resize to 224x224
    - Normalize to [0, 1]
    - Handle errors gracefully
    
    Args:
        image_path: Path to image file
        img_height: Target height (default 224)
        img_width: Target width (default 224)
    
    Returns:
        Preprocessed image array or None if loading fails
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img = cv2.resize(img, (img_width, img_height))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None


def apply_data_augmentation(image, augmentation_type="all"):
    """
    Apply data augmentation to image for better generalization
    
    Args:
        image: Input image array (H, W, 3)
        augmentation_type: Type of augmentation ('flip', 'rotate', 'brightness', 'all')
    
    Returns:
        Augmented image array
    """
    if augmentation_type in ["flip", "all"]:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
    
    if augmentation_type in ["rotate", "all"]:
        # Random rotation (-15 to +15 degrees)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
    
    if augmentation_type in ["brightness", "all"]:
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1).astype(np.float32)
    
    # Random Gaussian noise
    if augmentation_type == "all" and np.random.rand() > 0.7:
        noise = np.random.normal(0, 0.01, image.shape)
        image = np.clip(image + noise, 0, 1).astype(np.float32)
    
    return image


def prepare_batch(image_paths, labels, batch_size=32, augment=False):
    """
    Prepare a batch of images for training/inference
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Size of batch
        augment: Whether to apply augmentation
    
    Returns:
        Tuple of (images_array, labels_array)
    """
    images = []
    valid_labels = []
    
    for img_path, label in zip(image_paths, labels):
        img = load_and_preprocess_image(img_path)
        
        if img is not None:
            if augment:
                img = apply_data_augmentation(img)
            
            images.append(img)
            valid_labels.append(label)
    
    if len(images) == 0:
        logger.warning("No valid images loaded in batch")
        return None, None
    
    X = np.array(images, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int32)
    
    return X, y


def validate_image(image_path):
    """
    Validate if image file is readable and has correct format
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(image_path):
            return False
        
        img = Image.open(image_path)
        img.load()  # Verify it can be loaded
        
        # Check if it's a valid image format (RGB or RGBA)
        if img.mode not in ['RGB', 'RGBA', 'L']:
            return False
        
        return True
    
    except Exception as e:
        logger.debug(f"Image validation failed for {image_path}: {str(e)}")
        return False


def prepare_dataset(data_list, augment=False, batch_size=32):
    """
    Prepare full dataset for training
    
    Args:
        data_list: List of tuples (image_path, label, class_name)
        augment: Whether to apply augmentation
        batch_size: Batch size for yielding
    
    Yields:
        Batches of (X, y)
    """
    image_paths = [item[0] for item in data_list]
    labels = [item[1] for item in data_list]
    
    # Validate all images first
    valid_indices = [i for i, path in enumerate(image_paths) if validate_image(path)]
    
    if len(valid_indices) == 0:
        logger.error("No valid images found in dataset")
        return
    
    valid_paths = [image_paths[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    logger.info(f"Prepared {len(valid_paths)} valid images from {len(data_list)} total")
    
    # Yield batches
    for i in range(0, len(valid_paths), batch_size):
        batch_paths = valid_paths[i:i+batch_size]
        batch_labels = valid_labels[i:i+batch_size]
        
        X, y = prepare_batch(batch_paths, batch_labels, batch_size, augment)
        
        if X is not None:
            yield X, y


if __name__ == "__main__":
    # Test preprocessing
    logger.info("Testing image preprocessing functions...")
    
    # Create dummy test image
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, dummy_img)
    
    # Test load and preprocess
    processed_img = load_and_preprocess_image(test_path)
    logger.info(f"Loaded image shape: {processed_img.shape}")
    logger.info(f"Image value range: [{processed_img.min():.2f}, {processed_img.max():.2f}]")
    
    # Test augmentation
    augmented = apply_data_augmentation(processed_img, "all")
    logger.info(f"Augmented image shape: {augmented.shape}")
    
    # Test validation
    is_valid = validate_image(test_path)
    logger.info(f"Image validation: {is_valid}")
    
    # Cleanup
    os.remove(test_path)
    logger.info("Tests completed!")
