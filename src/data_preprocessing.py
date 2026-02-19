"""
Data preprocessing module
Handles image preprocessing, augmentation, and normalization
"""

import tensorflow as tf
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


def create_data_augmentation():
    """
    Create data augmentation layers
    
    Returns:
        tf.keras.Sequential: Data augmentation layer
    """
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.1, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ])
    logger.info("Data augmentation layers created.")
    return augmentation


def create_normalization_layer():
    """
    Create normalization layer
    
    Returns:
        tf.keras.layers.Rescaling: Normalization layer
    """
    normalization = layers.Rescaling(1./255)
    logger.info("Normalization layer created.")
    return normalization


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image file
        img_size: Target image size (height, width)
        
    Returns:
        tf.Tensor: Preprocessed image
    """
    # Read image file
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    
    return image


def create_dataset_from_directory(
    directory,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=None,
    subset=None,
    augment=False
):
    """
    Create TensorFlow dataset from directory structure
    
    Args:
        directory: Path to directory containing class subdirectories
        img_size: Target image size
        batch_size: Batch size for dataset
        validation_split: Float between 0 and 1 for validation split
        subset: "training", "validation", or None
        augment: Whether to apply data augmentation
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            seed=SEED,
            image_size=img_size,
            batch_size=batch_size,
            validation_split=validation_split,
            subset=subset
        )
        
        # Apply normalization
        normalization = create_normalization_layer()
        dataset = dataset.map(
            lambda images, labels: (normalization(images), labels),
            num_parallel_calls=AUTOTUNE
        )
        
        # Apply augmentation if requested
        if augment:
            augmentation = create_data_augmentation()
            dataset = dataset.map(
                lambda images, labels: (augmentation(images), labels),
                num_parallel_calls=AUTOTUNE
            )
        
        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        
        logger.info(f"Dataset created with {batch_size} batch size, augment={augment}")
        return dataset
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


def load_train_val_test_datasets(
    train_dir,
    val_dir,
    test_dir,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    augment_train=True,
    augment_val=False,
    augment_test=False
):
    """
    Load training, validation, and test datasets
    
    Args:
        train_dir: Path to training directory
        val_dir: Path to validation directory
        test_dir: Path to test directory
        img_size: Target image size
        batch_size: Batch size
        augment_train: Whether to augment training data
        augment_val: Whether to augment validation data
        augment_test: Whether to augment test data
        
    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    try:
        logger.info("Loading datasets...")
        
        # Load datasets
        train_ds = create_dataset_from_directory(
            train_dir,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment_train
        )
        
        val_ds = create_dataset_from_directory(
            val_dir,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment_val
        )
        
        test_ds = create_dataset_from_directory(
            test_dir,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment_test
        )
        
        # Get class names
        import os
        class_names = sorted(os.listdir(train_dir))
        
        logger.info(f"Datasets loaded. Classes: {class_names}")
        
        return train_ds, val_ds, test_ds, class_names
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise


def visualize_sample_batch(dataset, class_names, num_images=4):
    """
    Visualize a batch of images
    
    Args:
        dataset: TensorFlow dataset
        class_names: List of class names
        num_images: Number of images to display
    """
    try:
        import matplotlib.pyplot as plt
        
        for images, labels in dataset.take(1):
            plt.figure(figsize=(10, 8))
            for i in range(min(num_images, len(images))):
                ax = plt.subplot(2, 2, i + 1)
                
                # Denormalize image for display
                img = images[i].numpy() * 255
                plt.imshow(img.astype("uint8"))
                label_idx = int(labels[i].numpy())
                plt.title(class_names[label_idx])
                plt.axis("off")
            
            plt.tight_layout()
            plt.show()
        
        logger.info("Sample batch visualized.")
        
    except Exception as e:
        logger.error(f"Error visualizing batch: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Data preprocessing module initialized.")
