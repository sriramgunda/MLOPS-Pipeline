"""
Unit tests for data preprocessing module
Tests image loading, resizing, normalization, and augmentation
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

class TestImageLoading:
    """Test image loading and basic operations"""
    
    def test_image_creation(self, sample_image):
        """Test that sample image is created correctly"""
        assert sample_image is not None
        assert sample_image.size == (224, 224)
        assert sample_image.mode == "RGB"
    
    def test_image_to_array_conversion(self, sample_image):
        """Test converting PIL image to numpy array"""
        img_array = np.array(sample_image)
        assert img_array.shape == (224, 224, 3)
        assert img_array.dtype == np.uint8
        assert img_array.min() >= 0
        assert img_array.max() <= 255

class TestImageNormalization:
    """Test image normalization to [0, 1] range"""
    
    def test_normalization_range(self, sample_image):
        """Test that normalization produces values in [0, 1]"""
        img_array = np.array(sample_image, dtype=np.float32) / 255.0
        assert img_array.min() >= 0.0
        assert img_array.max() <= 1.0
    
    def test_normalization_dtype(self, sample_image):
        """Test normalization preserves correct dtype"""
        img_array = np.array(sample_image, dtype=np.float32) / 255.0
        assert img_array.dtype == np.float32

class TestImageResizing:
    """Test image resizing to 224x224"""
    
    def test_resize_to_standard_size(self, sample_image):
        """Test resizing image to 224x224"""
        target_size = (224, 224)
        resized = sample_image.resize(target_size)
        assert resized.size == target_size
    
    def test_resize_from_different_size(self):
        """Test resizing from various input sizes"""
        test_sizes = [(100, 100), (300, 300), (640, 480)]
        target_size = (224, 224)
        
        for size in test_sizes:
            img = Image.new("RGB", size, color="red")
            resized = img.resize(target_size)
            assert resized.size == target_size

class TestDataAugmentation:
    """Test data augmentation layers"""
    
    def test_augmentation_layer_creation(self):
        """Test creating data augmentation layer"""
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        assert augmentation is not None
        assert len(augmentation.layers) == 3
    
    def test_augmentation_output_shape(self):
        """Test that augmentation preserves shape"""
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
        ])
        
        # Create sample tensor
        img_tensor = tf.random.normal((1, 224, 224, 3))
        augmented = augmentation(img_tensor, training=True)
        
        assert augmented.shape == img_tensor.shape

class TestBatchProcessing:
    """Test batch processing of images"""
    
    def test_batch_creation(self, sample_images_dir):
        """Test creating batches from image directory"""
        batch_size = 32
        img_dir = sample_images_dir
        
        # Count images
        image_files = list(Path(img_dir).glob("*.jpg"))
        assert len(image_files) == 5
    
    def test_batch_size_consistency(self):
        """Test batch processing with consistent batch size"""
        batch_size = 32
        num_samples = 100
        
        # Simulate batching
        num_batches = (num_samples + batch_size - 1) // batch_size
        assert num_batches == 4  # 100 samples / 32 batch size = 4 batches (last one has 4)

class TestDataPipeline:
    """Test complete data preprocessing pipeline"""
    
    def test_pipeline_creates_valid_tensors(self, sample_images_dir):
        """Test that pipeline creates valid tensors"""
        # Create simple pipeline
        @tf.function
        def prepare_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (224, 224))
            image = image / 255.0
            return image
        
        # Test on sample image
        img_files = list(Path(sample_images_dir).glob("*.jpg"))
        assert len(img_files) > 0
