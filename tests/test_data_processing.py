"""
Unit tests for image preprocessing functions
Tests data validation, normalization, and augmentation
"""
import pytest
import numpy as np
import os
import cv2
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    load_and_preprocess_image,
    validate_image,
    apply_data_augmentation,
    prepare_batch,
    IMG_HEIGHT,
    IMG_WIDTH
)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image"""
    # Create random image
    img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test_image.jpg")
    
    # Save using PIL
    img = Image.fromarray(img_array)
    img.save(img_path)
    
    return img_path


def test_load_and_preprocess_image(sample_image_path):
    """Test image loading and preprocessing"""
    img = load_and_preprocess_image(sample_image_path)
    
    # Check output shape
    assert img is not None, "Image loading failed"
    assert img.shape == (IMG_HEIGHT, IMG_WIDTH, 3), f"Expected shape {(IMG_HEIGHT, IMG_WIDTH, 3)}, got {img.shape}"
    
    # Check value range (normalized to [0, 1])
    assert img.min() >= 0.0, "Image minimum value should be >= 0"
    assert img.max() <= 1.0, "Image maximum value should be <= 1"
    assert img.dtype == np.float32, f"Expected float32, got {img.dtype}"


def test_load_preset_failure():
    """Test handling of nonexistent image"""
    img = load_and_preprocess_image("nonexistent_image.jpg")
    assert img is None, "Should return None for nonexistent image"


def test_validate_image(sample_image_path):
    """Test image validation"""
    assert validate_image(sample_image_path) is True, "Valid image should pass validation"
    assert validate_image("nonexistent.jpg") is False, "Nonexistent image should fail validation"


def test_apply_data_augmentation(sample_image_path):
    """Test data augmentation"""
    img = load_and_preprocess_image(sample_image_path)
    
    # Test different augmentation types
    for aug_type in ["flip", "rotate", "brightness", "all"]:
        augmented = apply_data_augmentation(img, augmentation_type=aug_type)
        
        # Check output shape unchanged
        assert augmented.shape == img.shape, f"Augmentation {aug_type} changed shape"
        
        # Check value range still valid
        assert 0 <= augmented.min() and augmented.max() <= 1.0, \
            f"Augmentation {aug_type} produced out-of-range values"


def test_prepare_batch(sample_image_path):
    """Test batch preparation"""
    # Prepare batch with one image
    X, y = prepare_batch([sample_image_path], [0], batch_size=1, augment=False)
    
    assert X is not None, "Batch preparation failed"
    assert X.shape[0] == 1, "Batch should contain 1 image"
    assert X.shape[1:] == (IMG_HEIGHT, IMG_WIDTH, 3), "Image shape incorrect"
    assert y[0] == 0, "Label should be 0"


def test_prepare_batch_empty():
    """Test batch preparation with invalid images"""
    X, y = prepare_batch(
        ["nonexistent1.jpg", "nonexistent2.jpg"],
        [0, 1],
        batch_size=2,
        augment=False
    )
    
    assert X is None or len(X) == 0, "Should handle empty/invalid images gracefully"


def test_image_augmentation_randomness(sample_image_path):
    """Test that augmentation produces different results"""
    img = load_and_preprocess_image(sample_image_path)
    
    aug1 = apply_data_augmentation(img, augmentation_type="all")
    aug2 = apply_data_augmentation(img, augmentation_type="all")
    
    # They should be different due to randomness
    # (Extremely unlikely to get identical augmentations)
    assert not np.allclose(aug1, aug2), "Augmentations should be different due to randomness"


def test_pixel_value_range():
    """Test that preprocessing maintains valid pixel value range"""
    # Create test image with specific values
    img_array = np.array([
        [[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        [[64, 64, 64], [192, 192, 192], [200, 200, 200]],
        [[32, 32, 32], [16, 16, 16], [8, 8, 8]]
    ], dtype=np.uint8)
    
    # Save and load
    img = Image.fromarray(img_array)
    img_path = "test_pixel_range.jpg"
    img.save(img_path)
    
    processed = load_and_preprocess_image(img_path)
    
    # Clean up
    os.remove(img_path)
    
    # Check normalization
    assert processed[0, 0, 0] == 0.0, "Black pixel should be 0.0"
    assert processed[0, 2, 0] == 1.0, "White pixel should be 1.0"
    assert 0.4 < processed[0, 1, 0] < 0.6, "Gray pixel (128) should be ~0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

