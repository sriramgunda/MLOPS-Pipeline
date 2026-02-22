"""
Pytest configuration and fixtures
"""

import sys
import os
# Ensure project root is on sys.path so `import src` works when running pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple 224x224 RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

@pytest.fixture
def sample_images_dir(temp_data_dir):
    """Create directory with sample images for testing"""
    img_dir = Path(temp_data_dir) / "images"
    img_dir.mkdir(exist_ok=True)
    
    # Create 5 sample images
    for i in range(5):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(str(img_dir / f"test_{i}.jpg"))
    
    return str(img_dir)
