"""
Inference script for predictions
Handles loading saved models and making predictions on new images
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cats", "dogs"]


class Predictor:
    """Predictor class for model inference"""
    
    def __init__(self, model_path):
        """
        Initialize predictor with a trained model
        
        Args:
            model_path: Path to saved model
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image]:
        """
        Preprocess an image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (preprocessed_image_array, original_image)
        """
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            original_img = img.copy()
            
            # Resize
            img = img.resize(IMG_SIZE)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize (0-1 range)
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.info(f"Image preprocessed: {image_path}")
            return img_array, original_img
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Make prediction on an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (class_name, probability)
        """
        try:
            # Preprocess image
            img_array, _ = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            probability = float(prediction[0][0])
            
            # Determine class
            # Assuming binary classification: probability > 0.5 = dogs, else cats
            class_name = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
            
            logger.info(f"{image_path} - Prediction: {class_name}, Probability: {probability:.4f}")
            
            return class_name, probability
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            list: List of (class_name, probability) tuples
        """
        results = []
        for img_path in image_paths:
            try:
                class_name, probability = self.predict(img_path)
                results.append({
                    "image": img_path,
                    "class": class_name,
                    "probability": probability
                })
            except Exception as e:
                logger.warning(f"Could not process {img_path}: {e}")
                results.append({
                    "image": img_path,
                    "class": "error",
                    "probability": None
                })
        
        return results
    
    def predict_from_tensor(self, image_tensor: tf.Tensor) -> Tuple[str, float]:
        """
        Make prediction from a TensorFlow tensor
        
        Args:
            image_tensor: TensorFlow image tensor
            
        Returns:
            tuple: (class_name, probability)
        """
        try:
            # Ensure proper shape
            if len(image_tensor.shape) == 3:
                image_tensor = tf.expand_dims(image_tensor, axis=0)
            
            # Normalize if needed
            if tf.reduce_max(image_tensor) > 1.0:
                image_tensor = image_tensor / 255.0
            
            # Make prediction
            prediction = self.model.predict(image_tensor, verbose=0)
            probability = float(prediction[0][0])
            
            # Determine class
            class_name = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
            
            return class_name, probability
        
        except Exception as e:
            logger.error(f"Error making prediction from tensor: {e}")
            raise
    
    def get_class_names(self) -> list:
        """Get list of class names"""
        return CLASS_NAMES


def load_and_predict(model_path: str, image_path: str) -> Tuple[str, float]:
    """
    Convenience function to load model and make prediction
    
    Args:
        model_path: Path to saved model
        image_path: Path to image
        
    Returns:
        tuple: (class_name, probability)
    """
    predictor = Predictor(model_path)
    return predictor.predict(image_path)


def batch_predict(model_path: str, image_dir: str) -> list:
    """
    Make predictions on all images in a directory
    
    Args:
        model_path: Path to saved model
        image_dir: Directory containing images
        
    Returns:
        list: Prediction results
    """
    try:
        predictor = Predictor(model_path)
        image_dir = Path(image_dir)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        image_paths = [
            str(p) for p in image_dir.rglob('*')
            if p.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        
        # Make predictions
        results = predictor.predict_batch(image_paths)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Inference script initialized.")
    
    # Example usage
    if __name__ == "__main__":
        print("This module provides inference capabilities for the trained model.")
        print("\nUsage example:")
        print("  from predict import Predictor")
        print("  predictor = Predictor('models/mobilenet_v2.keras')")
        print("  class_name, probability = predictor.predict('path/to/image.jpg')")
