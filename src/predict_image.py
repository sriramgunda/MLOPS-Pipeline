"""
Prediction Script for Cats vs Dogs Classification
Load trained model and make predictions on new images
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_and_preprocess_image, IMG_HEIGHT, IMG_WIDTH


def load_model(model_path="app/model.h5"):
    """Load trained Keras model"""
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None


def predict_image(image_path, model):
    """
    Predict class for a single image
    
    Args:
        image_path: Path to image file
        model: Loaded Keras model
    
    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess image
    img = load_and_preprocess_image(image_path)
    
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0), verbose=0)
    confidence = float(prediction[0][0])
    predicted_class = "dog" if confidence > 0.5 else "cat"
    certainty = confidence if confidence > 0.5 else (1 - confidence)
    
    return {
        "image": image_path,
        "prediction": predicted_class,
        "confidence": round(certainty, 4),
        "cat_probability": round(1 - confidence, 4),
        "dog_probability": round(confidence, 4)
    }


def predict_batch(image_dir, model, pattern="*.jpg"):
    """
    Predict on all images in a directory
    
    Args:
        image_dir: Directory containing images
        model: Loaded Keras model
        pattern: File pattern to match (default: *.jpg)
    
    Returns:
        List of prediction results
    """
    import glob
    
    results = []
    image_files = glob.glob(os.path.join(image_dir, pattern))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for i, img_path in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {os.path.basename(img_path)}")
        
        result = predict_image(img_path, model)
        if result:
            results.append(result)
    
    return results


def save_results(results, output_file="predictions.json"):
    """Save predictions to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")


def print_results(results):
    """Print results in tabular format"""
    print("\n" + "=" * 80)
    print(f"{'Image':<40} {'Prediction':<10} {'Confidence':<12}")
    print("=" * 80)
    
    for result in results:
        image_name = os.path.basename(result['image'])[:40]
        prediction = result['prediction'].upper()
        confidence = f"{result['confidence']:.4f}"
        
        print(f"{image_name:<40} {prediction:<10} {confidence:>12}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict on images using trained model")
    parser.add_argument("image_path", help="Path to image or directory")
    parser.add_argument("--model", default="app/model.h5", help="Path to model.h5")
    parser.add_argument("--output", default="predictions.json", help="Output JSON file")
    parser.add_argument("--batch", action="store_true", help="Process directory as batch")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    # Get predictions
    if args.batch and os.path.isdir(args.image_path):
        logger.info(f"Batch processing directory: {args.image_path}")
        results = predict_batch(args.image_path, model)
    elif os.path.isfile(args.image_path):
        logger.info(f"Predicting on single image: {args.image_path}")
        result = predict_image(args.image_path, model)
        results = [result] if result else []
    else:
        logger.error(f"Invalid path: {args.image_path}")
        sys.exit(1)
    
    if results:
        # Print results
        print_results(results)
        
        # Save results
        save_results(results, args.output)
    else:
        logger.error("No valid predictions generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
