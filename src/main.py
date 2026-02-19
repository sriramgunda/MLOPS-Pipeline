"""
Main training pipeline
Orchestrates the complete machine learning pipeline using modular components
"""

import os
import logging
from pathlib import Path

# Import custom modules
from data_loader import main as prepare_dataset, configure_kaggle_api, organize_dataset_80_10_10
from data_preprocessing import (
    load_train_val_test_datasets,
    create_data_augmentation,
    visualize_sample_batch
)
from train_cnn import train_cnn_pipeline
from predict import Predictor, batch_predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution"""
    
    try:
        logger.info("=" * 80)
        logger.info("Starting Cat vs Dog Classification Pipeline")
        logger.info("=" * 80)
        
        # ============================================================================
        # Step 1: Prepare Dataset
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Data Preparation")
        logger.info("=" * 80)
        
        # Configure Kaggle API
        configure_kaggle_api()
        
        # Prepare dataset with 80/10/10 split
        paths_info = prepare_dataset()
        train_dir = paths_info["train"]
        val_dir = paths_info["validation"]
        test_dir = paths_info["test"]
        
        # ============================================================================
        # Step 2: Data Preprocessing
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("=" * 80)
        
        # Load datasets
        train_ds, val_ds, test_ds, class_names = load_train_val_test_datasets(
            train_dir,
            val_dir,
            test_dir,
            augment_train=True,
            augment_val=False,
            augment_test=False
        )
        
        logger.info(f"Class names: {class_names}")
        
        # Visualize sample batch (optional - may not work in non-interactive environment)
        try:
            visualize_sample_batch(train_ds, class_names, num_images=4)
        except Exception as e:
            logger.warning(f"Could not visualize batch: {e}")
        
        # ============================================================================
        # Step 4: Train CNN Model (MobileNetV2)
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Training CNN Model (MobileNetV2)")
        logger.info("=" * 80)
        
        # Create data augmentation
        data_augmentation = create_data_augmentation()
        
        cnn_model, cnn_history, cnn_test_loss, cnn_test_acc = train_cnn_pipeline(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            data_augmentation=data_augmentation,
            epochs=1
        )
        
        logger.info(f"CNN Model (MobileNetV2) - Test Accuracy: {cnn_test_acc:.4f}")
        
        # ============================================================================
        # Step 5: Model Inference
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Model Inference & Predictions")
        logger.info("=" * 80)
        
        best_model_path = "models/mobilenet_v2.keras"
        logger.info(f"Selected model: {best_model_path} (Accuracy: {cnn_test_acc:.4f})")
        
        # Load predictor
        predictor = Predictor(best_model_path)
        
        # Test predictions on sample images
        logger.info("\nTesting predictions on sample images...")
        
        # Get sample images from test set
        sample_images = []
        test_cats_dir = os.path.join(test_dir, "cats")
        test_dogs_dir = os.path.join(test_dir, "dogs")
        
        if os.path.exists(test_cats_dir):
            cat_samples = [os.path.join(test_cats_dir, f) for f in os.listdir(test_cats_dir)[:2]]
            sample_images.extend(cat_samples)
        
        if os.path.exists(test_dogs_dir):
            dog_samples = [os.path.join(test_dogs_dir, f) for f in os.listdir(test_dogs_dir)[:2]]
            sample_images.extend(dog_samples)
        
        # Make predictions
        for img_path in sample_images:
            try:
                class_name, probability = predictor.predict(img_path)
                logger.info(f"  {img_path}: {class_name} (confidence: {probability:.4f})")
            except Exception as e:
                logger.warning(f"  Could not predict on {img_path}: {e}")
        
        # ============================================================================
        # Step 6: Summary
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 80)
        logger.info(f"CNN Model (MobileNetV2) Test Accuracy: {cnn_test_acc:.4f}")
        logger.info(f"Best Model:                           {best_model_path}")
        logger.info(f"Dataset Path (Train):                 {train_dir}")
        logger.info(f"Dataset Path (Validation):            {val_dir}")
        logger.info(f"Dataset Path (Test):                  {test_dir}")
        logger.info("=" * 80)
        logger.info("Pipeline execution completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
