"""
Legacy AutoML training module
Can be repurposed for automated model selection and hyperparameter tuning
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


class AutoMLTrainer:
    """
    AutoML trainer for automatic model selection and hyperparameter tuning
    Can be extended for more sophisticated hyperparameter search
    """
    
    def __init__(self, experiment_dir="automl_experiments"):
        """
        Initialize AutoML trainer
        
        Args:
            experiment_dir: Directory to store experiment results
        """
        self.experiment_dir = experiment_dir
        Path(experiment_dir).mkdir(exist_ok=True)
        logger.info(f"AutoML trainer initialized with experiment dir: {experiment_dir}")
    
    def build_model_variant(self, variant_config):
        """
        Build a model based on configuration
        
        Args:
            variant_config: Dictionary with model configuration
            
        Returns:
            tf.keras.Model: Built model
        """
        try:
            model_type = variant_config.get("type", "baseline")
            
            if model_type == "baseline":
                return self._build_baseline(variant_config)
            elif model_type == "transfer":
                return self._build_transfer(variant_config)
            else:
                logger.warning(f"Unknown model type: {model_type}, using baseline")
                return self._build_baseline(variant_config)
        
        except Exception as e:
            logger.error(f"Error building model variant: {e}")
            raise
    
    def _build_baseline(self, config):
        """Build baseline CNN model"""
        conv_filters = config.get("conv_filters", [32, 64, 128])
        dense_units = config.get("dense_units", 128)
        dropout_rate = config.get("dropout_rate", 0.3)
        
        layers_list = [
            layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3))
        ]
        
        # Add convolutional blocks
        for filters in conv_filters:
            layers_list.extend([
                layers.Conv2D(filters, (3, 3), activation='relu'),
                layers.MaxPooling2D(),
            ])
        
        # Add dense layers
        layers_list.extend([
            layers.Flatten(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model = models.Sequential(layers_list)
        logger.info(f"Baseline model built with config: {config}")
        return model
    
    def _build_transfer(self, config):
        """Build transfer learning model"""
        base_model_name = config.get("base_model", "mobilenet_v2")
        dropout_rate = config.get("dropout_rate", 0.3)
        
        # Load base model
        if base_model_name == "mobilenet_v2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == "efficientnet":
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            logger.warning(f"Unknown base model: {base_model_name}, using MobileNetV2")
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])
        
        logger.info(f"Transfer model built with base: {base_model_name}")
        return model
    
    def train_variant(self, model, train_ds, val_ds, variant_name, epochs=5):
        """
        Train a model variant
        
        Args:
            model: tf.keras.Model to train
            train_ds: Training dataset
            val_ds: Validation dataset
            variant_name: Name of the variant
            epochs: Number of epochs
            
        Returns:
            tuple: (model, history, metrics)
        """
        try:
            logger.info(f"Training variant: {variant_name}")
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1
            )
            
            # Extract metrics
            metrics = {
                "train_loss": float(history.history['loss'][-1]),
                "train_accuracy": float(history.history['accuracy'][-1]),
                "val_loss": float(history.history['val_loss'][-1]),
                "val_accuracy": float(history.history['val_accuracy'][-1]),
                "epochs": epochs
            }
            
            logger.info(f"Variant {variant_name} trained. Val Accuracy: {metrics['val_accuracy']:.4f}")
            
            return model, history, metrics
        
        except Exception as e:
            logger.error(f"Error training variant: {e}")
            raise
    
    def evaluate_variant(self, model, test_ds, variant_name):
        """
        Evaluate a model variant
        
        Args:
            model: Trained model
            test_ds: Test dataset
            variant_name: Name of the variant
            
        Returns:
            dict: Test metrics
        """
        try:
            test_loss, test_accuracy = model.evaluate(test_ds)
            metrics = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy)
            }
            
            logger.info(f"Variant {variant_name} - Test Accuracy: {test_accuracy:.4f}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating variant: {e}")
            raise
    
    def run_automl_search(self, train_ds, val_ds, test_ds, num_variants=5):
        """
        Run AutoML search with multiple model variants
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            test_ds: Test dataset
            num_variants: Number of variants to try
            
        Returns:
            dict: AutoML results with best model info
        """
        try:
            logger.info(f"Starting AutoML search with {num_variants} variants...")
            
            # Define model configurations to try
            configs = [
                {
                    "name": "baseline_small",
                    "type": "baseline",
                    "conv_filters": [32, 64],
                    "dense_units": 64,
                    "dropout_rate": 0.2
                },
                {
                    "name": "baseline_medium",
                    "type": "baseline",
                    "conv_filters": [32, 64, 128],
                    "dense_units": 128,
                    "dropout_rate": 0.3
                },
                {
                    "name": "baseline_large",
                    "type": "baseline",
                    "conv_filters": [32, 64, 128, 256],
                    "dense_units": 256,
                    "dropout_rate": 0.4
                },
                {
                    "name": "transfer_mobilenet",
                    "type": "transfer",
                    "base_model": "mobilenet_v2",
                    "dropout_rate": 0.3
                },
                {
                    "name": "transfer_efficientnet",
                    "type": "transfer",
                    "base_model": "efficientnet",
                    "dropout_rate": 0.3
                }
            ]
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "num_variants": num_variants,
                "variants": {}
            }
            
            best_accuracy = -1
            best_variant = None
            
            # Train and evaluate each variant
            for i, config in enumerate(configs[:num_variants]):
                variant_name = config["name"]
                logger.info(f"\n--- Variant {i+1}/{num_variants}: {variant_name} ---")
                
                # Build variant
                model = self.build_model_variant(config)
                
                # Train variant
                model, history, train_metrics = self.train_variant(
                    model, train_ds, val_ds, variant_name, epochs=5
                )
                
                # Evaluate variant
                test_metrics = self.evaluate_variant(model, test_ds, variant_name)
                
                # Record results
                variant_result = {
                    "config": config,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics
                }
                results["variants"][variant_name] = variant_result
                
                # Track best variant
                if test_metrics["test_accuracy"] > best_accuracy:
                    best_accuracy = test_metrics["test_accuracy"]
                    best_variant = variant_name
            
            # Record best variant
            results["best_variant"] = best_variant
            results["best_accuracy"] = best_accuracy
            
            # Save results
            self._save_results(results)
            
            logger.info(f"\nAutoML search completed. Best variant: {best_variant} with accuracy {best_accuracy:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in AutoML search: {e}")
            raise
    
    def _save_results(self, results):
        """Save AutoML results to file"""
        try:
            results_path = Path(self.experiment_dir) / f"automl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def run_automl(train_ds, val_ds, test_ds):
    """
    Convenience function to run AutoML search
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        
    Returns:
        dict: AutoML results
    """
    trainer = AutoMLTrainer()
    results = trainer.run_automl_search(train_ds, val_ds, test_ds, num_variants=5)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("AutoML training module initialized.")
