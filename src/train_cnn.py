"""
CNN model training module
Handles baseline CNN and transfer learning model training with MLflow tracking
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras
import mlflow.tensorflow
import logging
from datetime import datetime

try:
    from mlflow_config import initialize_mlflow, load_experiment_config, start_mlflow_run, end_mlflow_run
    from mlflow_artifacts import (
        get_predictions_and_labels,
        log_mlflow_artifacts,
        save_loss_curves,
        save_confusion_matrix,
        save_classification_report
    )
except ImportError:
    # Fallback if modules not available
    def initialize_mlflow(exp_name=None, uri=None):
        pass
    def load_experiment_config():
        return {}
    def get_predictions_and_labels(*args, **kwargs):
        return None, None
    def log_mlflow_artifacts(*args, **kwargs):
        pass
    def save_loss_curves(*args, **kwargs):
        return None
    def save_confusion_matrix(*args, **kwargs):
        return None
    def save_classification_report(*args, **kwargs):
        return None
    def start_mlflow_run(*args, **kwargs):
        return None
    def end_mlflow_run(*args, **kwargs):
        return None

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (224, 224)
EPOCHS = 3
SEED = 42


def build_baseline_model(
    data_augmentation=None,
    img_size=IMG_SIZE,
    dropout_rate=0.3
):
    """
    Build a baseline CNN model from scratch
    
    Args:
        data_augmentation: Data augmentation layer (optional)
        img_size: Input image size (height, width)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Baseline CNN model
    """
    layers_list = []
    
    if data_augmentation is not None:
        layers_list.append(data_augmentation)
    
    layers_list.extend([
        layers.Rescaling(1./255, input_shape=(*img_size, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model = models.Sequential(layers_list)
    
    logger.info("Baseline CNN model created.")
    return model


def build_cnn_model(
    data_augmentation=None,
    img_size=IMG_SIZE,
    dropout_rate=0.3
):
    """
    Build a CNN model using MobileNetV2 pre-trained architecture
    
    Args:
        data_augmentation: Data augmentation layer (optional)
        img_size: Input image size (height, width)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        tf.keras.Model: MobileNetV2-based CNN model
    """
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Build model
    layers_list = []
    
    if data_augmentation is not None:
        layers_list.append(data_augmentation)
    
    layers_list.extend([
        #layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model = models.Sequential(layers_list)
    
    logger.info("CNN model (MobileNetV2) created.")
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile a model
    
    Args:
        model: tf.keras.Model to compile
        learning_rate: Learning rate for optimizer
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logger.info(f"Model compiled with learning rate: {learning_rate}")


def train_model(
    model,
    train_ds,
    val_ds,
    epochs=EPOCHS,
    model_name="cnn_model"
):
    """
    Train a model with MLflow metrics tracking
    (Assumes MLflow run context is already active)
    
    Args:
        model: tf.keras.Model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of epochs
        model_name: Name of the model for tracking
        
    Returns:
        tuple: (model, history)
    """
    try:
        logger.info(f"Starting training for {model_name}...")
        
        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )
        
        # Extract metrics
        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        # Log metrics (assumes we're in an MLflow run context)
        try:
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_acc)
        except Exception as e:
            logger.warning(f"Could not log metrics to MLflow: {e}")
        
        logger.info(f"Training completed. Val Accuracy: {val_acc:.4f}")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def evaluate_model(model, test_ds, model_name="cnn_model", active_run=True):
    """
    Evaluate a model on test dataset
    
    Args:
        model: Trained tf.keras.Model
        test_ds: Test dataset
        model_name: Name of the model
        active_run: Whether to log to active MLflow run (True) or create new run (False)
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    try:
        logger.info(f"Evaluating {model_name} on test dataset...")
        test_loss, test_accuracy = model.evaluate(test_ds)
        
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Log to MLflow (use existing run if available)
        try:
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
        except Exception as e:
            logger.warning(f"Could not log metrics to MLflow: {e}")
        
        return test_loss, test_accuracy
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def save_model(model, model_path, model_name="model"):
    """
    Save trained model to disk
    
    Args:
        model: Trained tf.keras.Model
        model_path: Path to save model (including filename)
        model_name: Name of the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        model.save(model_path, include_optimizer=False)
        
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_trained_model(model_path):
    """
    Load a trained model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        tf.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def train_and_compare_models(
    train_ds,
    val_ds,
    test_ds,
    data_augmentation=None,
    epochs=EPOCHS,
    mlflow_enabled=True
):
    """
    Train both baseline and MobileNetV2 models, compare performance, and select best
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        data_augmentation: Data augmentation layer
        epochs: Number of epochs
        mlflow_enabled: Whether to use MLflow tracking
        
    Returns:
        dict: Results containing {
            'best_model': trained best model,
            'best_model_name': name of best model,
            'baseline_results': baseline model metrics,
            'mobilenet_results': mobilenet model metrics,
            'histories': training histories,
            'comparison': performance comparison
        }
    """
    results = {
        'baseline_results': {},
        'mobilenet_results': {},
        'histories': {},
        'comparison': {}
    }
    
    models_dict = {}
    
    try:
        # ==================== TRAIN BASELINE MODEL ====================
        logger.info("\n" + "="*60)
        logger.info("TRAINING BASELINE MODEL")
        logger.info("="*60)
        
        baseline_model = build_baseline_model(data_augmentation=data_augmentation)
        baseline_model.summary()
        compile_model(baseline_model, learning_rate=0.001)
        
        if mlflow_enabled:
            try:
                mlflow.start_run(run_name=f"baseline_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                baseline_params = {
                    'epochs': epochs,
                    'model_name': 'baseline_cnn',
                    'img_size': str(IMG_SIZE),
                    'seed': SEED,
                    'learning_rate': 0.001,
                    'optimizer': 'Adam',
                    'loss_function': 'binary_crossentropy'
                }
                mlflow.log_params(baseline_params)
                logger.info("MLflow run started for baseline model")
            except Exception as e:
                logger.warning(f"Could not start MLflow run for baseline: {e}")
        
        baseline_history = baseline_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )
        
        baseline_train_loss = baseline_history.history['loss'][-1]
        baseline_train_acc = baseline_history.history['accuracy'][-1]
        baseline_val_loss = baseline_history.history['val_loss'][-1]
        baseline_val_acc = baseline_history.history['val_accuracy'][-1]
        
        baseline_test_loss, baseline_test_acc = baseline_model.evaluate(test_ds, verbose=0)
        
        results['baseline_results'] = {
            'train_loss': baseline_train_loss,
            'train_accuracy': baseline_train_acc,
            'val_loss': baseline_val_loss,
            'val_accuracy': baseline_val_acc,
            'test_loss': baseline_test_loss,
            'test_accuracy': baseline_test_acc
        }
        results['histories']['baseline'] = baseline_history
        
        logger.info(f"Baseline - Train Acc: {baseline_train_acc:.4f}, Val Acc: {baseline_val_acc:.4f}, Test Acc: {baseline_test_acc:.4f}")
        
        if mlflow_enabled:
            try:
                mlflow.log_metrics({
                    'train_loss': baseline_train_loss,
                    'train_accuracy': baseline_train_acc,
                    'val_loss': baseline_val_loss,
                    'val_accuracy': baseline_val_acc,
                    'test_loss': baseline_test_loss,
                    'test_accuracy': baseline_test_acc
                })
                mlflow.keras.log_model(baseline_model, "model", registered_model_name="baseline_cnn")
            except Exception as e:
                logger.warning(f"Could not log baseline metrics to MLflow: {e}")
        
        models_dict['baseline'] = baseline_model
        
        if mlflow_enabled:
            mlflow.end_run()
        
        # ==================== TRAIN MOBILENET V2 MODEL ====================
        logger.info("\n" + "="*60)
        logger.info("TRAINING MOBILENET V2 MODEL")
        logger.info("="*60)
        
        mobilenet_model = build_cnn_model(data_augmentation=data_augmentation)
        mobilenet_model.summary()
        compile_model(mobilenet_model, learning_rate=0.001)
        
        if mlflow_enabled:
            try:
                mlflow.start_run(run_name=f"mobilenet_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                mobilenet_params = {
                    'epochs': epochs,
                    'model_name': 'mobilenet_v2',
                    'img_size': str(IMG_SIZE),
                    'seed': SEED,
                    'learning_rate': 0.001,
                    'optimizer': 'Adam',
                    'loss_function': 'binary_crossentropy',
                    'transfer_learning': True,
                    'pretrained_weights': 'imagenet'
                }
                mlflow.log_params(mobilenet_params)
                logger.info("MLflow run started for MobileNetV2 model")
            except Exception as e:
                logger.warning(f"Could not start MLflow run for MobileNetV2: {e}")
        
        mobilenet_history = mobilenet_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )
        
        mobilenet_train_loss = mobilenet_history.history['loss'][-1]
        mobilenet_train_acc = mobilenet_history.history['accuracy'][-1]
        mobilenet_val_loss = mobilenet_history.history['val_loss'][-1]
        mobilenet_val_acc = mobilenet_history.history['val_accuracy'][-1]
        
        mobilenet_test_loss, mobilenet_test_acc = mobilenet_model.evaluate(test_ds, verbose=0)
        
        results['mobilenet_results'] = {
            'train_loss': mobilenet_train_loss,
            'train_accuracy': mobilenet_train_acc,
            'val_loss': mobilenet_val_loss,
            'val_accuracy': mobilenet_val_acc,
            'test_loss': mobilenet_test_loss,
            'test_accuracy': mobilenet_test_acc
        }
        results['histories']['mobilenet'] = mobilenet_history
        
        logger.info(f"MobileNetV2 - Train Acc: {mobilenet_train_acc:.4f}, Val Acc: {mobilenet_val_acc:.4f}, Test Acc: {mobilenet_test_acc:.4f}")
        
        if mlflow_enabled:
            try:
                mlflow.log_metrics({
                    'train_loss': mobilenet_train_loss,
                    'train_accuracy': mobilenet_train_acc,
                    'val_loss': mobilenet_val_loss,
                    'val_accuracy': mobilenet_val_acc,
                    'test_loss': mobilenet_test_loss,
                    'test_accuracy': mobilenet_test_acc
                })
                mlflow.keras.log_model(mobilenet_model, "model", registered_model_name="mobilenet_v2")
            except Exception as e:
                logger.warning(f"Could not log MobileNetV2 metrics to MLflow: {e}")
        
        models_dict['mobilenet'] = mobilenet_model
        
        if mlflow_enabled:
            mlflow.end_run()
        
        # ==================== COMPARE MODELS ====================
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison = {
            'baseline_test_acc': baseline_test_acc,
            'mobilenet_test_acc': mobilenet_test_acc,
            'accuracy_diff': abs(mobilenet_test_acc - baseline_test_acc),
            'improvement': ((mobilenet_test_acc - baseline_test_acc) / baseline_test_acc * 100) if baseline_test_acc > 0 else 0
        }
        results['comparison'] = comparison
        
        logger.info(f"\nBaseline Test Accuracy:    {baseline_test_acc:.4f}")
        logger.info(f"MobileNetV2 Test Accuracy: {mobilenet_test_acc:.4f}")
        logger.info(f"Difference:                {comparison['accuracy_diff']:.4f}")
        logger.info(f"Improvement:               {comparison['improvement']:.2f}%")
        
        # Select best model based on test accuracy
        if mobilenet_test_acc >= baseline_test_acc:
            best_model = mobilenet_model
            best_model_name = "mobilenet_v2"
            best_test_acc = mobilenet_test_acc
            logger.info(f"\nBEST MODEL SELECTED: MobileNetV2 (Test Acc: {mobilenet_test_acc:.4f})")
        else:
            best_model = baseline_model
            best_model_name = "baseline_cnn"
            best_test_acc = baseline_test_acc
            logger.info(f"\nBEST MODEL SELECTED: Baseline CNN (Test Acc: {baseline_test_acc:.4f})")
        
        results['best_model'] = best_model
        results['best_model_name'] = best_model_name
        results['best_test_accuracy'] = best_test_acc
        
        # Log comparison to MLflow in the context of best model
        if mlflow_enabled:
            try:
                mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                mlflow.log_metric("baseline_test_accuracy", baseline_test_acc)
                mlflow.log_metric("mobilenet_test_accuracy", mobilenet_test_acc)
                mlflow.log_metric("accuracy_difference", comparison['accuracy_diff'])
                mlflow.log_metric("improvement_percentage", comparison['improvement'])
                mlflow.log_param("best_model", best_model_name)
                mlflow.end_run()
                logger.info("Model comparison logged to MLflow")
            except Exception as e:
                logger.warning(f"Could not log comparison to MLflow: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during model training and comparison: {e}", exc_info=True)
        raise


def train_cnn_pipeline(
    train_ds,
    val_ds,
    test_ds,
    data_augmentation=None,
    epochs=EPOCHS
):
    """
    Complete pipeline for training and comparing CNN models (Baseline + MobileNetV2)
    Selects best model and saves it with MLflow integration
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        data_augmentation: Data augmentation layer
        epochs: Number of epochs
        
    Returns:
        tuple: (best_model, all_results)
    """
    # Load configuration
    config = load_experiment_config()
    mlflow_config = config.get('artifacts', {})
    
    # Initialize MLflow with configuration
    mlflow_enabled = True
    try:
        initialize_mlflow(
            experiment_name=mlflow_config.get('experiment_name', 'cat_dog_classification'),
            tracking_uri=mlflow_config.get('mlflow_uri')
        )
    except Exception as e:
        logger.warning(f"Could not initialize MLflow: {e}")
        logger.warning("Continuing without MLflow tracking...")
        mlflow_enabled = False
    
    # Train and compare models
    results = train_and_compare_models(
        train_ds, val_ds, test_ds,
        data_augmentation=data_augmentation,
        epochs=epochs,
        mlflow_enabled=mlflow_enabled
    )
    
    best_model = results['best_model']
    best_model_name = results['best_model_name']
    best_test_acc = results['best_test_accuracy']
    
    # Generate predictions for evaluation artifacts using the best model
    if mlflow_enabled:
        logger.info("\nGenerating evaluation artifacts for best model...")
        try:
            y_true, y_pred = get_predictions_and_labels(best_model, test_ds)
            
            if y_true is not None and y_pred is not None:
                # Create artifacts directory
                os.makedirs("artifacts", exist_ok=True)
                
                # Log to MLflow in new run for best model
                try:
                    mlflow.start_run(run_name=f"best_model_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    
                    # Log loss curves
                    try:
                        logger.info("Generating training curves...")
                        history = results['histories'].get(best_model_name.replace('_', ''))
                        if history:
                            curves_path = save_loss_curves(history, "artifacts")
                            if curves_path and os.path.exists(curves_path):
                                mlflow.log_artifact(curves_path, "metrics")
                                logger.info("Loss curves logged to MLflow")
                    except Exception as e:
                        logger.warning(f"Could not log loss curves: {e}")
                    
                    # Log confusion matrix
                    try:
                        logger.info("Generating confusion matrix...")
                        cm_path = save_confusion_matrix(y_true, y_pred, ['cat', 'dog'], "artifacts")
                        if cm_path and os.path.exists(cm_path):
                            mlflow.log_artifact(cm_path, "evaluation")
                            logger.info("Confusion matrix logged to MLflow")
                    except Exception as e:
                        logger.warning(f"Could not log confusion matrix: {e}")
                    
                    # Log classification report
                    try:
                        logger.info("Generating classification report...")
                        report_path = save_classification_report(y_true, y_pred, ['cat', 'dog'], "artifacts")
                        if report_path and os.path.exists(report_path):
                            mlflow.log_artifact(report_path, "evaluation")
                            logger.info("Classification report logged to MLflow")
                    except Exception as e:
                        logger.warning(f"Could not log classification report: {e}")
                    
                    # Log best model
                    try:
                        mlflow.keras.log_model(best_model, "model", registered_model_name=best_model_name)
                        logger.info(f"Best model logged to MLflow: {best_model_name}")
                    except Exception as e:
                        logger.warning(f"Could not log best model to MLflow: {e}")
                    
                    mlflow.end_run()
                    
                except Exception as e:
                    logger.warning(f"Error in MLflow artifact logging: {e}")
                    
            else:
                logger.warning("Could not generate evaluation artifacts")
                
        except Exception as e:
            logger.warning(f"Could not generate evaluation artifacts: {e}")
    
    # Save best model to disk
    os.makedirs("models", exist_ok=True)
    if best_model_name == "baseline_cnn":
        model_path = "models/baseline_cnn.keras"
    else:
        model_path = "models/best_model.keras"
    
    save_model(best_model, model_path, best_model_name)
    
    # Log best model artifact
    if mlflow_enabled:
        try:
            mlflow.log_artifact(model_path, "best_model")
        except Exception as e:
            logger.warning(f"Could not log best model artifact: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test Accuracy: {best_test_acc:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("="*60 + "\n")
    
    if mlflow_enabled:
        logger.info("To view MLflow experiments, run:")
        logger.info("  mlflow ui --host localhost --port 5000")
    
    return best_model, results


def main():
    """Prepare data and run the CNN training pipeline as a standalone script."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting standalone CNN training with model comparison...")

    try:
        # Prepare dataset (downloads/extracts/organizes if needed)
        from data_loader import main as prepare_data
        from data_preprocessing import load_train_val_test_datasets, create_data_augmentation

        paths = prepare_data()
        train_dir = paths.get("train")
        val_dir = paths.get("validation")
        test_dir = paths.get("test")

        if not (train_dir and val_dir and test_dir):
            logger.error("Dataset directories not available. Ensure data preparation succeeded.")
            return

        # Build augmentation and datasets
        augmentation = create_data_augmentation()
        train_ds, val_ds, test_ds, class_names = load_train_val_test_datasets(
            train_dir, val_dir, test_dir, augment_train=True
        )

        # Run training pipeline with model comparison
        best_model, results = train_cnn_pipeline(
            train_ds, val_ds, test_ds, data_augmentation=augmentation, epochs=EPOCHS
        )

        best_model_name = results['best_model_name']
        best_test_acc = results['best_test_accuracy']
        
        logger.info(f"\nStandalone training finished!")
        logger.info(f"  Best Model: {best_model_name}")
        logger.info(f"  Test Accuracy: {best_test_acc:.4f}")
        logger.info(f"\nComparison Results:")
        logger.info(f"  {results['comparison']}")

    except Exception as e:
        logger.error(f"Error running standalone training: {e}", exc_info=True)


if __name__ == "__main__":
    main()
