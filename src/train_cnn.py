"""
CNN model training module
Handles baseline CNN and transfer learning model training with MLflow tracking
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras
import mlflow.tensorflow
import logging
import os
from datetime import datetime

try:
    from mlflow_config import initialize_mlflow, load_experiment_config
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

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (224, 224)
EPOCHS = 1
SEED = 42


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
        layers.Rescaling(1./255),
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


def train_cnn_pipeline(
    train_ds,
    val_ds,
    test_ds,
    data_augmentation=None,
    epochs=EPOCHS
):
    """
    Complete pipeline for training CNN model (MobileNetV2)
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        data_augmentation: Data augmentation layer
        epochs: Number of epochs
        
    Returns:
        tuple: (model, history, test_loss, test_acc)
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
    
    # Build and compile CNN model
    cnn_model = build_cnn_model(
        data_augmentation=data_augmentation
    )
    cnn_model.summary()
    
    compile_model(cnn_model, learning_rate=0.001)
    
    # Training with optional MLflow tracking
    if mlflow_enabled:
        try:
            # Set MLflow experiment 
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'cat_dog_classification'))
            run_context = mlflow.start_run(run_name=f"mobilenet_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {e}")
            logger.warning("Continuing training without MLflow...")
            mlflow_enabled = False
            run_context = None
    else:
        run_context = None
    
    try:
        if mlflow_enabled and run_context:
            run_context.__enter__()
        
        # Log training parameters
        if mlflow_enabled:
            try:
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("model_name", "mobilenet_v2")
                mlflow.log_param("img_size", IMG_SIZE)
                mlflow.log_param("seed", SEED)
                mlflow.log_param("learning_rate", 0.001)
                mlflow.log_param("optimizer", "Adam")
                mlflow.log_param("loss_function", "binary_crossentropy")
            except Exception as e:
                logger.warning(f"Could not log parameters to MLflow: {e}")
        
        # Train model
        logger.info(f"Starting training...")
        history = cnn_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )
        
        # Extract and log training metrics
        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        if mlflow_enabled:
            try:
                mlflow.log_metric("train_loss", train_loss)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("val_loss", val_loss)
                mlflow.log_metric("val_accuracy", val_acc)
            except Exception as e:
                logger.warning(f"Could not log training metrics to MLflow: {e}")
        
        logger.info(f"Training completed. Val Accuracy: {val_acc:.4f}")
        
        # Evaluate on test set and log metrics
        logger.info(f"Evaluating on test dataset...")
        test_loss, test_acc = cnn_model.evaluate(test_ds, verbose=0)
        
        if mlflow_enabled:
            try:
                mlflow.log_metric("test_loss", test_loss)
                mlflow.log_metric("test_accuracy", test_acc)
            except Exception as e:
                logger.warning(f"Could not log test metrics to MLflow: {e}")
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        # Generate predictions for confusion matrix
        if mlflow_enabled:
            logger.info("Generating predictions for evaluation artifacts...")
            try:
                y_true, y_pred = get_predictions_and_labels(cnn_model, test_ds)
                
                if y_true is not None and y_pred is not None:
                    # Create artifacts directory
                    os.makedirs("artifacts", exist_ok=True)
                    
                    # Log loss curves
                    try:
                        logger.info("Generating training curves...")
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
                else:
                    logger.warning("Could not generate evaluation artifacts")
                    
            except Exception as e:
                logger.warning(f"Could not generate evaluation artifacts: {e}")
        
        # Log the model as Keras artifact
        if mlflow_enabled:
            try:
                mlflow.keras.log_model(cnn_model, "model", registered_model_name="mobilenet_v2")
                logger.info(f"Model logged to MLflow: mobilenet_v2")
            except Exception as e:
                logger.warning(f"Could not log model to MLflow: {e}")
        
        # Save model to disk
        os.makedirs("models", exist_ok=True)
        save_model(cnn_model, "models/mobilenet_v2.keras", "mobilenet_v2")
        
        # Log training artifacts
        if mlflow_enabled:
            try:
                mlflow.log_artifact("models/mobilenet_v2.keras", "model")
            except Exception as e:
                logger.warning(f"Could not log model artifact: {e}")
        
        if mlflow_enabled:
            logger.info("Training run completed successfully!")
        else:
            logger.info("Training completed (without MLflow tracking)")
            logger.info("To enable MLflow tracking, ensure MLflow UI is running:")
            logger.info("  mlflow ui --host localhost --port 5000")
        
    finally:
        if mlflow_enabled and run_context:
            try:
                run_context.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing MLflow run: {e}")
    
    return cnn_model, history, test_loss, test_acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("CNN training module initialized.")
