"""
CNN model training module
Handles baseline CNN and transfer learning model training with MLflow tracking
"""
import os
# Do not force CPU-only mode here; allow environment to control GPU visibility

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
# Reasonable defaults: short head training + fine-tuning
EPOCHS = 2
INITIAL_EPOCHS = 1
FINE_TUNE_EPOCHS = max(1, EPOCHS - INITIAL_EPOCHS)
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
    
    # Freeze base model weights by default; we'll optionally unfreeze for fine-tuning
    base_model.trainable = False
    
    # Build model
    layers_list = []
    
    # Note: dataset pipeline already applies `Rescaling(1./255)` and optional augmentation.
    # Avoid double-normalization or duplicate augmentation by not adding them here.
    layers_list.extend([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model = models.Sequential(layers_list)

    logger.info("CNN model (MobileNetV2) created.")
    # Return both model and base_model so training pipeline can unfreeze for fine-tuning
    return model, base_model


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
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
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
    
    # Build and compile CNN model (returns model and base_model)
    cnn_model, base_model = build_cnn_model(
        data_augmentation=data_augmentation
    )
    cnn_model.summary()

    # Initial compile for head training
    compile_model(cnn_model, learning_rate=0.001)
    
    # Training with optional MLflow tracking
    run_context = None
    if mlflow_enabled:
        try:
            run_name = f"mobilenet_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            params = {
                'epochs': epochs,
                'model_name': 'mobilenet_v2',
                'img_size': str(IMG_SIZE),
                'seed': SEED,
                'learning_rate': 0.001,
                'optimizer': 'Adam',
                'loss_function': 'binary_crossentropy'
            }
            run_context = start_mlflow_run(run_name=run_name, params=params)
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {e}")
            logger.warning("Continuing training without MLflow...")
            mlflow_enabled = False
            run_context = None
    
    try:
        
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
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False)
        ]

        # Stage 1: train the head (top classifier) with frozen base
        initial_epochs = min(INITIAL_EPOCHS, epochs)
        logger.info(f"Starting head training for {initial_epochs} epochs...")
        history = cnn_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Stage 2: optionally unfreeze some of the base model and fine-tune
        fine_tune_epochs = max(0, epochs - initial_epochs)
        if fine_tune_epochs > 0:
            logger.info("Unfreezing top layers of base model for fine-tuning...")
            # Unfreeze the top N layers of the base model for fine-tuning
            base_model.trainable = True
            # Freeze all layers except last few (to avoid overfitting)
            fine_tune_at = -50
            if abs(fine_tune_at) <= len(base_model.layers):
                for layer in base_model.layers[:fine_tune_at]:
                    layer.trainable = False
            # Re-compile with a lower learning rate
            compile_model(cnn_model, learning_rate=1e-5)

            logger.info(f"Starting fine-tuning for {fine_tune_epochs} epochs...")
            history_fine = cnn_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=initial_epochs + fine_tune_epochs,
                initial_epoch=initial_epochs,
                verbose=1,
                callbacks=callbacks
            )

            # Merge histories
            for k, v in history_fine.history.items():
                if k in history.history:
                    history.history[k].extend(v)
                else:
                    history.history[k] = v
        
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
                end_mlflow_run()
            except Exception as e:
                logger.warning(f"Error closing MLflow run: {e}")
    
    return cnn_model, history, test_loss, test_acc


def main():
    """Prepare data and run the CNN training pipeline as a standalone script."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting standalone CNN training...")

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

        # Run training pipeline
        model, history, test_loss, test_acc = train_cnn_pipeline(
            train_ds, val_ds, test_ds, data_augmentation=augmentation, epochs=EPOCHS
        )

        logger.info(f"Standalone training finished. Test accuracy: {test_acc:.4f}")

    except Exception as e:
        logger.error(f"Error running standalone training: {e}", exc_info=True)


if __name__ == "__main__":
    main()
