"""
CNN Model Training for Cats vs Dogs Classification with MLflow Tracking
Trains a baseline CNN model and logs experiments to MLflow
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# MLflow for experiment tracking
import mlflow
import mlflow.keras

# Local imports
from data_loader import load_data, get_dataset_info, download_dataset, create_data_splits
from data_preprocessing import prepare_batch, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


def build_baseline_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """
    Build a baseline CNN model for binary image classification
    Architecture:
    - 3 Conv blocks with ReLU + MaxPooling
    - Flatten and Dense layers
    - Binary output (Sigmoid)
    
    Args:
        input_shape: Shape of input images (H, W, C)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer (Binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def build_transfer_learning_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """
    Build a transfer learning model using MobileNetV2 pretrained weights
    Fine-tunes the last layers for binary classification
    
    Args:
        input_shape: Shape of input images (H, W, C)
    
    Returns:
        Compiled Keras model
    """
    # Load pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Add custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def train_model(model_type="baseline", dataset_path="data/cats_and_dogs"):
    """
    Train CNN model with MLflow experiment tracking
    
    Args:
        model_type: 'baseline' or 'transfer_learning'
        dataset_path: Path to dataset directory
    """
    logger.info(f"Starting training with model_type={model_type}")
    
    # Setup MLflow
    mlflow.set_experiment("Cats_vs_Dogs_Classification")
    
    run_name = f"{model_type}_CNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("image_size", f"{IMG_HEIGHT}x{IMG_WIDTH}")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        
        # Load data (attempt to create structure if missing)
        logger.info("Loading dataset...")
        train_data = load_data(dataset_path, "train")
        val_data = load_data(dataset_path, "val")
        test_data = load_data(dataset_path, "test")

        if not train_data or not val_data:
            logger.warning("Dataset splits missing; attempting to create dataset structure for local testing...")
            dataset_path = download_dataset(dataset_path)
            create_data_splits(dataset_path)

            # Retry loading
            train_data = load_data(dataset_path, "train")
            val_data = load_data(dataset_path, "val")
            test_data = load_data(dataset_path, "test")

        if not train_data or not val_data:
            logger.error("Failed to load dataset after attempting to create splits. Please provide dataset at data/cats_and_dogs")
            raise RuntimeError("Dataset not found or empty; please prepare dataset at data/cats_and_dogs")
        
        # Extract paths and labels
        train_paths = [item[0] for item in train_data]
        train_labels = [item[1] for item in train_data]
        
        val_paths = [item[0] for item in val_data]
        val_labels = [item[1] for item in val_data]
        
        test_paths = [item[0] for item in test_data]
        test_labels = [item[1] for item in test_data]
        
        logger.info(f"Train samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Prepare data
        logger.info("Preparing training data...")
        X_train, y_train = prepare_batch(train_paths, train_labels, BATCH_SIZE, augment=True)
        X_val, y_val = prepare_batch(val_paths, val_labels, BATCH_SIZE, augment=False)
        X_test, y_test = prepare_batch(test_paths, test_labels, BATCH_SIZE, augment=False)
        
        # Build model
        logger.info(f"Building {model_type} model...")
        if model_type == "transfer_learning":
            model = build_transfer_learning_model()
        else:
            model = build_baseline_cnn()
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info(model.summary())
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, model_checkpoint],
            verbose=1
        )
        
        # Evaluate
        logger.info("Evaluating model...")
        test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
        
        # Get predictions for detailed metrics
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        y_pred_proba = model.predict(X_test).flatten()
        
        # Log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", test_auc)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        
        # Save artifacts
        os.makedirs("artifacts", exist_ok=True)
        
        # Save model
        model_path = "artifacts/model.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        history_path = "artifacts/training_history.png"
        plt.savefig(history_path)
        mlflow.log_artifact(history_path)
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cat', 'Dog'],
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Log summary
        summary = {
            "model_type": model_type,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "test_auc": float(test_auc),
            "total_epochs_trained": len(history.history['loss']),
            "dataset_info": {
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data)
            }
        }
        
        summary_path = "artifacts/training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        mlflow.log_artifact(summary_path)
        
        logger.info(f"Training completed! Run ID: {run.info.run_id}")
        logger.info(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        
        return model, history


if __name__ == "__main__":
    # Train baseline model
    try:
        model, history = train_model(model_type="baseline")
    except RuntimeError as e:
        logger.error(f"Training aborted: {e}")
        sys.exit(1)
