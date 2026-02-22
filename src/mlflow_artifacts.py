"""
MLflow artifacts logging utilities
Handles saving and logging visualizations and artifacts to MLflow
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_loss_curves(history, output_dir="artifacts"):
    """
    Generate and save training loss and accuracy curves
    
    Args:
        history: Training history object from model.fit()
        output_dir: Directory to save plots
        
    Returns:
        str: Path to saved figure
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving loss curves: {e}")
        raise


def save_confusion_matrix(y_true, y_pred, class_names, output_dir="artifacts"):
    """
    Generate and save confusion matrix heatmap
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        class_names: List of class names
        output_dir: Directory to save plot
        
    Returns:
        str: Path to saved figure
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {e}")
        raise


def save_classification_report(y_true, y_pred, class_names, output_dir="artifacts"):
    """
    Generate and save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save report
        
    Returns:
        str: Path to saved report
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Save as JSON
        output_path = os.path.join(output_dir, "classification_report.json")
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Classification report saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving classification report: {e}")
        raise


def get_predictions_and_labels(model, dataset, class_names=None):
    """
    Get predictions and true labels from a dataset
    
    Args:
        model: Trained tf.keras.Model
        dataset: tf.data.Dataset
        class_names: List of class names (for label encoding)
        
    Returns:
        tuple: (y_true, y_pred_class)
    """
    try:
        import tensorflow as tf
        
        y_true = []
        y_pred_probs = []
        
        for images, labels in dataset:
            # Get predictions
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds.flatten())
            
            # Get true labels
            if isinstance(labels, tf.Tensor):
                y_true.extend(labels.numpy())
            else:
                y_true.extend(labels)
        
        # Convert probabilities to class labels (threshold at 0.5 for binary)
        y_pred_class = (np.array(y_pred_probs) > 0.5).astype(int)
        y_true = np.array(y_true).astype(int)
        
        return y_true, y_pred_class
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise


def log_mlflow_artifacts(mlflow, history, y_true, y_pred, class_names=None):
    """
    Log all artifacts to MLflow
    
    Args:
        mlflow: MLflow module
        history: Training history
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    try:
        if class_names is None:
            class_names = ['cat', 'dog']
        
        # Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)
        
        # Save and log loss curves
        curves_path = save_loss_curves(history, "artifacts")
        mlflow.log_artifact(curves_path, "metrics")
        
        # Save and log confusion matrix
        cm_path = save_confusion_matrix(y_true, y_pred, class_names, "artifacts")
        mlflow.log_artifact(cm_path, "evaluation")
        
        # Save and log classification report
        report_path = save_classification_report(y_true, y_pred, class_names, "artifacts")
        mlflow.log_artifact(report_path, "evaluation")
        
        logger.info("All artifacts logged to MLflow successfully")
        
    except Exception as e:
        logger.error(f"Error logging artifacts: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("MLflow artifacts module initialized.")
