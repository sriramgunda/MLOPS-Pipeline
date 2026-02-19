"""
Unit tests for model training and inference
Tests MobileNetV2 model building, compilation, and inference
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

class TestModelBuilding:
    """Test CNN model architecture"""
    
    def test_mobilenet_v2_creation(self):
        """Test creating MobileNetV2 model"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        assert base_model is not None
        assert base_model.output_shape == (None, 7, 7, 1280)
    
    def test_model_architecture_valid(self):
        """Test complete model architecture is valid"""
        # Build full model
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.models.Model(inputs, outputs)
        
        assert model is not None
        assert model.output_shape == (None, 1)

class TestModelCompilation:
    """Test model compilation"""
    
    def test_model_compiles(self):
        """Test that model compiles with correct loss and optimizer"""
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.models.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        assert model.optimizer is not None
        assert model.loss is not None

class TestModelInference:
    """Test model inference capabilities"""
    
    def test_single_image_prediction(self):
        """Test prediction on single image"""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Create random image
        img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(img, verbose=0)
        
        assert prediction.shape == (1, 1)
        assert 0.0 <= prediction[0, 0] <= 1.0
    
    def test_batch_prediction(self):
        """Test prediction on batch of images"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Create batch of random images
        batch_size = 8
        imgs = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
        predictions = model.predict(imgs, verbose=0)
        
        assert predictions.shape == (batch_size, 1)
        assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    
    def test_prediction_bounds(self):
        """Test that predictions are in valid range for sigmoid"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pred = model.predict(img, verbose=0)
        
        assert pred[0, 0] >= 0.0
        assert pred[0, 0] <= 1.0

class TestModelPersistence:
    """Test saving and loading models"""
    
    def test_model_save_keras_format(self, temp_data_dir):
        """Test saving model in .keras format"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model_path = Path(temp_data_dir) / "model.keras"
        model.save(str(model_path))
        
        assert model_path.exists()
        assert model_path.suffix == ".keras"
    
    def test_model_load_keras_format(self, temp_data_dir):
        """Test loading model in .keras format"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model_path = Path(temp_data_dir) / "model.keras"
        model.save(str(model_path))
        
        # Load model
        loaded_model = tf.keras.models.load_model(str(model_path))
        
        assert loaded_model is not None
        assert loaded_model.output_shape == model.output_shape

class TestBinaryClassificationMetrics:
    """Test binary classification metrics"""
    
    def test_accuracy_metric(self):
        """Test accuracy metric"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        assert 'accuracy' in model.metrics_names
    
    def test_loss_values_reasonable(self):
        """Test that loss values are reasonable"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        x = np.random.rand(10, 224, 224, 3).astype(np.float32)
        y = np.random.randint(0, 2, (10, 1)).astype(np.float32)
        
        # Train for 1 epoch
        history = model.fit(x, y, epochs=1, verbose=0)
        
        # Check loss is reasonable
        assert 'loss' in history.history
        assert history.history['loss'][0] > 0
