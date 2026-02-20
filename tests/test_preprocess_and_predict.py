import os
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data_preprocessing import load_and_preprocess_image, IMG_SIZE
from src.predict import Predictor, CLASS_NAMES


def test_load_and_preprocess_image_resizes_and_channels(tmp_path):
    # Create a small RGB image and save to temporary path
    img_path = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (100, 80), color=(123, 222, 64))
    img.save(img_path)

    # Call the function under test
    tensor = load_and_preprocess_image(str(img_path), img_size=IMG_SIZE)

    # Ensure returned is a Tensor and has expected spatial shape and 3 channels
    assert isinstance(tensor, tf.Tensor)
    assert tuple(tensor.shape) == (IMG_SIZE[0], IMG_SIZE[1], 3)

    # Values should be in 0-255 range (function does not normalize)
    t_min = int(tf.reduce_min(tensor).numpy())
    t_max = int(tf.reduce_max(tensor).numpy())
    assert 0 <= t_min <= 255
    assert 0 <= t_max <= 255


def test_predictor_predict_from_tensor_returns_class_and_probability(tmp_path):
    # Build and save a tiny binary classification model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")

    model_file = tmp_path / "tiny_model.keras"
    # Save in Keras format
    model.save(str(model_file))

    # Create predictor and a dummy image tensor
    predictor = Predictor(str(model_file))

    # Create a dummy image tensor with correct shape (values 0-255)
    dummy = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    dummy_tensor = tf.convert_to_tensor(dummy)

    class_name, prob = predictor.predict_from_tensor(dummy_tensor)

    # Validate outputs
    assert class_name in CLASS_NAMES
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
