"""
Converted from src/CatDog.ipynb

Note: cells that used shell/Colab magics are left as comments or converted
to equivalent Python where appropriate.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# -- Configuration / Paths -------------------------------------------------
TRAIN_PATH = "training_set/training_set"
TEST_PATH = "test_set/test_set"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# -- Utility: show random images -------------------------------------------
def show_random_images(folder_path, num_images=4):
    class_names = os.listdir(folder_path)

    plt.figure(figsize=(10, 8))

    for i in range(num_images):
        class_name = random.choice(class_names)
        class_folder = os.path.join(folder_path, class_name)
        image_name = random.choice(os.listdir(class_folder))
        image_path = os.path.join(class_folder, image_name)

        img = Image.open(image_path)

        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# -- Data pipeline ---------------------------------------------------------
def build_datasets(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

# -- Data augmentation & normalization ------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

normalization_layer = layers.Rescaling(1.0 / 255)

# -- Models ---------------------------------------------------------------
def build_baseline_model():
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    return model

def build_transfer_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    return model

# -- Prediction helper ----------------------------------------------------
def predict_image(model, path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = "Dog" if prob > 0.5 else "Cat"

    return prob, label

# -- Main execution (runs the notebook flow) ------------------------------
def main():
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

    # If dataset folders exist, show sample images and build datasets
    if os.path.isdir(TRAIN_PATH):
        try:
            print("Showing sample images from:", TRAIN_PATH)
            show_random_images(TRAIN_PATH)
        except Exception as e:
            print("Could not show sample images:", e)

    if not os.path.isdir(TRAIN_PATH) or not os.path.isdir(TEST_PATH):
        print("Warning: TRAIN_PATH or TEST_PATH not found. Update paths and retry.")
        print("If you used the original notebook's Kaggle steps, re-run dataset preparation manually.")
        # continue, but building datasets will fail if missing

    # Build datasets (may raise if paths do not exist)
    train_ds, val_ds, test_ds = build_datasets()

    # Baseline model
    baseline_model = build_baseline_model()
    baseline_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("Training baseline model...")
    baseline_history = baseline_model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Transfer learning model
    transfer_model = build_transfer_model()
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print("Training transfer model (MobileNetV2)...")
    transfer_history = transfer_model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Evaluate on test set
    test_loss, test_acc = transfer_model.evaluate(test_ds)
    print("MobileNet Test Accuracy:", test_acc)

    # Save model
    os.makedirs("models", exist_ok=True)
    MODEL_PATH = "models/cats_dogs_mobilenet.keras"
    transfer_model.save(MODEL_PATH, include_optimizer=False)
    print("Model saved to:", MODEL_PATH)

    # Reload and test prediction helper
    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    # If example images exist, run quick predictions
    example_cat = os.path.join(TRAIN_PATH, "cats", "cat.1.jpg")
    example_dog = os.path.join(TRAIN_PATH, "dogs", "dog.1.jpg")

    if os.path.exists(example_cat):
        prob, label = predict_image(loaded_model, example_cat)
        print("Cat image probability:", prob, "Prediction:", label)

    if os.path.exists(example_dog):
        prob, label = predict_image(loaded_model, example_dog)
        print("Dog image probability:", prob, "Prediction:", label)


if __name__ == "__main__":
    main()
