# HUMAN ACTIVITY RECOGNITION - GOOGLE COLAB NOTEBOOK
# Copy each cell below and paste into Google Colab
# Run cells in order for complete training pipeline

# ============================================================
# CELL 1: SETUP GOOGLE COLAB ENVIRONMENT
# ============================================================

import tensorflow as tf
import torch

print("=" * 60)
print("GPU SETUP")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")
print(f"\nGPU Available: {len(gpus)} GPU(s)")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU Memory Growth enabled")

# ============================================================
# CELL 2: MOUNT GOOGLE DRIVE
# ============================================================

from google.colab import drive
import os

drive_path = "/content/drive"
drive.mount(drive_path)

project_dir = f"{drive_path}/MyDrive/HAR_Project"
os.makedirs(project_dir, exist_ok=True)

print(f"Project directory: {project_dir}")

# ============================================================
# CELL 3: INSTALL REQUIRED PACKAGES
# ============================================================

print("Installing packages...")

import subprocess
import sys

packages = [
    "tensorflow-gpu",
    "opencv-python",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pyyaml",
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Installation complete!")

# ============================================================
# CELL 4: CREATE PROJECT STRUCTURE
# ============================================================

from pathlib import Path

base_path = Path(project_dir)
directories = [
    base_path / "data" / "raw",
    base_path / "data" / "processed",
    base_path / "models",
    base_path / "outputs" / "predictions",
    base_path / "outputs" / "visualizations",
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)

print("Project structure created!")

# ============================================================
# CELL 5: CONFIGURATION
# ============================================================

CONFIG = {
    "ACTIVITY_CLASSES": [
        "WalkingWithDog",
        "JumpingJack",
        "Punch",
        "Basketball",
        "HorseRiding",
        "PushUps",
        "TaiChi",
        "SoccerJuggling",
    ],
    "NUM_CLASSES": 8,
    "VIDEOS_PER_CLASS": 30,
    "FRAME_SIZE": 112,
    "SEQUENCE_LENGTH": 20,
    "FRAME_SAMPLING_RATE": 2,
    "LSTM_UNITS": 128,
    "DENSE_UNITS": [64, 32],
    "DROPOUT_RATE": 0.5,
    "BATCH_SIZE": 32,
    "EPOCHS": 30,
    "LEARNING_RATE": 1e-4,
    "EARLY_STOPPING_PATIENCE": 5,
    "USE_MIXED_PRECISION": True,
    "RANDOM_SEED": 42,
}

print("Configuration loaded!")
print(f"Classes: {len(CONFIG['ACTIVITY_CLASSES'])}")
print(f"Batch size: {CONFIG['BATCH_SIZE']}")
print(f"Epochs: {CONFIG['EPOCHS']}")

# ============================================================
# CELL 6: UTILITY FUNCTIONS
# ============================================================

import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path, num_frames=20, frame_size=112):
    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return None

        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (frame_size, frame_size))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            frame_count += 1

            if len(frames) == num_frames:
                break

        cap.release()

        if len(frames) != num_frames:
            return None

        return np.array(frames)

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return None


print("Utility functions loaded!")

# ============================================================
# CELL 7: BUILD MODEL
# ============================================================

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_har_model(config):
    input_shape = (
        config["SEQUENCE_LENGTH"],
        config["FRAME_SIZE"],
        config["FRAME_SIZE"],
        3,
    )

    inputs = layers.Input(shape=input_shape)

    mobilenet = MobileNetV2(
        input_shape=(config["FRAME_SIZE"], config["FRAME_SIZE"], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    for layer in mobilenet.layers[:-20]:
        layer.trainable = False

    cnn_features = layers.TimeDistributed(mobilenet)(inputs)

    lstm_out = layers.LSTM(
        config["LSTM_UNITS"],
        activation="relu",
        dropout=config["DROPOUT_RATE"],
        recurrent_dropout=0.2,
    )(cnn_features)

    x = lstm_out
    for units in config["DENSE_UNITS"]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(config["DROPOUT_RATE"])(x)

    outputs = layers.Dense(config["NUM_CLASSES"], activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="HAR_MobileNetV2_LSTM")

    return model


model = build_har_model(CONFIG)

optimizer = keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"])
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

print("Model built and compiled!")
print(f"Total parameters: {model.count_params():,}")

# ============================================================
# CELL 8: PREPARE DUMMY DATA FOR TESTING
# ============================================================

X_train = np.random.randn(
    100, CONFIG["SEQUENCE_LENGTH"], CONFIG["FRAME_SIZE"], CONFIG["FRAME_SIZE"], 3
).astype(np.float32)
y_train = np.random.randint(0, CONFIG["NUM_CLASSES"], 100)
y_train_encoded = keras.utils.to_categorical(y_train, CONFIG["NUM_CLASSES"])

X_val = np.random.randn(
    20, CONFIG["SEQUENCE_LENGTH"], CONFIG["FRAME_SIZE"], CONFIG["FRAME_SIZE"], 3
).astype(np.float32)
y_val = np.random.randint(0, CONFIG["NUM_CLASSES"], 20)
y_val_encoded = keras.utils.to_categorical(y_val, CONFIG["NUM_CLASSES"])

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print("Note: Using dummy data for demonstration")

# ============================================================
# CELL 9: SETUP TRAINING CALLBACKS
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(base_path / "models" / "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["EARLY_STOPPING_PATIENCE"],
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    ),
]

print("Callbacks configured!")

# ============================================================
# CELL 10: TRAIN MODEL
# ============================================================

print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    X_train,
    y_train_encoded,
    batch_size=CONFIG["BATCH_SIZE"],
    epochs=CONFIG["EPOCHS"],
    validation_data=(X_val, y_val_encoded),
    callbacks=callbacks,
    verbose=1,
)

print("\nTraining completed!")

# ============================================================
# CELL 11: VISUALIZE TRAINING
# ============================================================

import json

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="Train", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Model Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    str(base_path / "outputs" / "visualizations" / "training_history.png"), dpi=100
)
plt.show()

print("Training plots saved!")

# ============================================================
# CELL 12: MODEL EVALUATION
# ============================================================

print("\nEVALUATING MODEL")
print("=" * 60)

val_loss, val_accuracy = model.evaluate(X_val, y_val_encoded)

print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Validation Loss: {val_loss:.4f}")

# ============================================================
# CELL 13: MAKE PREDICTIONS
# ============================================================

print("\nMAKING PREDICTIONS")
print("=" * 60)

sample_batch = X_val[:5]
predictions = model.predict(sample_batch)

print(f"Input batch shape: {sample_batch.shape}")
print(f"Predictions shape: {predictions.shape}")

print("\nSample predictions:")
for i in range(min(3, len(predictions))):
    print(f"\nSample {i+1}:")
    top_3_idx = np.argsort(predictions[i])[-3:][::-1]
    for idx in top_3_idx:
        class_name = CONFIG["ACTIVITY_CLASSES"][idx]
        confidence = predictions[i][idx]
        print(f"  {class_name}: {confidence:.2%}")

# ============================================================
# CELL 14: SAVE MODEL
# ============================================================

print("\nSAVING MODEL")
print("=" * 60)

model.save(str(base_path / "models" / "final_model.h5"))
print(f"Model saved to: {base_path / 'models' / 'final_model.h5'}")

import pickle

with open(str(base_path / "models" / "classes.pkl"), "wb") as f:
    pickle.dump(CONFIG["ACTIVITY_CLASSES"], f)
print("Classes saved!")

# ============================================================
# CELL 15: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print("\nNext steps:")
print("1. Download trained model: models/final_model.h5")
print("2. Use inference script for predictions")
print("3. Deploy with Streamlit or Gradio")
print("\nAll files saved to Google Drive!")
