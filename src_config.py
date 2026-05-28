"""
Configuration file for HAR project
Contains all hyperparameters, paths, and constants
Change these values to experiment with the model
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    VISUALIZATIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATASET CONFIGURATION ====================

# Activity classes to use (8 lightweight classes for fast training)
ACTIVITY_CLASSES = [
    "WalkingWithDog",
    "JumpingJack",
    "Punch",
    "Basketball",
    "HorseRiding",
    "PushUps",
    "TaiChi",
    "SoccerJuggling",
]

NUM_CLASSES = len(ACTIVITY_CLASSES)
NUM_CLASSES_FULL = 101  # Total UCF101 classes (for reference)

# Dataset parameters
VIDEO_PER_CLASS = 30  # Videos per class (conservative for free Colab)
TRAIN_SPLIT = 0.7  # 70% training
VAL_SPLIT = 0.15  # 15% validation
TEST_SPLIT = 0.15  # 15% testing

# ==================== VIDEO PREPROCESSING ====================

# Frame extraction parameters
FRAME_SIZE = 112  # Resize all frames to 112x112
# Why 112? Matches MobileNetV2 input
# Smaller = less memory, faster inference
# 112x112 is 25% of 224x224 = ~6.25x less memory

SEQUENCE_LENGTH = 20  # Use 20 frames per video
# ~0.6-1 second of activity captured
# Too short: Miss the action
# Too long: Too much memory

FRAME_SAMPLING_RATE = 2  # Take every 2nd frame from video
# Balances motion capture vs memory

# Normalization (ImageNet statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB std dev
# Why ImageNet stats?
# MobileNetV2 was trained with these
# Using same stats maintains consistency

# Alternative: Normalize to [0, 1] by dividing by 255
USE_IMAGENET_NORM = False  # Set True if using ImageNet statistics
# Set False for simple [0, 1] normalization

# ==================== MODEL ARCHITECTURE ====================

# CNN (MobileNetV2) parameters
CNN_NAME = "MobileNetV2"  # Pre-trained model from ImageNet
FEATURE_DIM = 1280  # Output features from MobileNetV2
# Why MobileNetV2?
# - Only 9 MB (vs ResNet 100+ MB)
# - Optimized for mobile/edge devices
# - Still powerful enough for HAR
# - Fast inference (~10 ms per frame)

# LSTM parameters
LSTM_UNITS = 128  # LSTM hidden units
# Higher = more capacity but slower & more memory
# 128 is good balance for Colab

LSTM_LAYERS = 1  # Number of LSTM layers
# Single layer is good, adding more doesn't help much

# Dense (fully connected) layers
DENSE_UNITS = [64, 32]  # Hidden layer sizes after LSTM
# [64, 32] means:
# LSTM output → 64 units → 32 units → final

# Dropout rate (prevents overfitting)
DROPOUT_RATE = 0.5  # Drop 50% of units during training
# Reduces overfitting on small dataset
# 0.5 is standard, try 0.3-0.5

# ==================== TRAINING PARAMETERS ====================

BATCH_SIZE = 32  # Videos per training step
# 32 is good for Colab free GPU
# Larger = faster but more memory
# Smaller = slower but less memory
# If OOM error: reduce to 16 or 8

EPOCHS = 30  # Maximum training iterations through dataset
# Usually converges in 20-50 epochs
# Use early stopping to stop early

LEARNING_RATE = 1e-4  # Adam optimizer learning rate
# 1e-4 (0.0001) for fine-tuning
# Smaller = slower learning but more stable
# Why small? MobileNetV2 already trained,
# we only adapt last layers

# Learning rate schedule (reduce during training)
LEARNING_RATE_SCHEDULE = {
    10: 5e-5,  # After epoch 10, reduce to 5e-5
    20: 1e-5,  # After epoch 20, reduce to 1e-5
    # Helps fine-tune when loss plateaus
}

# ==================== MODEL BACKBONE OPTIONS ====================
# Select the backbone for feature extraction / video modeling.
# Options: 'MobileNetV2' (default, Keras) or 'r3d_18' (PyTorch 3D ResNet-18)
MODEL_BACKBONE = "MobileNetV2"

# If using r3d_18 (ResNet3D-18), typical input is (C, T, H, W).
# Recommended values for r3d_18: FRAME_SIZE=112, SEQUENCE_LENGTH=16
# You can override these when switching backbones.
R3D_FRAME_SIZE = 112
R3D_SEQUENCE_LENGTH = 16

OPTIMIZER = "adam"  # Adam is best for deep learning
LOSS_FUNCTION = "categorical_crossentropy"  # For multi-class classification
METRICS = ["accuracy"]  # Primary metric

# ==================== REGULARIZATION ====================

L1_REGULARIZATION = 1e-6  # L1 penalty on weights
# Encourages sparsity
# 1e-6 is very weak, standard value

L2_REGULARIZATION = 1e-6  # L2 penalty on weights (weight decay)
# Prevents weights from growing too large
# 1e-6 is standard

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5  # Stop if val_loss doesn't improve for 5 epochs
EARLY_STOPPING_RESTORE_BEST = True  # Restore best weights

# ==================== MIXED PRECISION TRAINING ====================
# GPU optimization for Colab free tier
USE_MIXED_PRECISION = True  # Use float16 for forward pass (faster)
# float32 for loss calculation (stable)
# This can speed up training by 30-50%
# Available on all GPUs

# ==================== DATA AUGMENTATION ====================
# These are applied during training only (not testing)

USE_DATA_AUGMENTATION = True

AUGMENTATION_PARAMS = {
    "rotation_range": 10,  # Random rotation ±10 degrees
    "width_shift_range": 0.1,  # Random horizontal shift 10%
    "height_shift_range": 0.1,  # Random vertical shift 10%
    "zoom_range": 0.1,  # Random zoom 0.9-1.1x
    "horizontal_flip": True,  # Randomly flip horizontally
    "brightness_range": [0.8, 1.2],  # Random brightness 80-120%
    "fill_mode": "nearest",  # Fill missing pixels with nearest
}

# ==================== INFERENCE PARAMETERS ====================

WEBCAM_CONFIDENCE_THRESHOLD = 0.6  # Only show predictions >60% confident
WEBCAM_FPS = 10  # Display 10 frames per second
WEBCAM_BUFFER_SIZE = SEQUENCE_LENGTH  # Frames to buffer before prediction

# ==================== HARDWARE CONFIGURATION ====================

USE_GPU = True  # Use GPU if available
DEVICE = "cuda"  # PyTorch: 'cuda' or 'cpu'
# TensorFlow: automatically detected

# GPU Memory Growth (TensorFlow)
GPU_MEMORY_GROWTH = True  # Don't allocate all GPU memory at start
# Allocate as needed (prevents crashes)

# Mixed precision policy (TensorFlow)
MIXED_PRECISION_POLICY = "mixed_float16"  # Speeds up training 30-50%

# ==================== REPRODUCIBILITY ====================

RANDOM_SEED = 42  # For reproducible results
NUMPY_SEED = 42
TF_SEED = 42
TORCH_SEED = 42

# ==================== LOGGING ====================

LOG_LEVEL = "INFO"  # 'DEBUG', 'INFO', 'WARNING'
LOG_INTERVAL = 10  # Print loss every 10 batches

# ==================== FILE PATHS ====================

MODEL_SAVE_PATH = MODELS_DIR / "best_model.h5"
MODEL_SAVE_PATH_KERAS = MODELS_DIR / "best_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"
TRAINING_LOG_PATH = OUTPUTS_DIR / "training.log"

# ==================== FUNCTION TO PRINT CONFIG ====================


def print_config():
    """Print all configuration values (useful for debugging)"""
    print("\n" + "=" * 60)
    print("HAR PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"\n📊 DATASET:")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(ACTIVITY_CLASSES)})")
    print(f"  Videos per class: {VIDEO_PER_CLASS}")
    print(f"  Train/Val/Test split: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")

    print(f"\n📹 VIDEO PROCESSING:")
    print(f"  Frame size: {FRAME_SIZE}×{FRAME_SIZE}")
    print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"  Sampling rate: Every {FRAME_SAMPLING_RATE}nd frame")

    print(f"\n🧠 MODEL:")
    print(f"  CNN: {CNN_NAME} (feature dim: {FEATURE_DIM})")
    print(f"  LSTM: {LSTM_UNITS} units × {LSTM_LAYERS} layer(s)")
    print(f"  Dense: {' → '.join(map(str, DENSE_UNITS))} → {NUM_CLASSES}")
    print(f"  Dropout: {DROPOUT_RATE}")

    print(f"\n⚙️ TRAINING:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(
        f"  Epochs: {EPOCHS} (with early stopping patience: {EARLY_STOPPING_PATIENCE})"
    )
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  L1 regularization: {L1_REGULARIZATION}")
    print(f"  L2 regularization: {L2_REGULARIZATION}")

    print(f"\n🚀 OPTIMIZATION:")
    print(f"  Mixed precision: {USE_MIXED_PRECISION}")
    print(f"  Data augmentation: {USE_DATA_AUGMENTATION}")
    print(f"  GPU memory growth: {GPU_MEMORY_GROWTH}")

    print(f"\n💾 PATHS:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Outputs: {OUTPUTS_DIR}")
    print("=" * 60 + "\n")


# Call this if running as main
if __name__ == "__main__":
    print_config()
