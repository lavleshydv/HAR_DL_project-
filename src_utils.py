"""
Utility functions for HAR project
Common functions used across the pipeline
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from src_config import *

# ==================== LOGGING SETUP ====================


def setup_logging():
    """Configure logging for the project"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=LOG_LEVEL,
        format=log_format,
        handlers=[logging.FileHandler(TRAINING_LOG_PATH), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


logger = setup_logging()

# ==================== FILE I/O UTILITIES ====================


def save_json(data, filepath):
    """Save data as JSON file"""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_pickle(data, filepath):
    """Save data as pickle file"""
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath):
    """Load pickle file"""
    import pickle

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded pickle from {filepath}")
    return data


# ==================== IMAGE PROCESSING ====================


def resize_frame(frame, size=FRAME_SIZE):
    """
    Resize frame to specified size

    Args:
        frame: Input image (H, W, 3)
        size: Target size (size, size)

    Returns:
        Resized frame (size, size, 3)
    """
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_frame(frame, method="minmax"):
    """
    Normalize frame to [0, 1] range

    Args:
        frame: Input image (H, W, 3) with values [0, 255]
        method: 'minmax' for [0,1] or 'imagenet' for ImageNet normalization

    Returns:
        Normalized frame
    """
    frame = frame.astype(np.float32)

    if method == "minmax":
        # Simple min-max normalization to [0, 1]
        normalized = frame / 255.0
    elif method == "imagenet":
        # ImageNet normalization
        frame = frame / 255.0
        normalized = np.zeros_like(frame)
        for i in range(3):  # For R, G, B channels
            normalized[:, :, i] = (frame[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    else:
        normalized = frame / 255.0

    return normalized


def preprocess_frame(frame, normalize=True):
    """
    Complete preprocessing for a single frame

    Args:
        frame: Input image
        normalize: Whether to normalize

    Returns:
        Preprocessed frame (FRAME_SIZE, FRAME_SIZE, 3), [0, 1]
    """
    # Resize
    frame = resize_frame(frame, FRAME_SIZE)

    # Normalize
    if normalize:
        frame = normalize_frame(frame, method="minmax")

    return frame


# ==================== VIDEO PROCESSING ====================


def extract_frames_from_video(
    video_path, sequence_length=SEQUENCE_LENGTH, sampling_rate=FRAME_SAMPLING_RATE
):
    """
    Extract equally-spaced frames from a video

    Args:
        video_path: Path to video file
        sequence_length: Number of frames to extract
        sampling_rate: Extract every Nth frame

    Returns:
        List of frames (sequence_length, FRAME_SIZE, FRAME_SIZE, 3)
        or None if extraction fails

    Example:
        If video has 150 frames:
        - sampling_rate=2 means use every 2nd frame → 75 available
        - Extract 20 equally-spaced frames
        - Result: [Frame0, Frame7, Frame15, ..., Frame142]
    """
    frames = []

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If less than sequence_length * sampling_rate, can't extract
        if total_frames < sequence_length * sampling_rate:
            logger.warning(f"Video too short: {video_path} ({total_frames} frames)")
            return None

        # Calculate which frames to extract
        indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)

        frame_idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Check if this is a frame we want
            if frame_idx in indices:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Preprocess
                frame = preprocess_frame(frame)
                frames.append(frame)

            frame_idx += 1

            # Early exit if we have all frames
            if len(frames) == sequence_length:
                break

        cap.release()

        if len(frames) != sequence_length:
            logger.warning(
                f"Could only extract {len(frames)}/{sequence_length} frames from {video_path}"
            )
            return None

        return np.array(frames)

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None


def create_sequences(frames_list, sequence_length=SEQUENCE_LENGTH):
    """
    Create overlapping sequences from a list of frames

    Args:
        frames_list: List of frames (N, H, W, 3)
        sequence_length: Length of each sequence

    Returns:
        List of sequences, each of shape (sequence_length, H, W, 3)

    Example:
        If you have 100 frames and sequence_length=20:
        - Sequence 1: frames [0-19]
        - Sequence 2: frames [10-29] (overlapping by 10)
        - Sequence 3: frames [20-39]
        - ...
    """
    sequences = []
    stride = sequence_length // 2  # 50% overlap

    for i in range(0, len(frames_list) - sequence_length, stride):
        sequence = frames_list[i : i + sequence_length]
        sequences.append(sequence)

    return sequences


# ==================== DATA VALIDATION ====================


def validate_dataset():
    """Check if dataset exists and has required structure"""
    logger.info("Validating dataset structure...")

    # Check if raw data directory exists
    if not RAW_DATA_DIR.exists():
        logger.warning(f"Raw data directory not found: {RAW_DATA_DIR}")
        logger.info("Download dataset first using dataset download script")
        return False

    # Check for activity class directories
    missing_classes = []
    for activity_class in ACTIVITY_CLASSES:
        class_dir = RAW_DATA_DIR / activity_class
        if not class_dir.exists():
            missing_classes.append(activity_class)

    if missing_classes:
        logger.warning(f"Missing activity classes: {missing_classes}")
        return False

    logger.info("✓ Dataset structure is valid")
    return True


def get_dataset_statistics():
    """Get statistics about the dataset"""
    stats = {}

    for activity_class in ACTIVITY_CLASSES:
        class_dir = RAW_DATA_DIR / activity_class
        if class_dir.exists():
            video_files = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
            stats[activity_class] = len(video_files)

    logger.info("Dataset Statistics:")
    for activity, count in stats.items():
        logger.info(f"  {activity}: {count} videos")

    return stats


# ==================== VISUALIZATION ====================


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss curves)

    Args:
        history: Training history dict with keys like 'loss', 'val_loss', etc.
        save_path: Path to save figure (optional)

    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    axes[0].plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Model Accuracy Over Time", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("Model Loss Over Time", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved training history plot to {save_path}")

    return fig


def plot_confusion_matrix(cm, class_names=ACTIVITY_CLASSES, save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix (N×N)
        class_names: List of class names
        save_path: Path to save figure

    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_class_distribution(labels, save_path=None):
    """
    Plot distribution of classes in dataset

    Args:
        labels: List of class labels
        save_path: Path to save figure

    Returns:
        matplotlib figure object
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(unique)), counts, color="steelblue")

    ax.set_xlabel("Activity Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Videos", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(unique, rotation=45, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved class distribution to {save_path}")

    return fig


# ==================== MEMORY UTILITIES ====================


def get_memory_usage():
    """Get current memory usage info"""
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident set size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual memory
        }
    except:
        return None


def get_gpu_memory_usage():
    """Get GPU memory usage (for TensorFlow)"""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Get memory stats
            logger.info(f"GPU available: {len(gpus)} GPU(s)")
            return len(gpus)
    except:
        pass
    return 0


# ==================== METRICS UTILITIES ====================


def calculate_metrics(y_true, y_pred, class_names=ACTIVITY_CLASSES):
    """
    Calculate evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def print_metrics(y_true, y_pred, class_names=ACTIVITY_CLASSES):
    """Print evaluation metrics nicely"""
    metrics = calculate_metrics(y_true, y_pred, class_names)

    logger.info("=" * 50)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 50)
    logger.info(
        f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
    )
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info("=" * 50)


# ==================== REPRODUCIBILITY ====================


def set_seeds():
    """Set all random seeds for reproducibility"""
    import random
    import tensorflow as tf

    random.seed(RANDOM_SEED)
    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TF_SEED)

    logger.info(f"Random seeds set to {RANDOM_SEED}")


# ==================== TEST FUNCTION ====================

if __name__ == "__main__":
    """Test utilities"""
    print_config()
    logger.info("Utility functions loaded successfully!")
