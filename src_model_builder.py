"""
Model architecture for HAR
Builds MobileNetV2 + LSTM model for video classification

Architecture:
Video (20 frames, 112×112×3)
    ↓
MobileNetV2 (pre-trained on ImageNet)
    ↓
LSTM (temporal sequence learning)
    ↓
Dense layers (classification)
    ↓
Output (8 activity classes)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from src_config import *
from src_utils import logger

# ==================== MODEL BUILDING ====================


def build_model(input_shape=(SEQUENCE_LENGTH, FRAME_SIZE, FRAME_SIZE, 3)):
    """
    Build MobileNetV2 + LSTM model for activity recognition

    Args:
        input_shape: (sequence_length, height, width, channels)
                    Example: (20, 112, 112, 3)

    Returns:
        Compiled Keras model

    Architecture Explanation:
    ├─ TimeDistributed(MobileNetV2): Process each frame independently
    │   └─ Extracts 1280-dim feature vector from each frame
    ├─ LSTM: Learn temporal patterns across 20 frames
    │   └─ Outputs 128-dim vector summarizing motion
    ├─ Dense layers: Refine and classify
    │   └─ 64 → 32 → 8 (activity classes)
    └─ Softmax: Convert to probabilities
    """

    logger.info(f"Building model with input shape: {input_shape}")

    # ==================== INPUT LAYER ====================
    inputs = layers.Input(shape=input_shape)
    logger.info(f"Input shape: {input_shape}")

    # ==================== CNN FEATURE EXTRACTION ====================
    # Load pre-trained MobileNetV2 (ImageNet weights)
    # Why MobileNetV2?
    # - Only 9 MB (vs ResNet 100+ MB)
    # - Fast inference (optimized for mobile)
    # - Still powerful enough for feature extraction
    # - Pre-trained on 1.2M ImageNet images (knows about humans, objects, etc.)

    mobilenet = MobileNetV2(
        input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
        include_top=False,  # Remove classification head
        weights="imagenet",  # Use pre-trained ImageNet weights
        pooling="avg",  # Average pooling instead of flatten
    )

    # Freeze early layers (don't update weights)
    # Why? These layers learned generic features (edges, colors, shapes)
    # that are useful for any vision task. We don't need to retrain them.
    for layer in mobilenet.layers[:-20]:
        layer.trainable = False

    logger.info(f"MobileNetV2 loaded. Froze early {len(mobilenet.layers)-20} layers.")
    logger.info(
        f"Training only last {len([l for l in mobilenet.layers[-20:] if l.trainable])} layers"
    )

    # Apply CNN to each frame using TimeDistributed
    # This means: for each of 20 frames, extract 1280-dim feature vector
    # Input: (batch, 20, 112, 112, 3)
    # Output: (batch, 20, 1280)
    cnn_features = layers.TimeDistributed(mobilenet, name="timeDistributed_cnn")(inputs)
    logger.info(
        f"CNN output shape after TimeDistributed: (batch, {SEQUENCE_LENGTH}, {FEATURE_DIM})"
    )

    # ==================== LSTM TEMPORAL LEARNING ====================
    # LSTM learns temporal patterns across the 20 frames
    # Each frame's features are fed sequentially
    # LSTM maintains hidden state that accumulates information from all previous frames
    # Result: Single 128-dim vector that summarizes the motion pattern

    lstm_out = layers.LSTM(
        LSTM_UNITS,
        activation="relu",  # ReLU activation (allows any positive output)
        return_sequences=False,  # Return only final output (not all timesteps)
        dropout=DROPOUT_RATE,  # Randomly drop 50% of inputs (prevents overfitting)
        recurrent_dropout=0.2,  # Dropout in recurrent connections
        name="lstm_temporal",
    )(cnn_features)

    logger.info(f"LSTM output shape: (batch, {LSTM_UNITS})")

    # Optional: Add second LSTM layer (uncomment if needed)
    # lstm_out = layers.LSTM(64, activation='relu', return_sequences=False)(lstm_out)

    # ==================== DENSE CLASSIFICATION LAYERS ====================
    # After temporal learning, use dense layers to make final classification

    x = lstm_out
    for i, units in enumerate(DENSE_UNITS):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
            name=f"dense_{i+1}",
        )(x)

        x = layers.Dropout(DROPOUT_RATE, name=f"dropout_{i+1}")(x)
        logger.info(f"Dense layer {i+1}: {units} units")

    # ==================== OUTPUT LAYER ====================
    # Softmax converts outputs to probabilities that sum to 1
    # Output: 8 probabilities (one per activity class)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    logger.info(f"Output layer: {NUM_CLASSES} activities (softmax)")

    # ==================== CREATE MODEL ====================
    model = models.Model(inputs=inputs, outputs=outputs, name="HAR_MobileNetV2_LSTM")

    logger.info("Model architecture built successfully!")

    return model


# ==================== MODEL COMPILATION ====================


def compile_model(model):
    """
    Compile model with optimizer, loss, and metrics

    Args:
        model: Keras model

    Explanation:
    - Optimizer (Adam): Updates weights to minimize loss
      - Learning rate 1e-4: Small updates (fine-tuning pre-trained model)
      - Adam: Adaptive learning rate for each parameter
    - Loss (Categorical Crossentropy): Measures how wrong predictions are
      - Lower loss = better model
    - Metrics (Accuracy): Shows % of correct predictions
    """

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=LOSS_FUNCTION,  # Categorical crossentropy for multi-class
        metrics=METRICS,  # Track accuracy
    )

    logger.info(f"Model compiled with:")
    logger.info(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    logger.info(f"  Loss: {LOSS_FUNCTION}")
    logger.info(f"  Metrics: {METRICS}")

    return model


# ==================== MODEL SUMMARY ====================


def print_model_summary(model):
    """Print detailed model architecture"""
    logger.info("\n" + "=" * 70)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 70)

    model.summary()

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    logger.info("=" * 70 + "\n")


# ==================== ALTERNATIVE ARCHITECTURES ====================


def build_model_simpler(input_shape=(SEQUENCE_LENGTH, FRAME_SIZE, FRAME_SIZE, 3)):
    """
    Simpler architecture for faster training on Colab
    Sacrifices some accuracy for speed
    """
    logger.info("Building SIMPLER model (faster, lower accuracy)")

    inputs = layers.Input(shape=input_shape)

    # Use fewer frames
    mobilenet = MobileNetV2(
        input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # Freeze all but last 10 layers
    for layer in mobilenet.layers[:-10]:
        layer.trainable = False

    cnn_features = layers.TimeDistributed(mobilenet)(inputs)

    # Single LSTM with fewer units
    lstm_out = layers.LSTM(64, activation="relu", dropout=0.5)(cnn_features)

    # Fewer dense layers
    x = layers.Dense(32, activation="relu")(lstm_out)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def build_model_powerful(input_shape=(SEQUENCE_LENGTH, FRAME_SIZE, FRAME_SIZE, 3)):
    """
    More powerful architecture with 2 LSTM layers
    Better accuracy but slower training
    """
    logger.info("Building POWERFUL model (slower, higher accuracy)")

    inputs = layers.Input(shape=input_shape)

    mobilenet = MobileNetV2(
        input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    for layer in mobilenet.layers[:-30]:
        layer.trainable = False

    cnn_features = layers.TimeDistributed(mobilenet)(inputs)

    # Two LSTM layers
    lstm_out = layers.LSTM(
        256,
        activation="relu",
        return_sequences=True,
        dropout=0.5,
        recurrent_dropout=0.2,
    )(cnn_features)
    lstm_out = layers.LSTM(128, activation="relu", dropout=0.5)(lstm_out)

    # More dense layers
    x = layers.Dense(128, activation="relu")(lstm_out)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def build_model_r3d18(pretrained=True, device=None):
    """
    Build a PyTorch ResNet3D-18 (r3d_18) pretrained model for video classification.

    Returns a PyTorch model (not a Keras model). This project primarily uses
    Keras/TensorFlow; use this function only if you intend to train/evaluate
    with PyTorch. Make sure `torch` and `torchvision` are installed (see
    requirements.txt).

    Args:
        pretrained (bool): Load pretrained weights if available.
        device (str or torch.device): Device to move the model to. If None,
            will use CUDA if available else CPU.

    Returns:
        torch.nn.Module: r3d_18 model with final `fc` layer adapted to `NUM_CLASSES`.
    """

    try:
        import torch
        from torchvision.models.video import r3d_18
    except Exception as e:
        logger.error(
            "PyTorch or torchvision not available. Install torch & torchvision to use r3d_18."
        )
        raise

    # Build the pretrained r3d_18 model
    try:
        model = r3d_18(pretrained=pretrained)
    except TypeError:
        # Fallback for newer torchvision versions that use `weights=`
        try:
            from torchvision.models.video import R3D_18_Weights

            weights = R3D_18_Weights.DEFAULT if pretrained else None
            model = r3d_18(weights=weights)
        except Exception:
            model = r3d_18(pretrained=pretrained)

    # Replace final fully-connected layer to match our number of classes
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, NUM_CLASSES)

    # Move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info(f"Built r3d_18 model (pretrained={pretrained}) on device={device}")
    logger.info(f"Replaced final fc -> Linear({in_features}, {NUM_CLASSES})")

    return model


# ==================== MODEL INSPECTION ====================


def get_layer_info(model, layer_name):
    """Get information about a specific layer"""
    layer = model.get_layer(layer_name)
    logger.info(f"\nLayer: {layer_name}")
    logger.info(f"Type: {type(layer).__name__}")
    logger.info(f"Trainable: {layer.trainable}")
    if hasattr(layer, "units"):
        logger.info(f"Units: {layer.units}")
    if hasattr(layer, "output_shape"):
        logger.info(f"Output shape: {layer.output_shape}")


def freeze_layer(model, layer_name):
    """Freeze a specific layer (don't update weights)"""
    layer = model.get_layer(layer_name)
    layer.trainable = False
    logger.info(f"Froze layer: {layer_name}")


def unfreeze_layer(model, layer_name):
    """Unfreeze a specific layer (allow weight updates)"""
    layer = model.get_layer(layer_name)
    layer.trainable = True
    logger.info(f"Unfroze layer: {layer_name}")


# ==================== TEST FUNCTION ====================

if __name__ == "__main__":
    """Test model building"""
    logger.info("Testing model building...")

    # Build model
    model = build_model()

    # Print summary
    print_model_summary(model)

    # Compile
    model = compile_model(model)

    logger.info("✓ Model built and compiled successfully!")

    # Test with dummy input
    logger.info("\nTesting with dummy input...")
    dummy_input = np.random.randn(2, SEQUENCE_LENGTH, FRAME_SIZE, FRAME_SIZE, 3).astype(
        np.float32
    )
    output = model.predict(dummy_input)
    logger.info(f"Dummy input shape: {dummy_input.shape}")
    logger.info(f"Model output shape: {output.shape}")
    logger.info(f"Output probabilities: {output[0]}")
    logger.info(f"Predicted class: {np.argmax(output[0])}")
