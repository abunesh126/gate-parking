"""Shared paths and model construction for the isolated ML validation."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


WORKSPACE = Path(__file__).resolve().parents[1]
ORIGINAL = WORKSPACE.parent / "License-Plate-Number-Detection"
DATA_ZIP = ORIGINAL / "data.zip"
CHECKPOINT = ORIGINAL / "checkpoints" / "my_checkpoint"
CLASS_ORDER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_documented_model() -> tf.keras.Sequential:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(16, (22, 22), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(32, (16, 16), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (8, 8), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(64, (4, 4), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(36, activation="softmax"),
        ]
    )
    return model


def restore_model_weights(model: tf.keras.Model) -> int:
    """Assign all model kernel/bias tensors from the TensorFlow checkpoint."""
    reader = tf.train.load_checkpoint(str(CHECKPOINT))
    restored = 0
    for layer_index, layer in enumerate(
        layer for layer in model.layers if layer.weights
    ):
        for weight in layer.weights:
            kind = "kernel" if "kernel" in weight.name.lower() else "bias"
            checkpoint_name = (
                f"layer_with_weights-{layer_index}/{kind}/"
                ".ATTRIBUTES/VARIABLE_VALUE"
            )
            value = reader.get_tensor(checkpoint_name)
            if tuple(value.shape) != tuple(weight.shape):
                raise ValueError(
                    f"Shape mismatch for {weight.name}: "
                    f"checkpoint={value.shape}, model={tuple(weight.shape)}"
                )
            weight.assign(value)
            restored += 1
    if restored != len(model.weights):
        raise ValueError(
            f"Only restored {restored} of {len(model.weights)} model tensors"
        )
    return restored
