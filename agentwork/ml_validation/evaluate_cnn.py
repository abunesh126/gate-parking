"""Evaluate the restored reference CNN on labelled validation characters."""

from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from validation_common import (
    CLASS_ORDER,
    CHECKPOINT,
    DATA_ZIP,
    WORKSPACE,
    build_documented_model,
    restore_model_weights,
)


def class_index(directory_name: str) -> int:
    value = directory_name.removeprefix("class_")
    return int(value) if value.isdigit() else 10 + ord(value) - ord("A")


def documented_input(
    data: bytes, normalize: bool, color_mode: str = "grayscale"
) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode image")
    if color_mode == "grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode != "bgr":
        raise ValueError(f"Unknown color mode: {color_mode}")
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    if normalize:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    return image


def restore_model() -> tf.keras.Model:
    model = build_documented_model()
    restore_model_weights(model)
    return model


def evaluate(
    model: tf.keras.Model,
    samples: list[tuple[str, int, bytes]],
    normalize: bool,
    color_mode: str = "grayscale",
):
    inputs = np.stack(
        [documented_input(data, normalize, color_mode) for _, _, data in samples]
    )
    labels = np.array([label for _, label, _ in samples], dtype=np.int64)
    started = time.perf_counter()
    probabilities = model.predict(inputs, batch_size=32, verbose=0)
    elapsed = time.perf_counter() - started
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((36, 36), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        confusion[actual, predicted] += 1
    per_class = {}
    for index, character in enumerate(CLASS_ORDER):
        total = int(confusion[index].sum())
        per_class[character] = {
            "total": total,
            "correct": int(confusion[index, index]),
            "accuracy": (float(confusion[index, index] / total) if total else None),
        }
    return {
        "normalization": "1./255" if normalize else "none (documented inference)",
        "color_mode": color_mode,
        "total": len(labels),
        "correct": int((predictions == labels).sum()),
        "incorrect": int((predictions != labels).sum()),
        "accuracy": float((predictions == labels).mean()),
        "elapsed_seconds": elapsed,
        "mean_latency_ms": elapsed / len(labels) * 1000 if labels.size else None,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "predictions": [
            {
                "file": name,
                "actual": CLASS_ORDER[actual],
                "predicted": CLASS_ORDER[predicted],
                "confidence": float(probabilities[row, predicted]),
            }
            for row, (name, actual, _) in enumerate(samples)
            for predicted in [int(predictions[row])]
        ],
    }


def main() -> int:
    samples: list[tuple[str, int, bytes]] = []
    with zipfile.ZipFile(DATA_ZIP) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            if len(parts) >= 4 and parts[:2] == ["data", "val"] and parts[2].startswith("class_"):
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append((name, class_index(parts[2]), archive.read(name)))
    if not samples:
        raise RuntimeError("No labelled validation images found")

    model = restore_model()
    results = {
        "dataset": str(DATA_ZIP) + "!data/val",
        "checkpoint": str(CHECKPOINT),
        "class_order": CLASS_ORDER,
        "documented_preprocessing": evaluate(
            model, samples, normalize=False, color_mode="grayscale"
        ),
        "training_scale_preprocessing": evaluate(
            model, samples, normalize=True, color_mode="grayscale"
        ),
        "rgb_raw_comparison": evaluate(
            model, samples, normalize=False, color_mode="rgb"
        ),
        "bgr_raw_comparison": evaluate(
            model, samples, normalize=False, color_mode="bgr"
        ),
    }
    out = WORKSPACE / "ml_validation"
    (out / "validation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "sample_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "actual", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(results["documented_preprocessing"]["predictions"][:20])
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
