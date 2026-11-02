"""Train and evaluate the 36-class character CNN from the reference archive."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from .model import CLASS_NAMES, IMAGE_SIZE, build_model


def _load_archive(archive_path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to load character data") from exc
    images, labels = [], []
    prefix = f"data/{split}/"
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(item for item in archive.namelist() if item.startswith(prefix) and item.lower().endswith(".jpg")):
            class_name = name.split("/")[-2].replace("class_", "")
            if class_name not in CLASS_NAMES:
                continue
            data = np.frombuffer(archive.read(name), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(cv2.resize(image, IMAGE_SIZE))
                labels.append(CLASS_NAMES.index(class_name))
    if not images:
        raise ValueError(f"No labelled images found for split {split!r} in {archive_path}")
    return np.asarray(images, dtype="float32")[..., np.newaxis] / 255.0, np.asarray(labels)


def train(archive_path: Path, model_dir: Path, epochs: int = 18) -> dict:
    train_images, train_labels = _load_archive(archive_path, "train")
    val_images, val_labels = _load_archive(archive_path, "val")
    model = build_model()
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs, verbose=1)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "character_cnn.keras")
    evaluation = model.evaluate(val_images, val_labels, verbose=0)
    metadata = {
        "classes": list(CLASS_NAMES),
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(val_labels)),
        "epochs": epochs,
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
        "validation_loss": float(evaluation[0]),
        "validation_accuracy": float(evaluation[1]),
        "dataset": str(archive_path),
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("ml/models"))
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.model_dir, args.epochs), indent=2))
