"""Reusable contour and CNN number-plate recognition pipeline."""

import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .character_segmentation import analyze_character_segmentation
from .model import CLASS_NAMES, IMAGE_SIZE
from .model_loader import load_checkpoint_model
from .plate_detection import analyze_plate


class PlateRecognizer:
    def __init__(self, model_path: Path | None = None):
        self.model: Any = None
        self.model_path = model_path
        if model_path is not None:
            self.model = load_checkpoint_model(model_path)

    @property
    def available(self) -> bool:
        return self.model is not None

    def recognize(
        self, image: np.ndarray, debug: bool = False,
        confidence_threshold: float = 0.60,
    ) -> dict:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return self._failure("invalid_image", "Input image is empty or invalid")
        detection, debug_images = analyze_plate(image)
        result = {
            "success": False,
            "plate_number": "",
            "plate_text": "",
            "characters": [],
            "confidence": 0.0,
            "bounding_box": None,
            "status": "not_detected",
            "message": "No plausible number plate was detected",
            "plate_image": None,
            "characters_image": None,
            "debug_images": {},
        }
        if detection is None:
            if debug:
                result["debug_images"] = {
                    name: self._image_data_uri(value)
                    for name, value in debug_images.items()
                }
                self._save_debug_artifacts(image, image, debug_images, result)
            return result
        result["bounding_box"] = detection.bounding_box
        result["plate_image"] = self._image_data_uri(detection.crop)
        debug_images["plate_crop"] = detection.crop
        characters, segmentation_debug = analyze_character_segmentation(
            detection.crop, debug=debug
        )
        if debug:
            debug_images.update(segmentation_debug)
            self._save_debug_artifacts(image, detection.crop, debug_images, result)
        if not characters:
            result.update(
                status="characters_not_detected",
                message="A plate was detected, but no characters were segmented",
            )
            if debug:
                result["debug_images"] = {
                    name: self._image_data_uri(value)
                    for name, value in debug_images.items()
                }
            return result

        if not self.available:
            result.update(
                status="model_unavailable",
                message="The trained CNN checkpoint is unavailable",
            )
            if debug:
                result["debug_images"] = {
                    name: self._image_data_uri(value)
                    for name, value in debug_images.items()
                }
            return result
        batch = []
        for character in characters:
            gray = character if character.ndim == 2 else cv2.cvtColor(
                character, cv2.COLOR_BGR2GRAY
            )
            gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            batch.append(np.repeat(gray[..., np.newaxis], 3, axis=2))
        batch = np.asarray(batch, dtype="float32") / 255.0
        probabilities = self.model.predict(batch, verbose=0)
        predictions = []
        for scores in probabilities:
            index = int(np.argmax(scores))
            confidence = float(scores[index])
            predictions.append({"character": CLASS_NAMES[index], "confidence": confidence})
        plate_number = "".join(item["character"] for item in predictions).upper()
        confidence = float(np.mean([item["confidence"] for item in predictions]))
        valid_length = 4 <= len(plate_number) <= 12
        valid_characters = plate_number.isalnum()
        confident = (
            valid_length
            and valid_characters
            and all(item["confidence"] >= confidence_threshold for item in predictions)
        )
        result.update(
            success=confident,
            plate_number=plate_number,
            plate_text=plate_number,
            characters=predictions,
            confidence=confidence,
            status="recognized" if confident else "uncertain",
            message=(
                "Plate recognized successfully"
                if confident
                else (
                    "Plate text failed validation"
                    if not valid_length or not valid_characters
                    else "Plate characters were recognized with low confidence"
                )
            ),
        )
        result["characters_image"] = self._characters_data_uri(characters)
        if debug:
            debug_images["segmented_characters"] = cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(result["characters_image"].split(",", 1)[1]),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            result["debug_images"] = {
                name: self._image_data_uri(value)
                for name, value in debug_images.items()
                if value is not None
            }
            self._save_debug_artifacts(image, detection.crop, debug_images, result)
        return result

    @staticmethod
    def _save_debug_artifacts(
        frame: np.ndarray,
        plate: np.ndarray,
        debug_images: dict[str, np.ndarray],
        result: dict,
    ) -> None:
        directory = Path(__file__).resolve().parents[2] / "debug_artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = directory / stamp
        target.mkdir()
        result["debug_artifacts"] = str(target)
        images = {"camera_frame": frame, "plate_crop": plate, **debug_images}
        for name, image in images.items():
            if isinstance(image, np.ndarray):
                cv2.imwrite(str(target / f"{name}.png"), image)

    @staticmethod
    def _failure(status: str, message: str) -> dict:
        return {
            "success": False,
            "plate_number": "",
            "plate_text": "",
            "characters": [],
            "confidence": 0.0,
            "bounding_box": None,
            "status": status,
            "message": message,
            "plate_image": None,
            "characters_image": None,
        }

    @staticmethod
    def _image_data_uri(image: np.ndarray) -> str | None:
        ok, encoded = cv2.imencode(".jpg", image)
        return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}" if ok else None

    @classmethod
    def _characters_data_uri(cls, characters: list[np.ndarray]) -> str | None:
        if not characters:
            return None
        height, width = characters[0].shape[:2]
        canvas = np.zeros((height, width * len(characters)), dtype=np.uint8)
        for index, character in enumerate(characters):
            canvas[:, index * width:(index + 1) * width] = character
        return cls._image_data_uri(canvas)
