"""Segment individual characters from a detected plate crop."""

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import to_grayscale


@dataclass(frozen=True)
class CharacterCandidate:
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray


def segment_characters(
    plate: np.ndarray, size: tuple[int, int] = (28, 28),
    debug: bool = False,
) -> list[np.ndarray]:
    characters, _ = analyze_character_segmentation(plate, size=size, debug=debug)
    return characters


def analyze_character_segmentation(
    plate: np.ndarray, size: tuple[int, int] = (28, 28), debug: bool = False
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    if plate is None or plate.size == 0:
        return [], {}
    gray = to_grayscale(plate)
    height, width = gray.shape[:2]
    plate_ratio = width / max(height, 1)
    if not 2.0 <= plate_ratio <= 8.0:
        return [], {}
    margin_x = max(1, int(width * 0.04))
    margin_y = max(1, int(height * 0.10))
    if width > margin_x * 2 and height > margin_y * 2:
        gray = gray[margin_y:height - margin_y, margin_x:width - margin_x]
    target_height = 96
    target_width = max(120, int(round(gray.shape[1] * target_height / gray.shape[0])))
    resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threshold_images = []
    for thresholded in (otsu, adaptive):
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        threshold_images.extend((cleaned, cv2.bitwise_not(cleaned)))

    candidate_sets = [_find_candidates(image, size) for image in threshold_images]
    best_index = max(
        range(len(candidate_sets)),
        key=lambda index: _candidate_set_score(candidate_sets[index], target_width, target_height),
    )
    best = candidate_sets[best_index]
    debug_images: dict[str, np.ndarray] = {}
    if debug:
        contour_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        for candidate in best:
            cv2.rectangle(
                contour_image,
                (candidate.x, candidate.y),
                (candidate.x + candidate.width, candidate.y + candidate.height),
                (0, 255, 0),
                2,
            )
        final = _character_strip(best)
        debug_images = {
            "grayscale": resized,
            "blurred": blurred,
            "thresholded": threshold_images[best_index],
            "morphology": cv2.morphologyEx(
                threshold_images[best_index],
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            ),
            "character_contours": contour_image,
            "segmented_characters": final,
        }
    return [candidate.image for candidate in best], debug_images


def _find_candidates(binary: np.ndarray, size: tuple[int, int]) -> list[CharacterCandidate]:
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = binary.shape
    found: list[CharacterCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        height_ratio = h / height
        width_ratio = w / width
        aspect_ratio = w / max(h, 1)
        if height_ratio < 0.28 or height_ratio > 0.92:
            continue
        if width_ratio < 0.012 or width_ratio > 0.18:
            continue
        if not 0.08 <= aspect_ratio <= 1.15:
            continue
        if cv2.contourArea(contour) < max(8, height * width * 0.00025):
            continue
        character = binary[y:y + h, x:x + w]
        character = cv2.resize(character, size, interpolation=cv2.INTER_AREA)
        found.append(CharacterCandidate(x, y, w, h, character))
    found.sort(key=lambda item: item.x)
    return _select_character_line(found, width, height)


def _select_character_line(
    candidates: list[CharacterCandidate], width: int, height: int
) -> list[CharacterCandidate]:
    if len(candidates) < 2:
        return []
    median_height = float(np.median([item.height for item in candidates]))
    line = [
        item for item in candidates
        if abs(item.height - median_height) <= max(4, median_height * 0.30)
    ]
    if len(line) < 2:
        return []
    median_center = float(np.median([item.y + item.height / 2 for item in line]))
    line = [
        item for item in line
        if abs(item.y + item.height / 2 - median_center) <= max(6, height * 0.12)
    ]
    return sorted(line, key=lambda item: item.x)


def _candidate_set_score(
    candidates: list[CharacterCandidate], width: int, height: int
) -> float:
    count = len(candidates)
    if count < 2 or count > 12:
        return -100.0
    heights = np.asarray([item.height for item in candidates], dtype=float)
    centers = np.asarray([item.y + item.height / 2 for item in candidates], dtype=float)
    gaps = np.diff([item.x + item.width for item in candidates])
    gap_score = 1.0 if len(gaps) == 0 else 1.0 / (1.0 + float(np.std(gaps) / max(np.mean(gaps), 1)))
    height_score = 1.0 - min(1.0, float(np.std(heights) / max(np.mean(heights), 1)))
    alignment_score = 1.0 - min(1.0, float(np.std(centers) / max(height * 0.12, 1)))
    count_score = 1.0 if 4 <= count <= 10 else 0.6
    return 0.35 * alignment_score + 0.30 * height_score + 0.20 * gap_score + 0.15 * count_score


def _character_strip(candidates: list[CharacterCandidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((28, 28), dtype=np.uint8)
    return np.hstack([candidate.image for candidate in candidates])
