"""Explainable contour-based license plate localization."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .preprocessing import blur_image, close_gaps, resize_image, to_grayscale


@dataclass
class PlateDetection:
    crop: np.ndarray
    bounding_box: tuple[int, int, int, int]
    score: float


def _candidate_score(area: float, image_area: int, ratio: float, rectangular: bool) -> float:
    area_score = min(1.0, area / max(image_area * 0.08, 1))
    ratio_score = max(0.0, 1.0 - abs(ratio - 4.0) / 5.5)
    return 0.45 * area_score + 0.40 * ratio_score + (0.15 if rectangular else 0.0)


def analyze_plate(
    image: np.ndarray, min_area: int = 400
) -> tuple[Optional[PlateDetection], dict[str, np.ndarray]]:
    """Return a plate candidate and intermediate images for diagnosis."""
    if image is None or image.size == 0:
        raise ValueError("image must contain pixels")
    target_width = min(1280, max(640, image.shape[1]))
    resized = resize_image(image, width=target_width)
    scale = image.shape[1] / resized.shape[1]
    gray = to_grayscale(resized)
    blurred = blur_image(gray, 5)
    edges = cv2.Canny(blurred, 50, 180)
    morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    morphology = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morphology_kernel, iterations=2)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5,
    )
    thresholded = cv2.morphologyEx(
        thresholded, cv2.MORPH_CLOSE, morphology_kernel, iterations=1
    )
    contour_sources = (morphology, thresholded)
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    image_area = resized.shape[0] * resized.shape[1]
    for source in contour_sources:
        contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if area < min(min_area, image_area * 0.002) or area > image_area * 0.75:
                continue
            ratio = width / max(height, 1)
            if not 2.0 <= ratio <= 12.0:
                continue
            if width < max(40, resized.shape[1] * 0.08) or height < max(12, resized.shape[0] * 0.015):
                continue
            rectangular = len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 4
            score = _candidate_score(area, image_area, ratio, rectangular)
            candidates.append((score, (x, y, width, height), contour))
    candidates.sort(key=lambda item: item[0], reverse=True)
    overlay = resized.copy()
    for _, (x, y, width, height), _ in candidates[:30]:
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 165, 255), 2)
    debug = {
        "original": resized,
        "grayscale": gray,
        "blurred": blurred,
        "edges": edges,
        "morphology": morphology,
        "contours": overlay,
    }
    if not candidates:
        return None, debug
    score, (x, y, width, height), _ = candidates[0]
    padding_x = max(2, int(width * 0.03))
    padding_y = max(2, int(height * 0.12))
    x0 = max(0, x - padding_x)
    y0 = max(0, y - padding_y)
    x1 = min(resized.shape[1], x + width + padding_x)
    y1 = min(resized.shape[0], y + height + padding_y)
    cv2.rectangle(debug["contours"], (x0, y0), (x1, y1), (0, 255, 0), 3)
    original_box = (
        int(x0 * scale), int(y0 * scale),
        int((x1 - x0) * scale), int((y1 - y0) * scale),
    )
    ox, oy, ow, oh = original_box
    ox = max(0, min(ox, image.shape[1] - 1))
    oy = max(0, min(oy, image.shape[0] - 1))
    ow = min(ow, image.shape[1] - ox)
    oh = min(oh, image.shape[0] - oy)
    if ow <= 0 or oh <= 0:
        return None, debug
    return PlateDetection(image[oy:oy + oh, ox:ox + ow].copy(), (ox, oy, ow, oh), score), debug


def detect_plate(image: np.ndarray, min_area: int = 400) -> Optional[PlateDetection]:
    """Find the best plausible plate contour and return its original crop."""
    detection, _ = analyze_plate(image, min_area)
    return detection
