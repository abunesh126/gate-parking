"""Small, reusable OpenCV preprocessing operations."""

import cv2
import numpy as np


def resize_image(image: np.ndarray, width: int = 800) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must contain pixels")
    height = max(1, int(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_image(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def detect_edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(blur_image(gray), 100, 200)


def threshold_image(gray: np.ndarray) -> np.ndarray:
    blurred = blur_image(gray)
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )


def close_gaps(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
