import cv2
import numpy as np

from smartpark.ml.character_segmentation import segment_characters
from smartpark.ml.preprocessing import detect_edges, to_grayscale


def test_preprocessing_and_character_segmentation():
    image = np.zeros((80, 240, 3), dtype=np.uint8)
    cv2.putText(image, "AB12", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    assert detect_edges(to_grayscale(image)).shape == (80, 240)
    assert len(segment_characters(image)) >= 2


def test_segmentation_rejects_non_plate_crop_aspect_ratio():
    image = np.zeros((160, 232, 3), dtype=np.uint8)
    cv2.putText(image, "LI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    assert segment_characters(image) == []
