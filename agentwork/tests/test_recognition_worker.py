import threading
import time

import numpy as np

from smartpark.backend.recognition_worker import RecognitionWorker


class FakeCamera:
    def __init__(self):
        self.frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def status(self):
        return {"connected": True}

    def latest_frame(self):
        return self.frame


class FakeRecognizer:
    available = True

    def recognize(self, frame, confidence_threshold=0.6):
        return {
            "status": "recognized",
            "plate_number": "KA01AB1234",
            "confidence": confidence_threshold,
        }


def test_worker_recognizes_without_blocking_camera():
    results = []
    ready = threading.Event()

    def on_result(result):
        results.append(result)
        ready.set()

    worker = RecognitionWorker(
        FakeCamera(), FakeRecognizer(),
        lambda: {"detection_interval_seconds": "0.2"},
        on_result,
    )
    try:
        assert ready.wait(2)
        assert results[0]["plate_number"] == "KA01AB1234"
    finally:
        worker.stop()
        time.sleep(0.05)
