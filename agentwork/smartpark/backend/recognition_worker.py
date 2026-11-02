"""Periodic camera recognition without blocking the camera reader."""

import threading
import time
from collections.abc import Callable


class RecognitionWorker:
    def __init__(
        self,
        camera,
        recognizer,
        get_settings: Callable[[], dict[str, str]],
        on_result: Callable[[dict], None],
        get_context: Callable[[], tuple[str, int]] | None = None,
    ):
        self.camera = camera
        self.recognizer = recognizer
        self.get_settings = get_settings
        self.on_result = on_result
        self.get_context = get_context
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="smartpark-recognition", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            settings = self.get_settings()
            try:
                interval = max(0.2, float(settings.get("detection_interval_seconds", "2")))
            except ValueError:
                interval = 2.0
            if self.camera.status()["connected"] and self.recognizer.available:
                frame = self.camera.latest_frame()
                if frame is not None:
                    try:
                        try:
                            threshold = float(settings.get(
                                "recognition_confidence_threshold", "0.60"
                            ))
                        except ValueError:
                            threshold = 0.60
                        result = self.recognizer.recognize(
                            frame, confidence_threshold=threshold
                        )
                        if self.get_context is not None:
                            mode, generation = self.get_context()
                            result["_mode"] = mode
                            result["_mode_generation"] = generation
                        self.on_result(result)
                    except Exception as exc:
                        self.on_result({
                            "success": False,
                            "status": "recognition_error",
                            "message": f"Recognition failed: {exc}",
                            "plate_number": "",
                            "characters": [],
                            "confidence": 0.0,
                            "bounding_box": None,
                        })
            self._stop.wait(interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
