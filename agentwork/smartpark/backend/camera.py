"""Windows camera discovery and one generation-safe capture loop."""

import threading
import time
import re

import cv2


class CameraManager:
    _BACKENDS = (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("OpenCV default", cv2.CAP_ANY),
    )
    _BACKEND_BY_NAME = dict(_BACKENDS)

    def __init__(self):
        self._lock = threading.RLock()
        self.capture: cv2.VideoCapture | None = None
        self.device_id: str | None = None
        self.index: int | None = None
        self.frame = None
        self.running = False
        self.backend = None
        self.name = None
        self.error = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._generation = 0
        self._known_devices: dict[str, dict] = {}

    @staticmethod
    def _read_valid_frame(capture: cv2.VideoCapture, attempts: int = 8):
        for _ in range(attempts):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size and len(frame.shape) == 3:
                return frame
            time.sleep(0.05)
        return None

    @classmethod
    def _probe(cls, index: int, backend_name: str) -> tuple[cv2.VideoCapture, object]:
        backend = cls._BACKEND_BY_NAME[backend_name]
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"{backend_name} could not open camera {index}")
        frame = cls._read_valid_frame(capture)
        if frame is None:
            capture.release()
            raise RuntimeError(f"{backend_name} opened camera {index}, but returned no frames")
        return capture, frame

    @classmethod
    def _enumerated_devices(cls) -> list[dict]:
        try:
            from cv2_enumerate_cameras import enumerate_cameras, supported_backends
        except ImportError:
            return []
        devices: list[dict] = []
        preferred = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for backend in [item for item in preferred if item in supported_backends]:
            backend_name = next(name for name, value in cls._BACKENDS if value == backend)
            for info in enumerate_cameras(backend):
                stable_key = (
                    re.sub(r"#\{[0-9a-f-]+\}", "", info.path, flags=re.IGNORECASE)
                    if info.path
                    else f"{info.name}:{info.index}"
                )
                devices.append({
                    "device_id": f"{backend_name.lower().replace(' ', '_')}:{info.index}",
                    "stable_id": stable_key,
                    "index": info.index,
                    "name": info.name or f"Camera device {info.index}",
                    "backend": backend_name,
                    "backend_preference": backend,
                    "path": info.path,
                })
        return devices

    def discover(self, maximum: int = 20) -> list[dict]:
        enumerated = self._enumerated_devices()
        candidates: list[dict] = []
        seen: set[str] = set()
        if enumerated:
            for device in enumerated:
                if device["stable_id"] in seen:
                    continue
                try:
                    capture, frame = self._probe(device["index"], device["backend"])
                except RuntimeError:
                    continue
                seen.add(device["stable_id"])
                height, width = frame.shape[:2]
                capture.release()
                result = {
                    "id": device["device_id"],
                    "stable_id": device["stable_id"],
                    "index": device["index"],
                    "name": device["name"],
                    "backend": device["backend"],
                    "available": True,
                    "frame_size": {"width": width, "height": height},
                }
                self._known_devices[device["device_id"]] = result
                candidates.append(result)
            if candidates:
                return candidates

        for index in range(maximum):
            for backend_name, _ in self._BACKENDS:
                try:
                    capture, frame = self._probe(index, backend_name)
                except RuntimeError:
                    continue
                height, width = frame.shape[:2]
                capture.release()
                result = {
                    "id": f"{backend_name.lower().replace(' ', '_')}:{index}",
                    "stable_id": f"opencv:{index}",
                    "index": index,
                    "name": f"Camera device {index}",
                    "backend": backend_name,
                    "available": True,
                    "frame_size": {"width": width, "height": height},
                }
                self._known_devices[result["id"]] = result
                candidates.append(result)
                break
        return candidates

    def connect(
        self, index: int | None = None, backend: str | None = None, device_id: str | None = None
    ) -> dict:
        self.disconnect()
        if device_id and ":" in device_id:
            backend_key, raw_index = device_id.rsplit(":", 1)
            index = int(raw_index)
            backend = backend_key.replace("_", " ").title()
            if backend == "Media Foundation":
                backend = "Media Foundation"
            elif backend == "Directshow":
                backend = "DirectShow"
        if index is None:
            raise ValueError("camera device or index is required")
        backend = backend if backend in self._BACKEND_BY_NAME else "DirectShow"
        try:
            capture, first_frame = self._probe(index, backend)
        except RuntimeError as exc:
            with self._lock:
                self.error = str(exc)
            raise
        with self._lock:
            self._generation += 1
            generation = self._generation
            self.capture = capture
            self.device_id = device_id or f"{backend.lower().replace(' ', '_')}:{index}"
            self.index = index
            self.backend = backend
            self.name = self._known_devices.get(self.device_id, {}).get(
                "name", f"Camera device {index}"
            )
            self.frame = first_frame
            self.error = None
            self._stop_event = threading.Event()
            self.running = True
            self._thread = threading.Thread(
                target=self._read_loop,
                args=(generation, self._stop_event),
                name="smartpark-camera",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def _read_loop(self, generation: int, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self._lock:
                capture = self.capture if generation == self._generation else None
            if capture is None:
                return
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                with self._lock:
                    if generation == self._generation and not stop_event.is_set():
                        self.frame = frame
            else:
                time.sleep(0.1)

    def disconnect(self) -> None:
        with self._lock:
            self._generation += 1
            stop_event = self._stop_event
            thread = self._thread
            capture = self.capture
            self.running = False
            stop_event.set()
            self.capture = None
            self.device_id = None
            self.index = None
            self.backend = None
            self.name = None
            self.frame = None
            self._thread = None
        if capture is not None:
            capture.release()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def status(self) -> dict:
        with self._lock:
            return {
                "connected": self.capture is not None and self.running,
                "device_id": self.device_id,
                "index": self.index,
                "name": self.name,
                "backend": self.backend,
                "frame_available": self.frame is not None,
                "error": self.error,
            }

    def latest_frame(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def capture_frame(self):
        frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("no valid camera frame is available")
        return frame
