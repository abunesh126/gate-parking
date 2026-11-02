import numpy as np
import pytest

from smartpark.app import create_app
from smartpark.ml import predict
from smartpark.ml.plate_detection import PlateDetection
from smartpark.ml.predict import PlateRecognizer


def test_segmented_characters_reach_cnn(monkeypatch):
    seen = {}

    class FakeModel:
        def predict(self, batch, verbose=0):
            seen["shape"] = batch.shape
            scores = np.zeros((batch.shape[0], 36), dtype=np.float32)
            scores[:, 10] = 0.95
            return scores

    crop = np.zeros((48, 160, 3), dtype=np.uint8)
    characters = [np.full((28, 28), 255, dtype=np.uint8) for _ in range(4)]
    monkeypatch.setattr(
        predict,
        "analyze_plate",
        lambda image: (PlateDetection(crop, (0, 0, 160, 48), 1.0), {}),
    )
    monkeypatch.setattr(
        predict,
        "analyze_character_segmentation",
        lambda plate, debug=False: (characters, {}),
    )
    recognizer = PlateRecognizer()
    recognizer.model = FakeModel()

    result = recognizer.recognize(np.zeros((80, 200, 3), dtype=np.uint8))

    assert result["status"] == "recognized"
    assert result["plate_number"] == "AAAA"
    assert result["confidence"] == pytest.approx(0.95)
    assert seen["shape"] == (4, 28, 28, 3)


def test_short_high_confidence_prediction_is_not_recognized(monkeypatch):
    class FakeModel:
        def predict(self, batch, verbose=0):
            scores = np.zeros((batch.shape[0], 36), dtype=np.float32)
            scores[:, 10] = 0.99
            return scores

    crop = np.zeros((48, 160, 3), dtype=np.uint8)
    characters = [np.full((28, 28), 255, dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(
        predict,
        "analyze_plate",
        lambda image: (PlateDetection(crop, (0, 0, 160, 48), 1.0), {}),
    )
    monkeypatch.setattr(
        predict,
        "analyze_character_segmentation",
        lambda plate, debug=False: (characters, {}),
    )
    recognizer = PlateRecognizer()
    recognizer.model = FakeModel()

    result = recognizer.recognize(np.zeros((80, 200, 3), dtype=np.uint8))

    assert result["status"] == "uncertain"
    assert result["success"] is False


def test_confident_entry_is_automatic_and_duplicate_safe(tmp_path):
    app = create_app(tmp_path / "parking.db")
    worker = app.extensions["recognition_worker"]
    client = app.test_client()
    try:
        worker.on_result({
            "status": "recognized",
            "plate_number": "KA01AB1234",
            "confidence": 0.95,
        })
        worker.on_result({
            "status": "recognized",
            "plate_number": "KA01AB1234",
            "confidence": 0.95,
        })
        sessions = app.extensions["database"].list_sessions("active")
        assert len(sessions) == 1
        assert client.get("/api/camera/recognition").get_json()["mode"] == "entry"
    finally:
        worker.stop()
        app.extensions["transactions"].stop()
        app.extensions["camera"].disconnect()


def test_confident_exit_completes_session_with_fee(tmp_path):
    app = create_app(tmp_path / "parking.db")
    worker = app.extensions["recognition_worker"]
    client = app.test_client()
    try:
        worker.on_result({"status": "recognized", "plate_number": "KA01AB1234"})
        assert client.post("/api/parking/mode", json={"mode": "exit"}).status_code == 200
        worker.on_result({"status": "recognized", "plate_number": "KA01AB1234"})
        sessions = app.extensions["database"].list_sessions("completed")
        assert len(sessions) == 1
        assert sessions[0]["parking_fee"] is not None
        assert sessions[0]["exit_time"] is not None
    finally:
        worker.stop()
        app.extensions["transactions"].stop()
        app.extensions["camera"].disconnect()


def test_uncertain_result_does_not_create_transaction(tmp_path):
    app = create_app(tmp_path / "parking.db")
    worker = app.extensions["recognition_worker"]
    try:
        worker.on_result({
            "status": "uncertain",
            "plate_number": "KA01AB1234",
            "confidence": 0.4,
        })
        assert app.extensions["database"].list_sessions() == []
    finally:
        worker.stop()
        app.extensions["transactions"].stop()
        app.extensions["camera"].disconnect()


def test_mode_change_rejects_stale_recognition(tmp_path):
    app = create_app(tmp_path / "parking.db")
    worker = app.extensions["recognition_worker"]
    client = app.test_client()
    try:
        assert client.post("/api/parking/mode", json={"mode": "exit"}).status_code == 200
        worker.on_result({
            "status": "recognized",
            "plate_number": "KA01AB1234",
            "_mode": "entry",
            "_mode_generation": 0,
        })
        assert app.extensions["database"].list_sessions() == []
        latest = client.get("/api/camera/recognition").get_json()["latest_detection"]
        assert latest["processing_status"] == "rejected"
    finally:
        worker.stop()
        app.extensions["transactions"].stop()
        app.extensions["camera"].disconnect()
