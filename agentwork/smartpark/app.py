"""Flask entry point for SmartPark."""

import atexit
import time
import sqlite3
import threading
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

from .backend.camera import CameraManager
from .backend.database import ParkingDatabase
from .backend.queue import TransactionQueue
from .backend.recognition_worker import RecognitionWorker
from .config import (
    CNN_CHECKPOINT,
    DATABASE_PATH,
    DEFAULT_RATE_PER_HOUR,
    PARKING_UPDATE_INTERVAL_SECONDS,
    RECOGNITION_COOLDOWN_SECONDS,
)
from .ml.predict import PlateRecognizer


def create_app(database_path: Path = DATABASE_PATH) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    db = ParkingDatabase(database_path)
    recognizer = PlateRecognizer(CNN_CHECKPOINT if CNN_CHECKPOINT.with_suffix(".index").exists() else None)
    state = {
        "mode": "entry",
        "mode_generation": 0,
        "latest_detection": None,
        "captured_frame": None,
        "last_plate": None,
        "last_recognition_at": 0.0,
        "auto_processing": True,
        "auto_last_processed": {},
        "recognition_error": None,
    }
    state_lock = threading.RLock()

    def process_plate(plate: str) -> None:
        if state["mode"] == "entry":
            if db.active_session(plate) is None:
                db.create_entry(plate)
        else:
            if db.active_session(plate) is not None:
                db.complete_exit(plate, rate_per_hour=float(db.setting("rate_per_hour", str(DEFAULT_RATE_PER_HOUR))))

    transactions = TransactionQueue(process_plate, PARKING_UPDATE_INTERVAL_SECONDS)
    camera = CameraManager()

    def apply_recognition(detection: dict) -> None:
        now = time.monotonic()
        with state_lock:
            state["latest_detection"] = detection
            state["recognition_error"] = (
                detection.get("message") if detection.get("status") == "recognition_error" else None
            )
        if detection.get("status") != "recognized" or not state["auto_processing"]:
            return
        plate = detection.get("plate_number", "")
        try:
            cooldown = float(db.setting("duplicate_cooldown_seconds", "10"))
        except ValueError:
            cooldown = 10.0
        with state_lock:
            mode = detection.get("_mode", state["mode"])
            generation = detection.get("_mode_generation")
            if generation is not None and generation != state["mode_generation"]:
                state["latest_detection"] = {
                    **detection,
                    "processing_status": "rejected",
                    "processing_message": "Recognition became stale after the parking mode changed",
                }
                return
            key = f"{mode}:{plate}"
            if now - state["auto_last_processed"].get(key, 0) < cooldown:
                return
            state["auto_last_processed"][key] = now
        try:
            if mode == "entry":
                session = db.create_entry(plate)
                message = f"Automatic entry recorded for {session['plate_number']}"
            else:
                session = db.complete_exit(plate)
                message = f"Automatic exit recorded for {session['plate_number']}"
            with state_lock:
                state["latest_detection"] = {
                    **detection, "processing_status": "processed", "processing_message": message,
                }
        except (ValueError, sqlite3.IntegrityError) as exc:
            with state_lock:
                state["latest_detection"] = {
                    **detection, "processing_status": "rejected", "processing_message": str(exc),
                }

    worker = RecognitionWorker(
        camera, recognizer, db.settings, apply_recognition,
        get_context=lambda: (state["mode"], state["mode_generation"]),
    )

    def shutdown():
        worker.stop()
        transactions.stop()
        camera.disconnect()

    atexit.register(shutdown)
    app.extensions.update(
        database=db, camera=camera, recognizer=recognizer,
        transactions=transactions, recognition_worker=worker,
        state=state,
    )

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", page_title="Dashboard")

    @app.get("/<page>")
    def pages(page: str):
        pages = {"camera": "Live Camera", "active": "Active Vehicles", "history": "Parking History",
                 "analytics": "Analytics & Reports", "settings": "Settings"}
        if page not in pages:
            return render_template("dashboard.html", page_title="Dashboard")
        return render_template(f"{page}.html", page_title=pages[page])

    @app.get("/api/status")
    def status():
        return jsonify({"backend": "online", "cnn_model": recognizer.available, "camera": camera.status(), "database": True})

    @app.get("/api/dashboard")
    def dashboard_data():
        return jsonify(db.dashboard())

    @app.get("/api/cameras")
    def cameras():
        try:
            return jsonify(camera.discover())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/cameras/connect")
    def connect_camera():
        payload = request.get_json(silent=True) or {}
        try:
            device_id = payload.get("device_id")
            index = int(payload["index"]) if payload.get("index") is not None else None
            if index is not None and index < 0:
                raise ValueError("camera index must be non-negative")
            return jsonify(camera.connect(index, payload.get("backend"), device_id))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/cameras/disconnect")
    def disconnect_camera():
        camera.disconnect()
        state["captured_frame"] = None
        return jsonify(camera.status())

    @app.get("/api/camera/status")
    def camera_status():
        return jsonify(camera.status())

    @app.get("/api/cameras/status")
    def cameras_status():
        return jsonify(camera.status())

    @app.get("/api/camera/feed")
    def camera_feed():
        def frames():
            while True:
                frame = camera.latest_frame()
                if frame is None:
                    import time
                    time.sleep(0.1)
                    continue
                with state_lock:
                    detection = state["latest_detection"]
                if detection and detection.get("bounding_box"):
                    x, y, width, height = detection["bounding_box"]
                    color = (0, 200, 0) if detection.get("status") == "recognized" else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/camera/capture")
    def capture_frame():
        try:
            state["captured_frame"] = camera.capture_frame()
            return jsonify(
                {"captured": True, "camera": camera.status(), "message": "Frame captured"}
            )
        except RuntimeError as exc:
            return jsonify({"captured": False, "error": str(exc)}), 409

    @app.post("/api/camera/recognize")
    def recognize_captured_frame():
        frame = state["captured_frame"]
        if frame is None:
            return jsonify({"success": False, "status": "no_capture", "message": "Capture a camera frame first"}), 409
        debug = request.args.get("debug", "").lower() in {"1", "true", "yes", "on"}
        try:
            threshold = float(db.setting("recognition_confidence_threshold", "0.60"))
        except ValueError:
            threshold = 0.60
        detection = recognizer.recognize(
            frame, debug=debug, confidence_threshold=threshold
        )
        now = time.monotonic()
        if (
            detection["status"] == "recognized"
            and detection["plate_number"] == state["last_plate"]
            and now - state["last_recognition_at"] < RECOGNITION_COOLDOWN_SECONDS
        ):
            detection["status"] = "duplicate_suppressed"
            detection["success"] = False
            detection["message"] = "The same plate was recognized recently"
        elif detection["status"] == "recognized":
            state["last_plate"] = detection["plate_number"]
            state["last_recognition_at"] = now
        state["latest_detection"] = detection
        return jsonify(detection)

    @app.get("/api/camera/recognition")
    def recognition_status():
        with state_lock:
            return jsonify({
                "automatic": state["auto_processing"],
                "latest_detection": state["latest_detection"],
                "error": state["recognition_error"],
                "mode": state["mode"],
            })

    @app.get("/api/vehicles/active")
    def active_vehicles():
        return jsonify(db.list_sessions("active", request.args.get("search", "")))

    @app.get("/api/parking/history")
    def history():
        return jsonify(db.list_sessions(
            status=request.args.get("status") or None,
            search=request.args.get("search", ""),
            date=request.args.get("date") or None,
        ))

    @app.get("/api/parking/mode")
    def get_mode():
        return jsonify({"mode": state["mode"]})

    @app.post("/api/parking/mode")
    def set_mode():
        payload = request.get_json(silent=True) or {}
        if payload.get("mode") not in {"entry", "exit"}:
            return jsonify({"error": "mode must be entry or exit"}), 400
        with state_lock:
            if state["mode"] != payload["mode"]:
                state["mode"] = payload["mode"]
                state["mode_generation"] += 1
        return jsonify({"mode": state["mode"]})

    @app.post("/api/parking/transactions")
    def queue_transaction():
        payload = request.get_json(silent=True) or {}
        plate = payload.get("plate", "")
        if not isinstance(plate, str) or not plate.strip():
            return jsonify({"error": "plate is required"}), 400
        if not transactions.add(plate):
            return jsonify({"queued": False, "reason": "duplicate or empty plate"}), 409
        return jsonify({"queued": True, "plate": plate.strip().upper()}), 202

    @app.get("/api/parking/latest-detection")
    def latest_detection():
        return jsonify(state["latest_detection"])

    def plate_from_payload(payload: dict) -> tuple[str, bool]:
        plate = payload.get("plate", "")
        manual = bool(payload.get("manual", False))
        if not isinstance(plate, str) or not plate.strip():
            raise ValueError("plate is required")
        normalized = "".join(plate.upper().split())
        if not 2 <= len(normalized) <= 12 or not normalized.isalnum():
            raise ValueError("plate must contain 2-12 letters or digits")
        return normalized, manual

    @app.post("/api/parking/entry")
    def parking_entry():
        try:
            plate, manual = plate_from_payload(request.get_json(silent=True) or {})
            session = db.create_entry(plate)
            session["entry_source"] = "manual" if manual else "recognition"
            return jsonify(session), 201
        except (TypeError, ValueError, sqlite3.IntegrityError) as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/parking/exit")
    def parking_exit():
        try:
            plate, manual = plate_from_payload(request.get_json(silent=True) or {})
            session = db.complete_exit(plate)
            session["exit_source"] = "manual" if manual else "recognition"
            return jsonify(session)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/vehicles")
    def vehicles():
        return jsonify(db.list_vehicles(request.args.get("search", "")))

    @app.get("/api/vehicles/<int:vehicle_id>")
    def vehicle(vehicle_id: int):
        rows = [row for row in db.list_vehicles() if row["id"] == vehicle_id]
        return jsonify(rows[0]) if rows else (jsonify({"error": "vehicle not found"}), 404)

    @app.get("/api/parking/sessions")
    def parking_sessions():
        status_filter = request.args.get("status")
        if status_filter == "active":
            return jsonify(db.list_sessions("active", request.args.get("search", "")))
        return jsonify(db.list_sessions(search=request.args.get("search", "")))

    @app.get("/api/parking/sessions/<int:session_id>")
    def parking_session(session_id: int):
        session = db.get_session(session_id)
        return jsonify(session) if session else (jsonify({"error": "session not found"}), 404)

    @app.get("/api/analytics/overview")
    def analytics():
        return jsonify(db.dashboard())

    @app.get("/api/analytics")
    def analytics_full():
        dashboard = db.dashboard()
        return jsonify({
            **dashboard,
            "completed_sessions": db.list_sessions("completed"),
        })

    @app.get("/api/analytics/vehicles")
    def analytics_vehicles():
        return jsonify({"active_vehicles": db.dashboard()["active_vehicles"]})

    @app.get("/api/settings")
    def settings():
        return jsonify(db.settings())

    @app.put("/api/settings")
    def update_settings():
        try:
            return jsonify(db.update_settings(request.get_json(silent=True) or {}))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
