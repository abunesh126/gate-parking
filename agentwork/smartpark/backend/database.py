"""SQLite persistence and parking business rules."""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ParkingDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def initialize(self) -> None:
        with self._lock, self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY,
                    plate_number TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parking_sessions (
                    id INTEGER PRIMARY KEY,
                    vehicle_id INTEGER NOT NULL,
                    plate_number TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    duration_minutes INTEGER,
                    parking_fee REAL,
                    session_status TEXT NOT NULL DEFAULT 'active',
                    FOREIGN KEY(vehicle_id) REFERENCES vehicles(id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_session
                    ON parking_sessions(plate_number)
                    WHERE session_status = 'active';
                CREATE TABLE IF NOT EXISTS system_settings (
                    id INTEGER PRIMARY KEY,
                    setting_key TEXT NOT NULL UNIQUE,
                    setting_value TEXT NOT NULL
                );
                """
            )
            self._remove_payment_schema(db)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(vehicles)")}
            if "last_seen" not in columns:
                db.execute("ALTER TABLE vehicles ADD COLUMN last_seen TEXT NOT NULL DEFAULT ''")
                db.execute("UPDATE vehicles SET last_seen = created_at WHERE last_seen = ''")
            db.executemany(
                "INSERT OR IGNORE INTO system_settings(setting_key, setting_value) VALUES (?, ?)",
                [
                    ("parking_capacity", "50"),
                    ("initial_fee", "20"),
                    ("hourly_fee", "10"),
                    ("grace_period_minutes", "10"),
                    ("currency", "INR"),
                    ("rate_per_hour", "40"),
                    ("recognition_confidence_threshold", "0.60"),
                    ("detection_interval_seconds", "2.0"),
                    ("duplicate_cooldown_seconds", "10.0"),
                ],
            )

    @staticmethod
    def _remove_payment_schema(db: sqlite3.Connection) -> None:
        db.execute("DROP TABLE IF EXISTS payments")
        columns = {
                row["name"] for row in db.execute("PRAGMA table_info(parking_sessions)")
        }
        if "payment_status" not in columns:
                return
        db.execute("DROP INDEX IF EXISTS one_active_session")
        db.execute(
                """
                CREATE TABLE parking_sessions_without_payments (
                    id INTEGER PRIMARY KEY,
                    vehicle_id INTEGER NOT NULL,
                    plate_number TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    duration_minutes INTEGER,
                    parking_fee REAL,
                    session_status TEXT NOT NULL DEFAULT 'active',
                    FOREIGN KEY(vehicle_id) REFERENCES vehicles(id)
                )
                """
        )
        db.execute(
                """
                INSERT INTO parking_sessions_without_payments
                    (id, vehicle_id, plate_number, entry_time, exit_time,
                     duration_minutes, parking_fee, session_status)
                SELECT id, vehicle_id, plate_number, entry_time, exit_time,
                       duration_minutes, parking_fee, session_status
                FROM parking_sessions
                """
        )
        db.execute("DROP TABLE parking_sessions")
        db.execute(
                "ALTER TABLE parking_sessions_without_payments RENAME TO parking_sessions"
        )
        db.execute(
                """
                CREATE UNIQUE INDEX one_active_session
                    ON parking_sessions(plate_number)
                    WHERE session_status = 'active'
                """
        )

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def create_entry(self, plate: str, entry_time: str | None = None) -> dict:
        normalized = plate.strip().upper()
        if not normalized:
            raise ValueError("plate number is required")
        timestamp = entry_time or utc_now()
        self._parse_timestamp(timestamp)
        with self._lock, self.connect() as db:
            capacity = int(float(self.setting("parking_capacity", "50")))
            active_count = db.execute(
                "SELECT COUNT(*) FROM parking_sessions WHERE session_status = 'active'"
            ).fetchone()[0]
            if active_count >= capacity:
                raise ValueError("parking capacity is full")
            active = db.execute(
                "SELECT id FROM parking_sessions WHERE plate_number = ? AND session_status = 'active'",
                (normalized,),
            ).fetchone()
            if active:
                raise ValueError("vehicle is already parked")
            vehicle = db.execute(
                "SELECT id FROM vehicles WHERE plate_number = ?", (normalized,)
            ).fetchone()
            if vehicle is None:
                vehicle_id = db.execute(
                    "INSERT INTO vehicles(plate_number, created_at, last_seen) VALUES (?, ?, ?)",
                    (normalized, timestamp, timestamp),
                ).lastrowid
            else:
                vehicle_id = vehicle["id"]
                db.execute("UPDATE vehicles SET last_seen = ? WHERE id = ?", (timestamp, vehicle_id))
            session_id = db.execute(
                "INSERT INTO parking_sessions(vehicle_id, plate_number, entry_time) VALUES (?, ?, ?)",
                (vehicle_id, normalized, timestamp),
            ).lastrowid
            row = db.execute("SELECT * FROM parking_sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row)

    def active_session(self, plate: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM parking_sessions WHERE plate_number = ? AND session_status = 'active'",
                (plate.strip().upper(),),
            ).fetchone()
            return dict(row) if row else None

    def complete_exit(self, plate: str, exit_time: str | None = None, rate_per_hour: float | None = None) -> dict:
        normalized = plate.strip().upper()
        with self._lock, self.connect() as db:
            row = db.execute(
                "SELECT * FROM parking_sessions WHERE plate_number = ? AND session_status = 'active'",
                (normalized,),
            ).fetchone()
            if row is None:
                raise ValueError("no active parking session for this plate")
            end = self._parse_timestamp(exit_time) if exit_time else datetime.now(timezone.utc)
            start = self._parse_timestamp(row["entry_time"])
            if end < start:
                raise ValueError("exit time cannot be before entry time")
            minutes = max(0, int((end - start).total_seconds() / 60))
            if rate_per_hour is not None:
                fee = round(max(1, (minutes + 59) // 60) * rate_per_hour, 2)
            else:
                initial = float(self.setting("initial_fee", "20"))
                hourly = float(self.setting("hourly_fee", "10"))
                grace = int(self.setting("grace_period_minutes", "10"))
                billable_hours = 0 if minutes <= grace else max(1, (minutes + 59) // 60)
                fee = round(initial + billable_hours * hourly, 2) if minutes > grace else 0.0
            db.execute(
                """UPDATE parking_sessions
                   SET exit_time = ?, duration_minutes = ?, parking_fee = ?,
                       session_status = 'completed'
                   WHERE id = ?""",
                (end.isoformat(), minutes, fee, row["id"]),
            )
            result = db.execute("SELECT * FROM parking_sessions WHERE id = ?", (row["id"],)).fetchone()
            return dict(result)

    def get_session(self, session_id: int) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT s.*, v.plate_number AS vehicle_registration
                   FROM parking_sessions s JOIN vehicles v ON v.id = s.vehicle_id
                   WHERE s.id = ?""",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(
        self, status: str | None = None, search: str = "",
        date: str | None = None,
    ) -> list[dict]:
        query = """SELECT s.*, v.plate_number AS vehicle_registration
                   FROM parking_sessions s JOIN vehicles v ON v.id = s.vehicle_id"""
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("s.session_status = ?")
            params.append(status)
        if search.strip():
            clauses.append("s.plate_number LIKE ?")
            params.append(f"%{search.strip().upper()}%")
        if date:
            clauses.append("(s.entry_time LIKE ? OR s.exit_time LIKE ?)")
            params.extend([f"{date}%", f"{date}%"])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY s.id DESC LIMIT 500"
        with self.connect() as db:
            return [dict(row) for row in db.execute(query, params)]

    def list_vehicles(self, search: str = "") -> list[dict]:
        query = "SELECT * FROM vehicles"
        params: list[object] = []
        if search.strip():
            query += " WHERE plate_number LIKE ?"
            params.append(f"%{search.strip().upper()}%")
        query += " ORDER BY id DESC LIMIT 500"
        with self.connect() as db:
            return [dict(row) for row in db.execute(query, params)]

    def dashboard(self) -> dict:
        with self.connect() as db:
            today = datetime.now(timezone.utc).date().isoformat()
            capacity = int(float(self.setting("parking_capacity", "50")))
            active = db.execute(
                "SELECT COUNT(*) FROM parking_sessions WHERE session_status = 'active'"
            ).fetchone()[0]
            return {
                "active_vehicles": active,
                "parking_capacity": capacity,
                "available_spaces": max(0, capacity - active),
                "parking_utilization_percent": round((active / capacity) * 100, 2) if capacity else 0,
                "total_sessions": db.execute("SELECT COUNT(*) FROM parking_sessions").fetchone()[0],
                "total_vehicles": db.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0],
                "todays_entries": db.execute("SELECT COUNT(*) FROM parking_sessions WHERE entry_time LIKE ?", (today + "%",)).fetchone()[0],
                "todays_exits": db.execute("SELECT COUNT(*) FROM parking_sessions WHERE exit_time LIKE ?", (today + "%",)).fetchone()[0],
            }

    def setting(self, key: str, default: str = "") -> str:
        with self.connect() as db:
            row = db.execute("SELECT setting_value FROM system_settings WHERE setting_key = ?", (key,)).fetchone()
            return row["setting_value"] if row else default

    def settings(self) -> dict[str, str]:
        with self.connect() as db:
            return {row["setting_key"]: row["setting_value"] for row in db.execute("SELECT setting_key, setting_value FROM system_settings")}

    def update_settings(self, values: dict[str, str]) -> dict[str, str]:
        allowed = {
            "parking_capacity", "initial_fee", "hourly_fee", "grace_period_minutes",
            "currency", "recognition_confidence_threshold",
            "detection_interval_seconds", "duplicate_cooldown_seconds",
        }
        if set(values) - allowed:
            raise ValueError("unsupported setting")
        try:
            clean = {}
            for key, value in values.items():
                if key == "currency":
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("currency is required")
                    clean[key] = value.strip().upper()
                elif key == "parking_capacity":
                    number = int(value)
                    if number < 1:
                        raise ValueError(f"{key} must be positive")
                    clean[key] = str(number)
                else:
                    number = float(value)
                    if number < 0:
                        raise ValueError(f"{key} cannot be negative")
                    if key == "recognition_confidence_threshold" and number > 1:
                        raise ValueError(f"{key} must be between 0 and 1")
                    if key in {"detection_interval_seconds", "duplicate_cooldown_seconds"} and number <= 0:
                        raise ValueError(f"{key} must be positive")
                    clean[key] = str(int(number) if key == "grace_period_minutes" else number)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid setting value") from exc
        with self._lock, self.connect() as db:
            for key, value in clean.items():
                db.execute(
                    """INSERT INTO system_settings(setting_key, setting_value) VALUES (?, ?)
                       ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value""",
                    (key, value),
                )
        return self.settings()
