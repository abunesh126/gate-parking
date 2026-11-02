from datetime import datetime, timezone

import pytest

from smartpark.backend.database import ParkingDatabase


def test_entry_duplicate_is_rejected(tmp_path):
    database = ParkingDatabase(tmp_path / "parking.db")
    database.create_entry("KA01AB1234")
    with pytest.raises(Exception):
        database.create_entry("KA01AB1234")


def test_exit_and_fee(tmp_path):
    database = ParkingDatabase(tmp_path / "parking.db")
    database.create_entry("KA01AB1234", "2026-10-30T10:00:00+00:00")
    session = database.complete_exit("KA01AB1234", "2026-10-30T11:10:00+00:00", 40)
    assert session["duration_minutes"] == 70
    assert session["parking_fee"] == 80


def test_settings_capacity_and_history_filters(tmp_path):
    database = ParkingDatabase(tmp_path / "parking.db")
    database.update_settings({
        "parking_capacity": "1",
        "recognition_confidence_threshold": "0.75",
        "detection_interval_seconds": "1",
        "duplicate_cooldown_seconds": "3",
    })
    database.create_entry("KA01AB1234", "2026-10-31T10:00:00+00:00")
    with pytest.raises(ValueError, match="capacity"):
        database.create_entry("KA02CD5678", "2026-10-31T10:01:00+00:00")
    assert len(database.list_sessions(status="active", date="2026-10-31")) == 1
    settings = database.settings()
    assert settings["parking_capacity"] == "1"
    assert settings["recognition_confidence_threshold"] == "0.75"
