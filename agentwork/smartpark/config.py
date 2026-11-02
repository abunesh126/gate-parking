from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "backend" / "data" / "parking.db"
REFERENCE_DATASET = BASE_DIR.parent / "License-Plate-Number-Detection" / "data.zip"
MODEL_DIR = BASE_DIR / "ml" / "models"
CNN_CHECKPOINT = MODEL_DIR / "my_checkpoint"
RECOGNITION_COOLDOWN_SECONDS = 5.0
PARKING_UPDATE_INTERVAL_SECONDS = 5
DEFAULT_RATE_PER_HOUR = 40.0
DEFAULT_CURRENCY = "INR"
