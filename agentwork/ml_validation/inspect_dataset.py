"""Inspect the reference character dataset without extracting or changing it."""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from validation_common import CLASS_ORDER, DATA_ZIP, ORIGINAL, WORKSPACE


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def image_info(data: bytes) -> tuple[str, str, str]:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        return "unreadable", "", ""
    channels = "1" if image.ndim == 2 else str(image.shape[2])
    return "ok", f"{image.shape[1]}x{image.shape[0]}", channels


def class_index(directory_name: str) -> int:
    value = directory_name.removeprefix("class_")
    if value.isdigit():
        return int(value)
    return 10 + ord(value) - ord("A")


def inspect_directory(path: Path) -> dict[str, object]:
    counts: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            index = class_index(file_path.parent.name)
        except (TypeError, ValueError):
            errors.append({"file": str(file_path), "error": "unrecognised class directory"})
            continue
        counts[index] += 1
        formats[file_path.suffix.lower()] += 1
        status, dimension, channel = image_info(file_path.read_bytes())
        if status != "ok":
            errors.append({"file": str(file_path), "error": status})
        else:
            dimensions[dimension] += 1
            channels[channel] += 1
    return {
        "path": str(path),
        "total_images": sum(counts.values()),
        "class_counts": dict(zip(CLASS_ORDER, [counts[i] for i in range(36)])),
        "missing_or_empty_classes": [
            CLASS_ORDER[i] for i in range(36) if counts[i] == 0
        ],
        "dimensions": dict(dimensions),
        "channels": dict(channels),
        "formats": dict(formats),
        "unreadable_images": errors,
    }


def main() -> int:
    output: dict[str, object] = {
        "data_zip": str(DATA_ZIP),
        "zip_exists": DATA_ZIP.is_file(),
        "archive_root": None,
        "directory_presence": {},
        "checkpoint_val_directories": [],
        "splits": {},
    }
    for candidate in (ORIGINAL, WORKSPACE):
        for relative in ("data", "data/train", "data/val", "checkpoints/val"):
            path = candidate / relative
            output["directory_presence"][str(path)] = path.is_dir()
        for split in ("train", "val"):
            path = candidate / "data" / split
            if path.is_dir():
                output.setdefault("extracted_split_inspections", []).append(
                    inspect_directory(path)
                )
        for path in candidate.rglob("val"):
            if path.is_dir() and "checkpoint" in str(path.parent).lower():
                output["checkpoint_val_directories"].append(str(path))
                output.setdefault("checkpoint_val_inspections", []).append(
                    inspect_directory(path)
                )

    if not DATA_ZIP.is_file():
        print(json.dumps(output, indent=2))
        return 1

    split_files: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(DATA_ZIP) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        output["archive_root"] = sorted({name.split("/")[0] for name in names})
        for name in names:
            parts = name.split("/")
            if len(parts) >= 4 and parts[0] == "data" and parts[1] in {"train", "val"}:
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    split_files[parts[1]].append(name)

        for split in ("train", "val"):
            counts: Counter[str] = Counter()
            dimensions: Counter[str] = Counter()
            channels: Counter[str] = Counter()
            formats: Counter[str] = Counter()
            errors: list[dict[str, str]] = []
            for name in sorted(split_files[split]):
                class_name = name.split("/")[2]
                counts[class_name] += 1
                formats[Path(name).suffix.lower()] += 1
                status, dimension, channel = image_info(archive.read(name))
                if status != "ok":
                    errors.append({"file": name, "error": status})
                else:
                    dimensions[dimension] += 1
                    channels[channel] += 1
            class_counts = {
                index: counts[f"class_{index}"]
                if index < 10
                else counts[f"class_{CLASS_ORDER[index]}"]
                for index in range(36)
            }
            output["splits"][split] = {
                "path": f"{DATA_ZIP}!data/{split}",
                "total_images": sum(class_counts.values()),
                "class_counts": dict(zip(CLASS_ORDER, class_counts.values())),
                "class_directory_counts": class_counts,
                "missing_or_empty_classes": [
                    CLASS_ORDER[index]
                    for index, count in class_counts.items()
                    if count == 0
                ],
                "dimensions": dict(dimensions),
                "channels": dict(channels),
                "formats": dict(formats),
                "unreadable_images": errors,
            }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
