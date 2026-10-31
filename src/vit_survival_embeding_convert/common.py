"""Small I/O and output-schema helpers for the legacy adapter."""

import csv
import json
import math
from pathlib import Path


NORM_A_ID = "tufts_train_global_v1"
BLOCK_NUMBERS = (3, 6, 9, 12)
MODALITIES = ("T1", "T2", "T1T2")
SITES = (1, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14)
SERIES_FIELDS = ("unique_id", "series_instance_uid", "series_number", "acquisition_number",
                 "series_description", "frame_count", "coverage_mm")
MANIFEST_FIELDS = ["patient_id", "study_id", "site_id", "CBI", "WMD", "embedding_path", "modality"] + [
    f"{prefix}_{field}" for prefix in ("t1", "t2") for field in SERIES_FIELDS
]


def read_csv(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return [{key: value.strip() for key, value in row.items()} for row in rows]


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=str) + "\n", encoding="utf-8")


def plain_id(value):
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"Not a plain file identifier: {value!r}")
    return value


def integer(value, name):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer: {value!r}")
    return int(number)
