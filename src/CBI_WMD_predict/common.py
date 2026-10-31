"""CSV I/O, the published site assignments, and fixed embedding loading."""

import csv
import json
from pathlib import Path

import numpy as np


MODALITIES = ("T1", "T2", "T1T2")
BLOCK_NUMBERS = (3, 6, 9, 12)

# MODELS.md at YueHu00000/neuroimage-classifiers commit 8bdb9fa.
# These are five site holdouts, not an exhaustive out-of-fold partition.
FOLDS = {
    1: {"train": (1, 2, 3, 6, 7, 9, 10, 11), "val": (12,), "test": (13, 14)},
    2: {"train": (1, 2, 3, 6, 7, 9, 13, 14), "val": (10, 11), "test": (12,)},
    3: {"train": (1, 2, 3, 6, 7, 12, 13, 14), "val": (9,), "test": (10, 11)},
    4: {"train": (1, 2, 3, 6, 10, 11, 12, 13, 14), "val": (7,), "test": (9,)},
    5: {"train": (1, 2, 3, 9, 10, 11, 12, 13, 14), "val": (6,), "test": (7,)},
}


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
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_features(rows, modality, embedding_field="final_embed", block=None):
    """Read only the selected full-ID records; never substitute another series."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    if (embedding_field not in ("final_embed", "block_embed")
            or (embedding_field == "final_embed" and block is not None)
            or (embedding_field == "block_embed" and block not in BLOCK_NUMBERS)):
        raise ValueError("Use final_embed without a block, or block_embed with block 3, 6, 9 or 12")
    prefixes = ("t1", "t2") if modality == "T1T2" else (modality.lower(),)
    features = []
    for row in rows:
        vectors = []
        with np.load(row["embedding_path"], allow_pickle=False) as archive:
            for prefix in prefixes:
                unique_id = row[f"{prefix}_unique_id"]
                if unique_id not in archive:
                    raise ValueError(f"{row['embedding_path']}: missing selected key {unique_id}")
                record = archive[unique_id]
                if record.shape != () or not record.dtype.names or embedding_field not in record.dtype.names:
                    raise ValueError(f"{unique_id}: expected new structured scalar with {embedding_field}")
                vector = np.asarray(record[embedding_field], dtype=np.float32)
                if embedding_field == "block_embed":
                    if vector.shape != (4, 768):
                        raise ValueError(f"{unique_id}: expected block_embed with shape (4,768)")
                    vector = vector[BLOCK_NUMBERS.index(block)]
                if vector.shape != (768,) or not np.isfinite(vector).all():
                    raise ValueError(f"{unique_id}: expected finite selected embedding with shape (768,)")
                vectors.append(vector)
        features.append(np.concatenate(vectors))
    width = 1536 if modality == "T1T2" else 768
    return np.stack(features) if features else np.empty((0, width), dtype=np.float32)
