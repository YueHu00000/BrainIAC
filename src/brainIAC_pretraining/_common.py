"""Small CSV helpers shared by the series workflow."""

import csv
from pathlib import Path


class MultiFrameDicomError(ValueError):
    """Unsupported multi-frame DICOM: stop the batch instead of skipping."""


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_ids(path):
    return [row["unique_id"] for row in read_rows(path)]


def write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_id(unique_id):
    if (not unique_id or unique_id.startswith(".")
            or unique_id.rstrip(" .") != unique_id
            or any(character in unique_id for character in '/\\:*?"<>|')):
        raise ValueError(f"Not a plain series filename: {unique_id!r}")


def scan_outputs(output_dir, csv_path):
    ids = sorted(path.name[:-7] for path in Path(output_dir).glob("*.nii.gz")
                 if path.is_file() and not path.name.startswith("."))
    write_rows(csv_path, ({"unique_id": unique_id} for unique_id in ids), ["unique_id"])
    return ids
