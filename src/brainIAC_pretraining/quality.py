"""Select consistent series statistics; record only frame count and coverage."""

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import read_rows, validate_id, write_rows


FIELDS = ["unique_id", "frame_count", "coverage_mm"]


def count(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number > 0 and number.is_integer() else None


def consistent_count(row):
    counts = [count(row.get(field)) for field in ("file_count", "frame_count", "unique_slice_count")]
    if counts[0] is not None and counts[1] is not None and counts[0] != counts[1]:
        raise ValueError(
            f"{row['unique_id']}: unsupported multi-frame or inconsistent DICOM count "
            f"(file_count={counts[0]}, frame_count={counts[1]})"
        )
    if None in counts:
        return None, "unknown_or_invalid_count"
    if len(set(counts)) != 1:
        return None, "count_mismatch"
    return counts[1], ""


def source_matches(manifest, selected):
    # Compare selection metadata, without opening or validating source files.
    return (
        manifest["acquisition_number"] == selected["acquisition_number"]
        and Path(manifest["study_directory"]).resolve() == Path(selected["study_directory"]).resolve()
        and json.loads(manifest["file_names"]) == json.loads(selected["file_names"])
    )


def build_quality(manifest, selected):
    tables = []
    for name, rows in (("manifest", manifest), ("selected statistics", selected)):
        table = {}
        for row in rows:
            unique_id = row["unique_id"]
            validate_id(unique_id)
            if unique_id in table:
                raise ValueError(f"Duplicate unique_id in {name}: {unique_id}")
            table[unique_id] = row
        tables.append(table)
    manifest_table, selected_table = tables
    accepted, rejected = [], []
    for unique_id, manifest_row in manifest_table.items():
        row = selected_table.get(unique_id)
        reason = "missing_statistics" if row is None else ""
        if row is not None and not source_matches(manifest_row, row):
            reason = "selected_source_mismatch"
        frames = None
        if row is not None and not reason:
            frames, reason = consistent_count(row)
        if reason:
            rejected.append(dict(unique_id=unique_id, reason=reason))
            print(f"Skipped {unique_id}: {reason}", flush=True)
            continue
        try:
            coverage = float(row["coverage_mm"])
        except (KeyError, ValueError, TypeError):
            coverage = math.nan
        accepted.append(dict(unique_id=unique_id, frame_count=frames,
                             coverage_mm=coverage if math.isfinite(coverage) and coverage >= 0 else ""))
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, rejected = build_quality(read_rows(args.manifest), read_rows(args.selected_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "image_number_coverage.csv", rows, FIELDS)
    write_rows(args.output_dir / "rejected_quality.csv", rejected, ["unique_id", "reason"])
    print(f"Quality: {len(rows)} accepted; {len(rejected)} skipped")


if __name__ == "__main__":
    main()
