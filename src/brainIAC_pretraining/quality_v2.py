"""Generate quality CSV with series_description directly after unique_id."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import read_rows, write_rows
from brainIAC_pretraining._dicom import measure_series
from brainIAC_pretraining.quality import build_quality as original_build_quality


FIELDS = ["unique_id", "series_description", "frame_count", "coverage_mm"]


def build_quality(manifest, selected=None):
    if selected is None:
        selected = [measure_series(row) for row in manifest]
    accepted, rejected = original_build_quality(manifest, selected)
    descriptions = {row["unique_id"]: row.get("series_description", "") for row in selected}
    rows = [dict(unique_id=row["unique_id"], series_description=descriptions[row["unique_id"]],
                 frame_count=row["frame_count"], coverage_mm=row["coverage_mm"]) for row in accepted]
    return rows, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selected-csv", type=Path,
                        help="selected_series.csv from summary_dicom_SeriesInstanceUID.py; "
                             "omitting this option reads manifest-listed DICOM headers")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    selected = read_rows(args.selected_csv) if args.selected_csv is not None else None
    rows, rejected = build_quality(read_rows(args.manifest), selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "image_number_coverage.csv", rows, FIELDS)
    write_rows(args.output_dir / "rejected_quality.csv", rejected, ["unique_id", "reason"])
    print(f"Quality v2: {len(rows)} accepted; {len(rejected)} skipped")


if __name__ == "__main__":
    main()
