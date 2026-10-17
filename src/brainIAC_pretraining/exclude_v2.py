"""Retain T1/T2 descriptions with 20-40 frames and sufficient coverage."""

import argparse
from functools import partial
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining.exclude import exclude_files as original_exclude_files
from brainIAC_pretraining.exclude import exclusion_reason as original_exclusion_reason


def exclusion_reason(row, frame_count_min, coverage_threshold_mm, *, frame_count_max=40):
    if row is None:
        return "missing_quality"
    if "series_description" not in row:
        raise ValueError("Use quality_v2.py output with a series_description column")
    original = original_exclusion_reason(row, frame_count_min, coverage_threshold_mm)
    reasons = original.split(";") if original else []
    description = str(row["series_description"] or "").lower()
    if "t1" not in description and "t2" not in description:
        reasons.append("no_t1_t2_in_description")
    try:
        frames = float(row["frame_count"])
    except (KeyError, TypeError, ValueError):
        frames = math.nan
    if math.isfinite(frames) and frames.is_integer() and frames > frame_count_max:
        reasons.append("high_frame_count")
    return ";".join(reasons)


def exclude_files(quality_csv, pre_process_csv, input_dir, excluded_dir, *,
                  frame_count_min=20, frame_count_max=40, coverage_threshold_mm,
                  final_csv=None, dry_run=False):
    if frame_count_min < 1 or frame_count_max < frame_count_min:
        raise ValueError("Require 1 <= frame_count_min <= frame_count_max")
    return original_exclude_files(
        quality_csv, pre_process_csv, input_dir, excluded_dir,
        frame_count_threshold=frame_count_min, coverage_threshold_mm=coverage_threshold_mm,
        final_csv=final_csv, dry_run=dry_run,
        reason_function=partial(exclusion_reason, frame_count_max=frame_count_max))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", type=Path, required=True, help="quality_v2.py output")
    parser.add_argument("--pre-process-csv", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--excluded-dir", type=Path, required=True)
    parser.add_argument("--frame-count-min", type=int, default=20, help="Minimum inclusive frame count (default: 20)")
    parser.add_argument("--frame-count-max", type=int, default=40, help="Maximum inclusive frame count (default: 40)")
    parser.add_argument("--coverage-threshold-mm", type=float, required=True)
    parser.add_argument("--final-csv", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    exclude_files(args.quality_csv, args.pre_process_csv, args.input_dir, args.excluded_dir,
                  frame_count_min=args.frame_count_min, frame_count_max=args.frame_count_max,
                  coverage_threshold_mm=args.coverage_threshold_mm,
                  final_csv=args.final_csv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
