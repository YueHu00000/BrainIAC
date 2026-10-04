"""Move preprocessed series below the frame-count or coverage threshold."""

import argparse
import math
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import read_ids, read_rows, scan_outputs, validate_id, write_rows


def exclusion_reason(row, frame_count_threshold, coverage_threshold_mm):
    if row is None:
        return "missing_quality"
    try:
        frames, coverage = float(row["frame_count"]), float(row["coverage_mm"])
    except (KeyError, TypeError, ValueError):
        return "unknown_quality"
    reasons = []
    if not math.isfinite(frames) or frames <= 0 or frames != int(frames):
        reasons.append("unknown_frame_count")
    elif frames < frame_count_threshold:
        reasons.append("low_frame_count")
    if not math.isfinite(coverage) or coverage < 0:
        reasons.append("unknown_coverage")
    elif coverage < coverage_threshold_mm:
        reasons.append("low_coverage")
    return ";".join(reasons)


def exclude_files(quality_csv, pre_process_csv, input_dir, excluded_dir, *,
                  frame_count_threshold, coverage_threshold_mm, final_csv=None,
                  dry_run=False):
    if (frame_count_threshold < 1 or not math.isfinite(coverage_threshold_mm)
            or coverage_threshold_mm < 0):
        raise ValueError("Require frame_count_threshold >= 1 and finite coverage_threshold_mm >= 0")
    input_dir, excluded_dir = Path(input_dir).resolve(), Path(excluded_dir).resolve()
    if (input_dir == excluded_dir or input_dir in excluded_dir.parents
            or excluded_dir in input_dir.parents):
        raise ValueError("Preprocess and excluded directories must not overlap")
    rows = read_rows(quality_csv)
    quality = {row["unique_id"]: row for row in rows}
    if len(rows) != len(quality):
        raise ValueError("Duplicate unique_id in quality CSV")
    ids = read_ids(pre_process_csv)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate unique_id in preprocessing CSV")
    moves = []
    for unique_id in ids:
        validate_id(unique_id)
        source = input_dir / f"{unique_id}.nii.gz"
        if not source.is_file():
            print(f"[skip] {unique_id}: no remaining preprocess file", flush=True)
            continue
        reason = exclusion_reason(quality.get(unique_id), frame_count_threshold,
                                  coverage_threshold_mm)
        if not reason:
            continue
        target = excluded_dir / source.name
        if target.exists():
            raise FileExistsError(f"Excluded destination already exists: {target}")
        # Moves must stay inside the two named roots, including resolved links.
        if source.resolve().parent != input_dir or target.resolve().parent != excluded_dir:
            raise ValueError(f"Move would leave the selected directories: {source}")
        moves.append((source, target, {"unique_id": unique_id, "reason": reason}))
    for _, target, row in moves:
        print(f"[{'dry-run' if dry_run else 'exclude'}] {row['unique_id']}: "
              f"{row['reason']} -> {target}", flush=True)
    if dry_run:
        return [row for _, _, row in moves]

    excluded_dir.mkdir(parents=True, exist_ok=True)
    report = excluded_dir / "excluded_preprocess.csv"
    previous = read_rows(report) if report.is_file() else []
    history = {row["unique_id"]: row for row in previous}
    final_csv = Path(final_csv) if final_csv is not None else input_dir / "final_pre_process.csv"
    try:
        for source, target, row in moves:
            shutil.move(str(source), str(target))
            history[row["unique_id"]] = row
            write_rows(report, history.values(), ["unique_id", "reason"])
        if not moves:
            write_rows(report, history.values(), ["unique_id", "reason"])
    finally:
        scan_outputs(input_dir, final_csv)
    return [row for _, _, row in moves]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--pre-process-csv", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--excluded-dir", type=Path, required=True)
    parser.add_argument("--frame-count-threshold", type=int, required=True)
    parser.add_argument("--coverage-threshold-mm", type=float, required=True)
    parser.add_argument("--final-csv", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    exclude_files(args.quality_csv, args.pre_process_csv, args.input_dir,
                  args.excluded_dir, frame_count_threshold=args.frame_count_threshold,
                  coverage_threshold_mm=args.coverage_threshold_mm,
                  final_csv=args.final_csv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
