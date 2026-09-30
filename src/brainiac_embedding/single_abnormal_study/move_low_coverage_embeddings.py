"""Move study NPZs when either T1 or T2 coverage is below the threshold."""

import argparse
import csv
import math
import shutil
from pathlib import Path


def low_coverage_ids(csv_path, threshold=101.0):
    """Read exporter display IDs; unknown/invalid coverage is not a low value."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    selected = set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = ("Study_ID", "T1_coverage_mm", "T2_coverage_mm")
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {columns}")
        unknown_rows = 0
        for row in reader:
            study_id = (row["Study_ID"] or "").strip().removeprefix("excluded_")
            if (not study_id or study_id.startswith(".") or study_id.rstrip(" .") != study_id
                    or any(char in study_id for char in '/\\:*?"<>|')):
                raise ValueError(f"Invalid Study_ID on CSV line {reader.line_num}: {study_id!r}")
            coverage = []
            for column in columns[1:]:
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    value = math.nan
                coverage.append(value if math.isfinite(value) and value >= 0 else math.nan)
            unknown_rows += any(math.isnan(value) for value in coverage)
            if any(value < threshold for value in coverage):
                selected.add(study_id)
    print(f"Selected CSV studies: {len(selected)}; threshold: < {threshold:g} mm")
    if unknown_rows:
        print(f"Rows with unknown/invalid coverage: {unknown_rows}; only valid values can qualify")
    return selected


def move_embeddings(csv_path, embedding_dir, transfer_dir, *, threshold=101.0, dry_run=False):
    """Move exact-ID NPZ matches, keeping paths relative to the embedding root."""
    selected = low_coverage_ids(csv_path, threshold)
    source_root = Path(embedding_dir).resolve(strict=True)
    target_root = Path(transfer_dir).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if target_root.exists() and not target_root.is_dir():
        raise NotADirectoryError(target_root)
    if (source_root == target_root or source_root in target_root.parents
            or target_root in source_root.parents):
        raise ValueError("embedding-dir and transfer-dir must be separate, non-nested directories")

    planned = []
    found = set()
    for source in sorted(source_root.rglob("*.npz")):
        if source.stem not in selected or not source.is_file():
            continue
        if source.is_symlink() or source_root not in source.resolve().parents:
            raise ValueError(f"Refusing linked embedding outside the source directory: {source}")
        relative = source.relative_to(source_root)
        target = target_root / relative
        if target_root not in target.resolve().parents:
            raise ValueError(f"Destination resolves outside transfer-dir: {target}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Destination already exists; nothing moved: {target}")
        for parent in target.parents:
            if parent.exists() and not parent.is_dir():
                raise NotADirectoryError(parent)
        planned.append((source, target))
        found.add(source.stem)

    missing = sorted(selected - found)
    for study_id in missing:
        print(f"[missing] {study_id}: no matching NPZ in embedding-dir (possibly already moved)")
    for source, target in planned:
        print(f"[{'would-move' if dry_run else 'move'}] {source} -> {target}", flush=True)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            shutil.move(str(source), str(target))
    stats = {"selected_studies": len(selected), "matched_studies": len(found),
             "missing_studies": len(missing), "files": len(planned), "dry_run": dry_run}
    print(f"{'Dry run' if dry_run else 'Finished'}: {stats}")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="low_coverage_clinical.csv")
    parser.add_argument("--threshold", type=float, default=101.0, help="Strict upper bound in mm (default: 101)")
    parser.add_argument("--embedding-dir", type=Path, required=True,
                        help="BrainIAC or vit_survival embedding root; searched recursively")
    parser.add_argument("--transfer-dir", type=Path, required=True, help="Destination preserving relative paths")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without moving or creating files")
    args = parser.parse_args(argv)
    move_embeddings(args.csv, args.embedding_dir, args.transfer_dir,
                    threshold=args.threshold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
