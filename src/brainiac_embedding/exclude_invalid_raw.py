"""Quarantine raw T1/T2 pairs that fail the minimum 3-D size requirement."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import SimpleITK as sitk


RESULTS_ROOT = Path('/mnt/image_test/C2D2AI/BrainMRI/BrainIAC/results')
FIELDS = ('Study_ID', 'T1_size', 'T2_size', 'Reason')


def inspect_study(study: Path) -> tuple[list[str], list[str]]:
    """Read headers only; sizes use SimpleITK's (x, y, z) order."""
    sizes = []
    reasons = []
    for modality in ('T1', 'T2'):
        path = study / f'{modality}.nii.gz'
        if not path.is_file():
            sizes.append('MISSING')
            reasons.append(f'{modality}: missing file')
            continue
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.ReadImageInformation()
            size = reader.GetSize()
            sizes.append(str(tuple(size)))
            if reader.GetDimension() != 3:
                reasons.append(f'{modality}: dimension={reader.GetDimension()}, expected 3')
            if reader.GetNumberOfComponents() != 1:
                reasons.append(f'{modality}: expected a scalar image')
            small_axes = [i for i, length in enumerate(size) if length < 4]
            if small_axes:
                reasons.append(f'{modality}: size < 4 on axes {small_axes}')
        except RuntimeError as error:
            sizes.append('UNREADABLE')
            reasons.append(f'{modality}: unreadable header: {" ".join(str(error).split())}')
    return sizes, reasons


def quarantine(raw_root: Path, excluded_root: Path, *, dry_run: bool = False) -> int:
    raw_root = raw_root.resolve(strict=True)
    excluded_root = excluded_root.resolve()
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    if (raw_root == excluded_root or raw_root in excluded_root.parents
            or excluded_root in raw_root.parents):
        raise ValueError('raw-root and excluded-root must be separate, non-nested directories')

    invalid = []
    checked = 0
    for study in sorted(raw_root.iterdir()):
        if study.name in ('.convert_tmp', '.preprocess_tmp'):
            continue
        if study.is_symlink() or (hasattr(study, 'is_junction') and study.is_junction()):
            raise ValueError(f'refusing linked Study path: {study}')
        if not study.is_dir():
            continue
        sizes, reasons = inspect_study(study)
        checked += 1
        if reasons:
            destination = excluded_root / study.name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f'excluded Study already exists; nothing moved: {destination}')
            invalid.append((study, sizes, reasons))
            print(f'[exclude] {study.name}; T1={sizes[0]}; T2={sizes[1]}; {"; ".join(reasons)}')

    if dry_run:
        print(f'Dry run: checked={checked}, would_move={len(invalid)}; no files changed')
        return len(invalid)

    excluded_root.mkdir(parents=True, exist_ok=True)
    # Rename keeps the whole Study together. Require the same filesystem so an
    # interrupted cross-filesystem copy cannot leave a partial quarantine.
    if invalid and raw_root.stat().st_dev != excluded_root.stat().st_dev:
        raise ValueError('raw-root and excluded-root must be on the same filesystem')
    log_path = excluded_root / 'Exclude_Reason.txt'
    if log_path.is_symlink():
        raise ValueError(f'refusing linked log file: {log_path}')
    with log_path.open('a+', encoding='utf-8', newline='') as log:
        log.seek(0)
        header = next(csv.reader(log, delimiter='\t'), None)
        if header is not None and header != list(FIELDS):
            raise ValueError(f'unrecognized log header; existing log preserved: {log_path}')
        log.seek(0, os.SEEK_END)
        writer = csv.writer(log, delimiter='\t', lineterminator='\n')
        if header is None:
            writer.writerow(FIELDS)
            log.flush()
            os.fsync(log.fileno())
        for study, sizes, reasons in invalid:
            destination = excluded_root / study.name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            study.rename(destination)
            writer.writerow((study.name, *sizes, '; '.join(reasons)))
            log.flush()
            os.fsync(log.fileno())
            print(f'[moved] {study.name} -> {destination}')
    print(f'Finished: checked={checked}, kept={checked - len(invalid)}, moved={len(invalid)}')
    return len(invalid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=RESULTS_ROOT / 'brainiac_raw_nifti')
    parser.add_argument('--excluded-root', type=Path, default=RESULTS_ROOT / 'excluded_raw_nifti')
    parser.add_argument('--dry-run', action='store_true', help='Inspect and report without moving or writing files')
    args = parser.parse_args()
    quarantine(args.raw_root, args.excluded_root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
