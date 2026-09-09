"""Study output validation and bounded cleanup shared by the three stages."""

from __future__ import annotations

import gzip
import os
import re
import shutil
import stat
import zlib
from pathlib import Path

import numpy as np


def validate_study_id(study_id: str) -> None:
    if (not study_id or study_id.startswith('.') or
            any(char in study_id for char in '/\\:*?"<>|') or
            study_id.rstrip(' .') != study_id):
        raise ValueError(f"Study ID must be a plain directory name: {study_id!r}")


def check_separate_roots(input_root: Path, output_root: Path) -> None:
    source, target = input_root.resolve(), output_root.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError(f"input and output roots must not overlap: {source}, {target}")


def remove_output(path: Path, root: Path) -> None:
    """Remove only a strict descendant of root, refusing links/junctions."""
    path = Path(os.path.abspath(path))
    root = root.resolve()
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError(f"cleanup target is outside output root: {path}")
    # Check ancestors as well as contents before any recursive deletion.
    if root not in path.parents:
        raise ValueError(f"cleanup target is outside output root: {path}")
    candidates = []
    cursor = path
    while cursor != root:
        candidates.append(cursor)
        cursor = cursor.parent
    if path.is_dir():
        for directory, dirs, files in os.walk(path, followlinks=False):
            candidates.extend(Path(directory) / name for name in dirs + files)
    for candidate in candidates:
        if candidate.is_symlink() or (
            candidate.exists() and
            getattr(candidate.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"refusing cleanup through a link/junction: {candidate}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_study_scratch(root: Path, stage: str, study_id: str) -> Path:
    validate_study_id(study_id)
    parent = root / f'.{stage}_tmp'
    work = parent / f'{study_id}.work'
    remove_output(work, root)
    if stage == 'preprocess' and parent.is_dir():
        # Previous releases used tempfile's eight-character random suffix.
        legacy = re.compile(re.escape(study_id) + r'_[a-z0-9_]{8}')
        for child in parent.iterdir():
            if legacy.fullmatch(child.name):
                remove_output(child, root)
    return work


def nifti_pair_complete(directory: Path, suffix: str = '') -> bool:
    import SimpleITK as sitk

    for modality in ('T1', 'T2'):
        path = directory / f'{modality}{suffix}.nii.gz'
        if not path.is_file():
            return False
        try:
            # ITK may accept a truncated gzip; exhaust it to check CRC/footer.
            with gzip.open(path, 'rb') as stream:
                while stream.read(1024 * 1024):
                    pass
            image = sitk.ReadImage(str(path))
            if (image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1 or
                    not all(image.GetSize()) or
                    not np.isfinite(sitk.GetArrayViewFromImage(image)).all()):
                return False
        except (OSError, EOFError, RuntimeError, ValueError, zlib.error):
            return False
    return True
