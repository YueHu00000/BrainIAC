"""Preprocess one corrected T1/T2 pair with the existing BrainIAC workflow."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainiac_embedding import preprocess_t1t2
from brainiac_embedding._resume import check_separate_roots, validate_study_id


def preprocess_study(study_id, raw_root, processed_root, *, brainiac_root=None):
    """Keep the batch imaging operations, but refuse existing study outputs."""
    validate_study_id(study_id)
    raw_root, processed_root = Path(raw_root).resolve(), Path(processed_root).resolve()
    check_separate_roots(raw_root, processed_root)
    for modality in ("T1", "T2"):
        source = raw_root / study_id / f"{modality}.nii.gz"
        if not source.is_file():
            raise FileNotFoundError(f"raw NIfTI does not exist: {source}")
    target = processed_root / study_id
    if target.exists():
        raise FileExistsError(f"Use a new processed root; already exists: {target}")
    return preprocess_t1t2.preprocess_study(
        study_id, raw_root, processed_root, brainiac_root=brainiac_root,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="R01_Study_002904")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--brainiac-root", type=Path)
    args = parser.parse_args(argv)
    paths = preprocess_study(
        args.study_id, args.raw_root, args.processed_root, brainiac_root=args.brainiac_root,
    )
    for path in paths:
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
