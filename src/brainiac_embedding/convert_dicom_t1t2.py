"""Convert KPSC-style DICOM folders to paired T1/T2 NIfTI volumes.

The series grouping, exception handling, T1/T2 matching, and SimpleITK
assembly intentionally mirror ``neuroimage-classifiers`` commit
8bdb9fafe3169174529a8eccad078853d7b39b8c.  Only study discovery differs:
this module resolves CSV-listed Study IDs across roots listed in a text file.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from ._resume import (
    check_separate_roots,
    clean_study_scratch,
    nifti_pair_complete,
    remove_output,
    validate_study_id,
)

EXCEPTIONS = {
    "STUDY_0230": 8,
    "STUDY_0462": 9,
    "STUDY_0836": 9,
    "STUDY_1109": 4,
}


def read_folder_list(path: str | Path) -> list[Path]:
    """Read non-empty DICOM roots from a UTF-8 text file."""
    list_path = Path(path)
    roots = [Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not roots:
        raise ValueError(f"folder list is empty: {list_path}")
    missing = [root for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"DICOM root does not exist: {missing[0]}")
    return roots


def read_study_ids(csv_path: str | Path, study_id_column: str = "Study_ID") -> list[str]:
    """Read the ordered, unique Study-ID whitelist from a CSV."""
    frame = pd.read_csv(csv_path, dtype={study_id_column: str})
    if study_id_column not in frame.columns:
        raise KeyError(f"CSV is missing required column {study_id_column!r}")
    if frame[study_id_column].isna().any():
        raise ValueError(f"CSV column {study_id_column!r} contains empty values")
    study_ids = [value.strip() for value in frame[study_id_column].tolist()]
    if any(not value for value in study_ids):
        raise ValueError(f"CSV column {study_id_column!r} contains empty values")
    for study_id in study_ids:
        validate_study_id(study_id)
    duplicate_mask = pd.Series(study_ids).duplicated()
    duplicates = [study_id for study_id, duplicate in zip(study_ids, duplicate_mask) if duplicate]
    if duplicates:
        raise ValueError(f"CSV contains duplicate Study ID: {duplicates[0]}")
    return study_ids


def resolve_study_dir(study_id: str, roots: Sequence[Path]) -> Path:
    """Resolve one Study ID to exactly one ``<root>/<Study_ID>`` directory."""
    matches = [root / study_id for root in roots if (root / study_id).is_dir()]
    if not matches:
        raise FileNotFoundError(f"Study folder was not found for {study_id}")
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(f"Study folder is ambiguous for {study_id}: {rendered}")
    return matches[0]


def group_dicom_files_in_dir(directory: str | Path) -> pd.DataFrame:
    """Mirror ``neuroimage-classifiers.src.mri.group_dicom_files_in_dir``."""
    import pydicom

    directory = str(directory)
    columns = ["Filename", "SeriesNumber", "SeriesDescription", "InstanceNumber"]
    dtypes = {
        "Filename": "string",
        "SeriesNumber": "int",
        "SeriesDescription": "string",
        "InstanceNumber": "int",
    }
    folder_df = pd.DataFrame(columns=columns).astype(dtypes)

    # Keep this loop and the four DICOM fields aligned with the reference code.
    for filename in os.listdir(directory):
        ds = pydicom.dcmread(f"{directory}/{filename}")
        folder_df.loc[len(folder_df)] = [
            filename,
            ds.SeriesNumber,
            ds.SeriesDescription,
            ds.InstanceNumber,
        ]

    folder_df = folder_df.sort_values(by=["SeriesNumber", "InstanceNumber"])
    groupby_columns = ["SeriesNumber", "SeriesDescription"]
    groupby_dict = {"Filename": list, "InstanceNumber": list}
    grouped_df = folder_df.groupby(groupby_columns).agg(groupby_dict).reset_index()
    return grouped_df.rename(columns={"Filename": "Filenames"})


def select_t1_t2_rows(grouped_df: pd.DataFrame, study_id: str) -> tuple[pd.Series, pd.Series]:
    """Apply the unmodified KPSC exception and max-SeriesNumber rules."""
    if study_id in EXCEPTIONS:
        grouped_df = grouped_df[grouped_df.SeriesNumber != EXCEPTIONS[study_id]]

    t1_rows = grouped_df[grouped_df.SeriesDescription.str.contains("T1", case=False, na=False)]
    t2_rows = grouped_df[grouped_df.SeriesDescription.str.contains("T2", case=False, na=False)]
    t1_row = t1_rows.loc[t1_rows["SeriesNumber"].idxmax()]
    t2_row = t2_rows.loc[t2_rows["SeriesNumber"].idxmax()]
    return t1_row, t2_row


def convert_study(study_id: str, study_dir: str | Path, output_root: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Convert the selected T1 and T2 series for one Study."""
    import SimpleITK as sitk

    validate_study_id(study_id)
    study_dir = Path(study_dir).resolve()
    output_root = Path(output_root).resolve()
    check_separate_roots(study_dir, output_root)
    output_dir = output_root / study_id
    t1_output = output_dir / "T1.nii.gz"
    t2_output = output_dir / "T2.nii.gz"
    if not overwrite and nifti_pair_complete(output_dir):
        print(f"[skip] conversion {study_id}")
        return t1_output, t2_output
    temporary = clean_study_scratch(output_root, "convert", study_id)
    remove_output(output_dir, output_root)
    print(f"[compute] conversion {study_id}")
    grouped_df = group_dicom_files_in_dir(study_dir)
    t1_row, t2_row = select_t1_t2_rows(grouped_df, study_id)
    t1_filepaths = [str(study_dir / filename) for filename in t1_row.Filenames]
    t2_filepaths = [str(study_dir / filename) for filename in t2_row.Filenames]

    temporary.mkdir(parents=True)
    try:
        reader = sitk.ImageSeriesReader()
        for modality, filepaths in (("T1", t1_filepaths), ("T2", t2_filepaths)):
            reader.SetFileNames(filepaths)
            image = reader.Execute()
            sitk.WriteImage(image, str(temporary / f"{modality}.nii.gz"))
        if not nifti_pair_complete(temporary):
            raise RuntimeError(f"conversion did not produce valid paired NIfTI: {study_id}")
        os.replace(temporary, output_dir)
    finally:
        remove_output(temporary, output_root)
    return t1_output, t2_output


def convert_whitelist(
    folder_list: str | Path,
    csv_path: str | Path,
    output_root: str | Path,
    *,
    study_id_column: str = "Study_ID",
    overwrite: bool = False,
) -> list[str]:
    """Convert every CSV-whitelisted Study in CSV order."""
    roots = read_folder_list(folder_list)
    study_ids = read_study_ids(csv_path, study_id_column)
    for root in roots:
        check_separate_roots(root, Path(output_root))
    for study_id in study_ids:
        if not overwrite and nifti_pair_complete(Path(output_root) / study_id):
            print(f"[skip] conversion {study_id}")
            continue
        study_dir = resolve_study_dir(study_id, roots)
        convert_study(study_id, study_dir, output_root, overwrite=overwrite)
    return study_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV-selected KPSC-style DICOM studies to T1/T2 NIfTI")
    parser.add_argument("--folder-list", required=True, help="UTF-8 text file containing one DICOM root per line")
    parser.add_argument("--csv", required=True, help="CSV containing the Study-ID whitelist")
    parser.add_argument("--study-id-column", default="Study_ID", help="Study-ID column name; default: Study_ID")
    parser.add_argument("--output-root", required=True, help="Destination for <Study_ID>/T1.nii.gz and T2.nii.gz")
    parser.add_argument("--overwrite", action="store_true", help="Force recomputation, including completed T1/T2 outputs")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    study_ids = convert_whitelist(
        args.folder_list,
        args.csv,
        args.output_root,
        study_id_column=args.study_id_column,
        overwrite=args.overwrite,
    )
    print(f"Conversion finished for {len(study_ids)} studies (computed or skipped) to {args.output_root}")


if __name__ == "__main__":
    main()
