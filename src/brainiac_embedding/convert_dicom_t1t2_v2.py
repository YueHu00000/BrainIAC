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


def group_dicom_files_in_dir_optimized(directory: str | Path) -> pd.DataFrame:
    import pydicom
    from pathlib import Path

    directory = Path(directory)
    data = []

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        try:
            # stop_before_pixels=True 只读取元数据，极大提升速度并节省内存
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            data.append({
                "Filename": file_path.name,
                "SeriesNumber": getattr(ds, "SeriesNumber", -1),
                "SeriesDescription": getattr(ds, "SeriesDescription", "Unknown"),
                "InstanceNumber": getattr(ds, "InstanceNumber", -1),
            })
        except Exception:
            # 自动跳过非 DICOM 文件
            continue

    if not data:
        return pd.DataFrame(columns=["SeriesNumber", "SeriesDescription", "Filenames", "InstanceNumber"])

    folder_df = pd.DataFrame(data)
    folder_df = folder_df.sort_values(by=["SeriesNumber", "InstanceNumber"])

    grouped_df = (
        folder_df.groupby(["SeriesNumber", "SeriesDescription"])
        .agg({"Filename": list, "InstanceNumber": list})
        .reset_index()
        .rename(columns={"Filename": "Filenames"})
    )

    return grouped_df

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

    study_dir = Path(study_dir)
    output_dir = Path(output_root) / study_id
    t1_output = output_dir / "T1.nii.gz"
    t2_output = output_dir / "T2.nii.gz"
    existing = [path for path in (t1_output, t2_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output already exists: {existing[0]}")
    grouped_df = group_dicom_files_in_dir(study_dir)
    t1_row, t2_row = select_t1_t2_rows(grouped_df, study_id)
    t1_filepaths = [str(study_dir / filename) for filename in t1_row.Filenames]
    t2_filepaths = [str(study_dir / filename) for filename in t2_row.Filenames]

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(t1_filepaths)
    image = reader.Execute()
    sitk.WriteImage(image, str(t1_output))
    reader.SetFileNames(t2_filepaths)
    image = reader.Execute()
    sitk.WriteImage(image, str(t2_output))
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
    for study_id in study_ids:
        study_dir = resolve_study_dir(study_id, roots)
        convert_study(study_id, study_dir, output_root, overwrite=overwrite)
    return study_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV-selected KPSC-style DICOM studies to T1/T2 NIfTI")
    parser.add_argument("--folder-list", required=True, help="UTF-8 text file containing one DICOM root per line")
    parser.add_argument("--csv", required=True, help="CSV containing the Study-ID whitelist")
    parser.add_argument("--study-id-column", default="Study_ID", help="Study-ID column name; default: Study_ID")
    parser.add_argument("--output-root", required=True, help="Destination for <Study_ID>/T1.nii.gz and T2.nii.gz")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing T1/T2 outputs")
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
    print(f"Converted {len(study_ids)} studies to {args.output_root}")


if __name__ == "__main__":
    main()
