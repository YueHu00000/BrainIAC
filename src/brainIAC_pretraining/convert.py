"""Convert manifest-selected DICOM files into one raw NIfTI per series."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import MultiFrameDicomError, read_rows, scan_outputs, validate_id, write_rows


def convert_series(row: dict, output_dir: Path) -> Path:
    unique_id = row["unique_id"]
    validate_id(unique_id)
    destination = output_dir / f"{unique_id}.nii.gz"
    if destination.is_file():
        print(f"[skip] convert {unique_id}", flush=True)
        return destination

    import SimpleITK as sitk
    import pydicom

    files = [str(Path(row["study_directory"]) / name) for name in json.loads(row["file_names"])]
    for filename in files:
        header = pydicom.dcmread(filename, stop_before_pixels=True, specific_tags=["NumberOfFrames"])
        frame_count = int(getattr(header, "NumberOfFrames", 1))
        if frame_count > 1:
            raise MultiFrameDicomError(f"{unique_id}: multi-frame DICOM is unsupported: {filename} (NumberOfFrames={frame_count})")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("selected series is not a scalar 3D volume")
    scratch = output_dir / ".convert_tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=f"{unique_id}_", dir=scratch) as temporary:
        temporary_path = Path(temporary) / "volume.nii.gz"
        sitk.WriteImage(image, str(temporary_path))
        os.replace(temporary_path, destination)
    print(f"[done] convert {unique_id}", flush=True)
    return destination


def convert_manifest(manifest: str | Path, output_dir: str | Path,
                     converted_csv: str | Path | None = None) -> list[str]:
    output_dir = Path(output_dir).resolve()
    converted_csv = Path(converted_csv) if converted_csv else output_dir / "converted.csv"
    rows = read_rows(manifest)
    for row in rows:
        source = Path(row["study_directory"]).resolve()
        if source == output_dir or source in output_dir.parents or output_dir in source.parents:
            raise ValueError("DICOM study directories and output-dir must not overlap")
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    try:
        for row in rows:
            try:
                convert_series(row, output_dir)
            except MultiFrameDicomError as error:
                failures.append({"unique_id": row["unique_id"], "reason": str(error)})
                raise
            except Exception as error:
                failures.append({"unique_id": row["unique_id"], "reason": str(error)})
                print(f"[failed] convert {row['unique_id']}: {error}", flush=True)
    finally:
        completed = scan_outputs(output_dir, converted_csv)
        write_rows(output_dir / "convert_errors.csv", failures, ["unique_id", "reason"])
    return completed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--converted-csv")
    args = parser.parse_args(argv)
    convert_manifest(args.manifest, args.output_dir, args.converted_csv)


if __name__ == "__main__":
    main()
