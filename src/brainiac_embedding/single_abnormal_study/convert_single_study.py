"""Convert one study to raw T1 and two acquisition-specific T2 NIfTIs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError


def select_series(dicom_dir):
    """Keep the upstream maximum-SeriesNumber rule for T1 and T2."""
    groups = defaultdict(list)
    for path in sorted(Path(dicom_dir).iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        description = str(getattr(ds, "SeriesDescription", ""))
        if "T1" in description.upper() or "T2" in description.upper():
            groups[(int(ds.SeriesNumber), description)].append((path, ds))

    selected = {}
    for modality in ("T1", "T2"):
        candidates = [key for key in groups if modality in key[1].upper()]
        if not candidates:
            raise ValueError(f"No {modality} series found in {dicom_dir}")
        maximum = max(key[0] for key in candidates)
        winners = [key for key in candidates if key[0] == maximum]
        if len(winners) != 1:
            raise ValueError(f"Ambiguous {modality} series at SeriesNumber={maximum}")
        records = groups[winners[0]]
        if len({str(ds.SeriesInstanceUID) for _, ds in records}) != 1:
            raise ValueError(f"Selected {modality} group contains multiple SeriesInstanceUIDs")
        selected[modality] = records
    return selected


def order_slices(records):
    """Order classic single-frame slices by position along their shared normal."""
    orientation = np.asarray(records[0][1].ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    ordered = []
    for path, ds in records:
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError(f"Only single-frame DICOM is supported: {path}")
        if not np.allclose(ds.ImageOrientationPatient, orientation, rtol=0, atol=1e-4):
            raise ValueError(f"Slice orientations differ: {path}")
        position = float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
        ordered.append((position, path, ds))
    ordered.sort(key=lambda record: record[0])
    if len(ordered) < 2 or np.any(np.diff([row[0] for row in ordered]) <= 0.01):
        raise ValueError("Expected at least two slices with no repeated planes (0.01 mm tolerance)")
    return ordered


def convert_study(dicom_dir, output_dir):
    selected = select_series(dicom_dir)
    acquisitions = defaultdict(list)
    for path, ds in selected["T2"]:
        acquisition = getattr(ds, "AcquisitionNumber", None)
        if acquisition is None or str(acquisition).strip() == "":
            raise ValueError(f"T2 AcquisitionNumber is missing: {path}")
        acquisitions[int(acquisition)].append((path, ds))
    if set(acquisitions) != {1, 2}:
        raise ValueError(f"Expected exactly T2 acquisitions 1 and 2, found {sorted(acquisitions)}")
    volumes = {"T1": order_slices(selected["T1"])}
    volumes.update({f"T2_acq{number}": order_slices(acquisitions[number]) for number in (1, 2)})

    output_dir = Path(output_dir)
    targets = [output_dir / f"{name}.nii.gz" for name in volumes]
    targets.append(output_dir / "selected_files.csv")
    for path in targets:
        if path.exists():
            raise FileExistsError(f"Use a new output directory; already exists: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, records in volumes.items():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(path.resolve()) for _, path, _ in records])
        image = reader.Execute()
        if image.GetDimension() != 3 or image.GetSize()[2] != len(records):
            raise ValueError(f"Unexpected converted shape for {name}: {image.GetSize()}")
        sitk.WriteImage(image, str(output_dir / f"{name}.nii.gz"))
        first = records[0][2]
        print(f"{name}: series={first.SeriesNumber} / {first.SeriesDescription}; "
              f"files={len(records)}; size={image.GetSize()}; spacing={image.GetSpacing()}", flush=True)
        for index, (position, path, ds) in enumerate(records):
            rows.append({"volume": name, "slice_index": index, "file": str(path.resolve()),
                         "series_uid": str(ds.SeriesInstanceUID),
                         "acquisition": str(getattr(ds, "AcquisitionNumber", "")),
                         "instance": str(getattr(ds, "InstanceNumber", "")),
                         "position_mm": position})
    with (output_dir / "selected_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert_study(args.dicom_dir, args.output_dir)


if __name__ == "__main__":
    main()
