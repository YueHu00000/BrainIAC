"""List T1 series and their geometry without reading DICOM pixel data."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


EXCEPTIONS = {"STUDY_0230": 8, "STUDY_0462": 9, "STUDY_0836": 9, "STUDY_1109": 4}
FIELDS = (
    "SeriesNumber", "SeriesDescription", "ProtocolName", "Modality", "ImageType",
    "SOPClassUID", "Rows", "Columns", "PixelSpacing", "SliceThickness",
    "SpacingBetweenSlices", "ImageOrientationPatient", "RepetitionTime",
    "EchoTime", "InversionTime", "FlipAngle", "ScanningSequence",
    "SequenceVariant", "MRAcquisitionType", "FrameOfReferenceUID",
    "RescaleSlope", "RescaleIntercept",
)


def value(ds, key):
    return str(getattr(ds, key, "<missing>"))


def geometry(ds, frame=None):
    """Get classic or enhanced MR geometry, including shared/per-frame overrides."""
    names = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "SliceThickness")
    result = {name: getattr(ds, name, None) for name in names}
    shared = getattr(ds, "SharedFunctionalGroupsSequence", [])
    groups = list(shared[:1]) + ([frame] if frame is not None else [])
    for group in groups:
        for sequence, keys in (
            ("PlanePositionSequence", names[:1]),
            ("PlaneOrientationSequence", names[1:2]),
            ("PixelMeasuresSequence", names[2:]),
        ):
            items = getattr(group, sequence, [])
            if items:
                for key in keys:
                    result[key] = getattr(items[0], key, result[key])
    return result


def build_report(source, study_id=None):
    source = Path(source).resolve()
    directory = source.parent if source.is_file() else source
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    study_id = study_id or directory.name
    groups, skipped = defaultdict(list), []
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    for path in files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError, EOFError) as exc:
            skipped.append(f"{path.relative_to(directory)}: {type(exc).__name__}: {exc}")
            continue
        uid = value(ds, "SeriesInstanceUID")
        # Keep missing-UID records separate rather than inventing a series identity.
        key = (value(ds, "StudyInstanceUID"), uid if uid != "<missing>" else str(path))
        groups[key].append((path, ds))
    t1_groups = [items for items in groups.values()
                 if any("t1" in value(ds, "SeriesDescription").lower() for _, ds in items)]
    lines = [
        "DICOM T1 SERIES REPORT", f"Input: {source}", f"Scanned directory (recursive): {directory}",
        f"Study ID used for Harvey exceptions: {study_id}",
        f"Files scanned: {len(files)}", f"DICOM headers read: {sum(map(len, groups.values()))}",
        f"T1 series groups: {len(t1_groups)}", f"Unreadable/non-DICOM files: {len(skipped)}",
        "T1 rule: SeriesDescription contains T1, case insensitive (not a validated sequence classification).",
        "Grouping: StudyInstanceUID + SeriesInstanceUID; missing series UIDs stay separate.",
        "Harvey instead groups by (SeriesNumber, SeriesDescription), selects max SeriesNumber,",
        "and has no minimum slice count. Identical grouping keys can mix different UIDs.",
        f"Harvey excluded SeriesNumber for this study ID: {EXCEPTIONS.get(study_id, 'none')}",
        "Recursive discovery here differs from Harvey's direct-child-only directory scan.",
        "Coordinates: DICOM patient LPS, millimetres. PixelSpacing order: row, column.",
        "Pixel data NOT read. File count, frame count and distinct spatial positions are different quantities.",
        "Missing NumberOfFrames is counted as 1. SliceThickness is not necessarily inter-slice spacing.",
    ]
    for index, items in enumerate(t1_groups, 1):
        items.sort(key=lambda item: (int(getattr(item[1], "InstanceNumber", -1)), str(item[0])))
        first = items[0][1]
        frame_count = sum(int(getattr(ds, "NumberOfFrames", 1)) for _, ds in items)
        lines += ["", f"=== T1 SERIES {index} ===",
                  f"StudyInstanceUID: {value(first, 'StudyInstanceUID')}",
                  f"SeriesInstanceUID: {value(first, 'SeriesInstanceUID')}",
                  f"File count: {len(items)}", f"Total frames: {frame_count}"]
        for field in FIELDS:
            distinct = list(dict.fromkeys(value(ds, field) for _, ds in items))
            lines.append(f"{field} (all distinct values): {' ; '.join(distinct)}")
        sop_uids = [value(ds, "SOPInstanceUID") for _, ds in items if hasattr(ds, "SOPInstanceUID")]
        lines.append(f"Repeated SOPInstanceUID entries: {len(sop_uids) - len(set(sop_uids))}")
        positions, orientations = [], []
        for path, ds in items:
            lines += [f"  File: {path.relative_to(directory)}",
                      f"    InstanceNumber={value(ds, 'InstanceNumber')}; SOPInstanceUID={value(ds, 'SOPInstanceUID')}",
                      f"    NumberOfFrames={getattr(ds, 'NumberOfFrames', 1)}; SeriesNumber={value(ds, 'SeriesNumber')}; SeriesDescription={value(ds, 'SeriesDescription')}",
                      f"    TransferSyntaxUID={value(ds.file_meta, 'TransferSyntaxUID')}"]
            frames = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
            if int(getattr(ds, "NumberOfFrames", 1)) > 1 and not frames:
                lines.append("    Per-frame geometry unavailable; top-level/shared geometry only follows.")
            for n, frame in enumerate(frames or [None], 1):
                geo = geometry(ds, frame)
                lines.append(f"    Frame {n}: " + "; ".join(f"{k}={v}" for k, v in geo.items()))
                if geo["ImagePositionPatient"] is not None:
                    positions.append([float(v) for v in geo["ImagePositionPatient"]])
                if geo["ImageOrientationPatient"] is not None:
                    orientations.append([float(v) for v in geo["ImageOrientationPatient"]])
        lines.append(f"Frames/records with position: {len(positions)} / {frame_count}")
        if positions:
            pos = np.asarray(positions)
            lines.append(f"Distinct positions (rounded to 0.0001 mm): {len(np.unique(np.round(pos, 4), axis=0))}")
            lines.append(f"Position bounds LPS mm: min={pos.min(axis=0).tolist()}, max={pos.max(axis=0).tolist()}")
            if len(orientations) == len(positions) == frame_count and np.allclose(orientations, orientations[0], atol=1e-5, rtol=0):
                normal = np.cross(orientations[0][:3], orientations[0][3:])
                projected = pos @ normal
                lines.append(f"Slice normal LPS: {normal.tolist()}")
                lines.append(f"Projected positions in reported order (mm): {projected.tolist()}")
                lines.append(f"Signed position steps in reported order (mm): {np.diff(projected).tolist()}")
            else:
                lines.append("Single slice-normal spacing summary unavailable: missing or varying geometry.")
    lines += ["", "=== SKIPPED FILES ===", *(skipped or ["none"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One study directory, or a DICOM file whose parent will be scanned recursively")
    parser.add_argument("--output", type=Path, help="New TXT path; default: <directory>_t1_series.txt in current directory")
    parser.add_argument("--study-id", help="Study ID for displaying Harvey's hard-coded exception; default: directory name")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    directory = args.input.parent if args.input.is_file() else args.input
    output = args.output or Path(f"{directory.resolve().name}_t1_series.txt")
    if output.exists():
        parser.error(f"Output already exists; choose another --output: {output}")
    report = build_report(args.input, args.study_id)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(f"Report saved: {output.resolve()}")


if __name__ == "__main__":
    main()
