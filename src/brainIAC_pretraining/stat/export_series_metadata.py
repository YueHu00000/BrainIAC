"""Export all MR series metadata directly from DICOM, keeping maximum acquisition."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pydicom

from _series_statistics import geometry, summarize_series


FIELDS = ("SeriesDescription", "ImageType", "SliceThickness", "SpacingBetweenSlices")
META_COLUMNS = ["unique_id", "study_id", "SeriesInstanceUID", "SeriesNumber",
                "SeriesDescription", "AcquisitionNumber", "ImageType", "SliceThickness",
                "SpacingBetweenSlices", "dicom_file_list", "coverage_mm"]
MULTIVALUE_COLUMNS = ["unique_id", "multiple_fields"]
ISSUE_COLUMNS = ["study_id", "unique_id", "file", "reason"]


def integer(value):
    return int(value) if value is not None and str(value).strip() else None


def metadata_values(ds):
    """Preserve whole ImageType combinations and shared/per-frame pixel measures."""
    description = str(ds.get("SeriesDescription") or "").strip()
    image_type = ds.get("ImageType")
    if image_type is not None and not isinstance(image_type, str):
        image_type = "\\".join(str(part).strip() for part in image_type)
    values = {"SeriesDescription": [description], "ImageType": [str(image_type or "").strip()],
              "SliceThickness": [], "SpacingBetweenSlices": []}
    shared = list(ds.get("SharedFunctionalGroupsSequence", [])[:1])
    for frame in ds.get("PerFrameFunctionalGroupsSequence", []) or [None]:
        values["SliceThickness"].append(geometry(ds, frame)["SliceThickness"])
        spacing = ds.get("SpacingBetweenSlices")
        for group in shared + ([frame] if frame is not None else []):
            measures = group.get("PixelMeasuresSequence", [])
            if measures:
                spacing = measures[0].get("SpacingBetweenSlices", spacing)
        values["SpacingBetweenSlices"].append(spacing)
    for field in ("SliceThickness", "SpacingBetweenSlices"):
        for index, value in enumerate(values[field]):
            try:
                number = float(value)
                values[field][index] = number if math.isfinite(number) else None
            except (TypeError, ValueError):
                values[field][index] = None
    return values


def inspect_study(study_id, roots):
    issues, groups = [], defaultdict(list)

    def report(path, reason, unique_id=""):
        issues.append(dict(study_id=study_id, unique_id=unique_id, file=str(path), reason=reason))

    matches = [root / study_id for root in roots if (root / study_id).is_dir()]
    if len(matches) != 1:
        report("", "study_not_found" if not matches else "study_ambiguous")
        return [], [], issues
    directory = matches[0]
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(ds.get("Modality", "")).upper() != "MR":
                continue
            uid = str(ds.get("SeriesInstanceUID") or "").strip()
            if not uid:
                raise ValueError("Missing SeriesInstanceUID")
            number = integer(ds.get("SeriesNumber"))
            if number is None:
                raise ValueError("Missing SeriesNumber")
        except Exception as error:
            report(path, f"{type(error).__name__}: {error}")
            continue
        unique_id = f"{study_id}_{uid}_{number}"
        try:
            acquisition = integer(ds.get("AcquisitionNumber"))
        except (TypeError, ValueError):
            acquisition = None
        if acquisition is None:
            report(path, "Missing or invalid AcquisitionNumber", unique_id)
        try:
            instance = integer(ds.get("InstanceNumber"))
        except (TypeError, ValueError):
            instance = None
        if instance is None:
            ds.InstanceNumber = None
            report(path, "Missing or invalid InstanceNumber", unique_id)
        groups[(uid, number)].append((path, ds, acquisition, instance))

    rows, multivalue_rows = [], []
    for (uid, number), items in sorted(groups.items()):
        unique_id = f"{study_id}_{uid}_{number}"
        acquisitions = [a for _, _, a, _ in items if a is not None]
        chosen = max(acquisitions) if acquisitions else None
        selected = [(p, ds, instance) for p, ds, a, instance in items if a == chosen]
        selected.sort(key=lambda item: (item[2] if item[2] is not None else 0, str(item[0])))
        values = {field: set() for field in FIELDS}
        for path, ds, _ in selected:
            missing = []
            for field, entries in metadata_values(ds).items():
                values[field].update(v for v in entries if v is not None and v != "")
                if any(v is None or v == "" for v in entries):
                    missing.append(field)
            if missing:
                report(path, "Missing or invalid metadata: " + ", ".join(missing), unique_id)
            if ds.get("ImageType") is None:
                ds.ImageType = []
        row = dict(unique_id=unique_id, study_id=study_id, SeriesInstanceUID=uid, SeriesNumber=number,
                   AcquisitionNumber=chosen if chosen is not None else "unknown",
                   dicom_file_list=[p.stem for p, _, _ in selected])
        for field, entries in values.items():
            distinct = sorted(entries)
            row[field] = distinct if len(distinct) > 1 else (distinct[0] if distinct else "unknown")
        statistics = summarize_series([(p, ds) for p, ds, _ in selected])
        row["coverage_mm"] = statistics["coverage_mm"]
        if not math.isfinite(row["coverage_mm"]):
            report("", "Unknown coverage_mm: " + ", ".join(statistics["flags"]), unique_id)
        rows.append(row)
        multiple = [field for field in FIELDS if len(values[field]) > 1]
        if multiple:
            multivalue_rows.append(dict(unique_id=unique_id, multiple_fields=multiple))
    return rows, multivalue_rows, issues


def write_csv(rows, path, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list)
                             else ("unknown" if isinstance(value, float) and not math.isfinite(value) else value)
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Study ID CSV, not a statistics report")
    parser.add_argument("--folder-list", required=True, type=Path)
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if args.study_id_column not in (reader.fieldnames or []):
            parser.error(f"Missing column: {args.study_id_column}")
        ids = [(row[args.study_id_column] or "").strip() for row in reader]
    if not ids or len(set(ids)) != len(ids) or any(not s or s in (".", "..") or any(c in s for c in '/\\:*?"<>|') for s in ids):
        parser.error("Study IDs must be unique, nonempty directory names")
    roots = []
    for line in args.folder_list.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            root = Path(line.strip()).expanduser()
            roots.append((root if root.is_absolute() else args.folder_list.resolve().parent / root).resolve())
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        parser.error("Folder list must contain existing, unique roots")
    output = args.output_dir.resolve()
    if output.exists() or any(output == root or root in output.parents for root in roots):
        parser.error("Output must be a new directory outside DICOM roots")
    rows, multivalue, issues = [], [], []
    for index, study_id in enumerate(ids, 1):
        detail, multiple, errors = inspect_study(study_id, roots)
        rows.extend(detail)
        multivalue.extend(multiple)
        issues.extend(errors)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}: series={len(detail)}, issues={len(errors)}", flush=True)
    output.mkdir(parents=True)
    write_csv(rows, output / "series_meta.csv", META_COLUMNS)
    write_csv(multivalue, output / "series_multivalue.csv", MULTIVALUE_COLUMNS)
    write_csv(issues, output / "scan_issues.csv", ISSUE_COLUMNS)
    print(f"Finished: {len(rows)} unique series; {len(multivalue)} with multiple values. Reports: {output}")


if __name__ == "__main__":
    main()
