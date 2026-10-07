"""Build a non-projection MR manifest, selecting the largest acquisition per UID/number."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import read_rows, validate_id, write_rows
from brainIAC_pretraining._dicom import inspect_study


FIELDS = ["unique_id", "study_id", "series_instance_uid", "series_number", "study_directory",
          "acquisition_number", "file_names"]


def read_inputs(csv_path, folder_list, study_id_column="Study_ID"):
    labels = read_rows(csv_path)
    ids = [row[study_id_column].strip() for row in labels]
    for study_id in ids:
        validate_id(study_id)
    if len(set(ids)) != len(ids):
        raise ValueError("Study IDs must be unique")
    folder_list = Path(folder_list).resolve()
    roots = []
    for line in folder_list.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            root = Path(line.strip()).expanduser()
            roots.append((root if root.is_absolute() else folder_list.parent / root).resolve())
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        raise ValueError("Folder list must contain existing, unique roots")
    return ids, roots


def build_manifest(ids, roots):
    rows, issues = [], []
    for index, study_id in enumerate(ids, 1):
        selected, study, errors = inspect_study(study_id, roots)
        for selected_row in selected:
            validate_id(selected_row["unique_id"])
            row = {field: selected_row[field] for field in FIELDS}
            row["file_names"] = json.dumps(row["file_names"], ensure_ascii=False)
            rows.append(row)
        issues.extend(errors)
        if not selected and not errors:
            issues.append(dict(study_id=study_id, file="", reason="no_mr_series"))
        print(f"[{index}/{len(ids)}] {study_id}: {study['status']}, series={len(selected)}", flush=True)
    return rows, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--folder-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study-id-column", default="Study_ID")
    args = parser.parse_args()
    ids, roots = read_inputs(args.csv, args.folder_list, args.study_id_column)
    output = args.output_dir.resolve()
    if any(output == root or root in output.parents for root in roots):
        raise ValueError("Manifest output must be outside DICOM roots")
    rows, issues = build_manifest(ids, roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "manifest.csv", rows, FIELDS)
    write_rows(args.output_dir / "manifest_issues.csv", issues, ["study_id", "file", "reason"])
    print(f"Manifest: {len(rows)} series; scan issues: {len(issues)}")


if __name__ == "__main__":
    main()
