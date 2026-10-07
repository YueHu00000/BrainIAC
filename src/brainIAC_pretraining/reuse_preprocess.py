"""Match old paired T1/T2 preprocess files and copy them to UID-based filenames."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import read_rows, validate_id, write_rows


FIELDS = ["study_id", "modality", "old_file", "unique_id", "acquisition_number", "status", "reason", "action", "error"]


def file_list(value):
    names = json.loads(value)
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Missing or invalid file_names")
    return [str(PurePosixPath(name.replace("\\", "/"))) for name in names]


def audit_reuse(old_selected, manifest, old_preprocess_dir):
    """Compare recorded selections; only check existence of processed images."""
    current = {}
    for row in manifest:
        try:
            uid = row["series_instance_uid"]
            expected = f"{row['study_id']}_{uid}_{int(row['series_number'])}"
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Use the SeriesInstanceUID manifest with study_id, series_instance_uid and series_number") from error
        if not uid or uid == "unknown" or row["unique_id"] != expected:
            raise ValueError("Manifest unique_id must equal {Study_ID}_{SeriesInstanceUID}_{SeriesNumber}")
        validate_id(expected)
        if expected in current:
            raise ValueError(f"Duplicate manifest unique_id: {expected}")
        current[expected] = row

    reports, seen = [], set()
    for old in old_selected:
        study, modality = old["study_id"], old["modality"]
        validate_id(study)
        if modality not in ("T1", "T2"):
            raise ValueError("Use the original T1/T2 selected_series.csv, with modality T1 or T2")
        key = (study, modality)
        if key in seen:
            raise ValueError(f"Duplicate old study/modality: {study}/{modality}")
        seen.add(key)
        source = Path(old_preprocess_dir) / study / f"{modality}.nii.gz"
        report = dict(study_id=study, modality=modality, old_file=str(source), unique_id="",
                      acquisition_number="", status="abandoned", reason="")
        reports.append(report)
        if old["status"] != "selected":
            report["reason"] = "old_selection_not_selected"
            continue
        try:
            uids = json.loads(old["series_instance_uids"])
            flags = json.loads(old.get("flags") or "[]")
            if not isinstance(uids, list) or not isinstance(flags, list):
                raise ValueError("Invalid UID/flags list")
            if (not uids or any(not isinstance(uid, str) or not uid.strip() or uid == "unknown" for uid in uids)
                    or "missing_SeriesInstanceUID" in flags):
                report["reason"] = "missing_series_instance_uid"
                continue
            if len(uids) != 1:
                report["reason"] = "multiple_series_instance_uids"
                continue
            unique_id = f"{study}_{uids[0]}_{int(old['series_number'])}"
            report["unique_id"] = unique_id
            target = current.get(unique_id)
            if target is None:
                report["reason"] = "not_in_manifest"
                continue
            report["acquisition_number"] = target["acquisition_number"]
            if not old["study_directory"] or not target["study_directory"]:
                raise ValueError("Missing source directory")
            if Path(old["study_directory"].replace("\\", "/")).resolve() != Path(target["study_directory"].replace("\\", "/")).resolve():
                report["reason"] = "study_directory_mismatch"
                continue
            previous_files, selected_files = file_list(old["file_names"]), file_list(target["file_names"])
            if sorted(previous_files) != sorted(selected_files):
                report["reason"] = "selected_files_mismatch"
                continue
            if previous_files != selected_files:
                report["reason"] = "file_order_mismatch"
                continue
        except (KeyError, ValueError, TypeError):
            report["reason"] = "invalid_selection_metadata"
            continue
        if not source.is_file():
            report["reason"] = "missing_preprocess_file"
            continue
        report.update(status="reusable", reason="exact_source_match")

    for source in sorted(Path(old_preprocess_dir).glob("*/T[12].nii.gz")):
        study, modality = source.parent.name, source.name[:2]
        if source.is_file() and not study.startswith(".") and (study, modality) not in seen:
            reports.append(dict(study_id=study, modality=modality, old_file=str(source), unique_id="",
                                acquisition_number="", status="abandoned", reason="missing_old_statistics"))

    # Two separately processed T1/T2 outputs must not silently compete for one ID.
    targets = Counter(row["unique_id"] for row in reports if row["status"] == "reusable")
    for row in reports:
        if row["status"] == "reusable" and targets[row["unique_id"]] > 1:
            row.update(status="abandoned", reason="multiple_old_outputs_for_unique_id")
    return reports


def reuse_preprocess(old_selected_csv, manifest_csv, old_preprocess_dir, output_dir):
    old_root, output = Path(old_preprocess_dir).resolve(), Path(output_dir).resolve()
    if not old_root.is_dir():
        raise ValueError("Old preprocess directory does not exist")
    if output == old_root or old_root in output.parents or output in old_root.parents:
        raise ValueError("Old and new preprocess directories must not overlap")
    if any(Path(path).resolve() == output / "old_name_change.csv"
           for path in (old_selected_csv, manifest_csv)):
        raise ValueError("Input CSV must not be overwritten by an output report")
    reports = audit_reuse(read_rows(old_selected_csv), read_rows(manifest_csv), old_root)
    for row in reports:
        row.update(action="abandoned" if row["status"] == "abandoned" else "pending", error="")
    output.mkdir(parents=True, exist_ok=True)
    try:
        for row in reports:
            if row["status"] != "reusable":
                print(f"[abandoned] {row['study_id']}/{row['modality']}: {row['reason']}", flush=True)
                continue
            destination = output / f"{row['unique_id']}.nii.gz"
            if destination.is_file():
                row["action"] = "skipped_existing"
                print(f"[skip] {row['unique_id']}", flush=True)
                continue
            try:
                scratch = output / ".reuse_tmp"
                scratch.mkdir(exist_ok=True)
                with TemporaryDirectory(dir=scratch) as temporary:
                    staged = Path(temporary) / "volume.nii.gz"
                    shutil.copyfile(row["old_file"], staged)
                    os.replace(staged, destination)
                row["action"] = "copied"
                print(f"[copied] {row['unique_id']}", flush=True)
            except Exception as error:
                row.update(status="abandoned", reason="copy_failed", action="copy_failed", error=str(error))
                print(f"[failed] {row['unique_id']}: {error}", flush=True)
    finally:
        write_rows(output / "old_name_change.csv", reports, FIELDS)
    reusable = [row for row in reports if row["status"] == "reusable"]
    abandoned = [row for row in reports if row["status"] == "abandoned"]
    reasons = Counter(row["reason"] for row in abandoned)
    print(f"Reuse finished: {len(reusable)} reusable; {len(abandoned)} abandoned. {output}")
    print(f"Actions: {dict(Counter(row['action'] for row in reports))}; abandoned reasons: {dict(reasons)}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-selected-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-preprocess-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Pretraining preprocess directory; existing NIfTI files are skipped")
    args = parser.parse_args()
    reuse_preprocess(args.old_selected_csv, args.manifest, args.old_preprocess_dir, args.output_dir)


if __name__ == "__main__":
    main()
