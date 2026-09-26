"""Check whether the original T1/T2 selection uses the same DICOM files per study."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from summarize_dicom_t1t2 import inspect_study


def compare_pair(pair):
    """Compare directory entries within one resolved study, not image content."""
    selected = {row["modality"]: row for row in pair}
    t1, t2 = selected["T1"], selected["T2"]
    assessed = all(row["status"] == "selected" for row in (t1, t2))
    names1, names2 = t1.get("file_names", []), t2.get("file_names", [])
    files1, files2 = set(names1), set(names2)
    shared = [name for name in names1 if name in files2] if assessed else []
    status = "unknown"
    if assessed:
        status = "same_files" if files1 == files2 else ("partial_overlap" if shared else "disjoint")
    result = dict(Study_ID=t1["study_id"], study_directory=t1.get("study_directory", "unknown"),
                  comparison_status=status, same_file_set=(files1 == files2) if assessed else "unknown",
                  same_file_order=(names1 == names2) if assessed else "unknown",
                  shared_file_count=len(shared) if assessed else "unknown",
                  shared_file_names=shared)
    for modality, row in (("T1", t1), ("T2", t2)):
        for key in ("status", "series_number", "series_description", "candidate_group_count",
                    "file_count", "series_instance_uids", "file_names", "flags", "error"):
            result[f"{modality}_{key}"] = row.get(key, "" if key == "error" else "unknown")
    return result


def make_summary(rows, csv_path, folder_list, study_id_column):
    counts = Counter(row["comparison_status"] for row in rows)
    total = len(rows)
    assessed = total - counts["unknown"]
    assessed_percent = f"{100*counts['same_files']/assessed:.2f}%" if assessed else "unknown"
    lines = ["DICOM T1/T2 SELECTED FILE OVERLAP CHECK",
             f"Study CSV: {csv_path}", f"Study ID column: {study_id_column}", f"DICOM root list: {folder_list}",
             f"CSV studies: {total}", f"Both modalities selected: {assessed}",
             f"Same file set: {counts['same_files']} ({100*counts['same_files']/total:.2f}% of all studies; {assessed_percent} of assessable studies)",
             f"Partial overlap: {counts['partial_overlap']}", f"Disjoint file sets: {counts['disjoint']}",
             f"Unknown: {counts['unknown']}",
             f"Same set and same order: {sum(row['same_file_set'] is True and row['same_file_order'] is True for row in rows)}",
             "", "Selection calls summarize_dicom_t1t2.inspect_study without changing its rules:",
             "Direct children; group by (SeriesNumber, SeriesDescription); case-insensitive T1/T2 substrings;",
             "maximum SeriesNumber after the original exception rules; ties follow sorted group order.",
             "All CSV studies are checked, including repeated-plane and low-coverage studies.",
             "No clinical, coverage or geometry filter removes a study from this report.",
             "A missing/ambiguous directory, unreadable child or absent modality stays UNKNOWN, not disjoint.",
             "Files are compared by filename within the same resolved study directory, including list order.",
             "This tests reuse of source files, not equal pixels or equal geometry in different source files.",
             "The current grouping partitions files: normally selected sets are identical or disjoint.",
             "Partial overlap would be unexpected under that grouping and warrants checking the selection implementation.",
             "Only DICOM headers are read; no image conversion or pixel decoding is performed.",
             "", "Outputs:",
             "study_file_overlap.csv: every CSV-listed study, including unknowns.",
             "overlapping_studies.csv: same_files or partial_overlap, with selection metadata and filenames.",
             "unknown_studies.csv: studies that could not be compared.",
             "List fields are JSON; source paths and Study IDs are retained for internal investigation."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Target labels.csv; all studies are checked")
    parser.add_argument("--folder-list", default=Path("folder_address.txt"), type=Path,
                        help="One DICOM root per line; default: ./folder_address.txt")
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory outside DICOM roots")
    args = parser.parse_args()
    # Match the cohort program's input validation and relative-root interpretation.
    table = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.study_id_column not in table:
        parser.error(f"Missing CSV column: {args.study_id_column}")
    ids = table[args.study_id_column].str.strip().tolist()
    if not ids or any(not value or value in (".", "..") or any(c in value for c in '/\\:*?"<>|') for value in ids):
        parser.error("Study IDs must be nonempty plain directory names; CSV must not be empty")
    if len(set(ids)) != len(ids):
        parser.error("Duplicate Study IDs in CSV; resolve duplicates")
    roots = []
    for line in args.folder_list.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            path = Path(line.strip()).expanduser()
            roots.append((path if path.is_absolute() else args.folder_list.resolve().parent / path).resolve())
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        parser.error("DICOM roots must be existing, unique directories; relative paths resolve against the list file")
    output = args.output_dir.resolve()
    if any(output == root or root in output.parents for root in roots):
        parser.error("Output directory must be outside DICOM roots")
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    output.mkdir(parents=True)
    rows = []
    for index, study_id in enumerate(ids, 1):
        result = compare_pair(inspect_study(study_id, roots))
        rows.append(result)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}: {result['comparison_status']}", flush=True)
    table = pd.DataFrame(rows)
    for column in table:
        table[column] = table[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
    for name, data in (("study_file_overlap", table),
                       ("overlapping_studies", table[table.comparison_status.isin(["same_files", "partial_overlap"])]),
                       ("unknown_studies", table[table.comparison_status.eq("unknown")])):
        data.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig", na_rep="unknown", float_format="%.9g")
    summary = make_summary(rows, args.csv.resolve(), args.folder_list.resolve(), args.study_id_column)
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
