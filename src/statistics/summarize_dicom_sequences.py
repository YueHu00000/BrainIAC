"""Count all MR series per study and study coverage per SeriesDescription."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


STUDY_FIELDS = ["study_id", "status", "series_count", "sequence_type_count",
                "sequence_types", "mr_file_count", "non_mr_file_count", "issue_count"]
SERIES_FIELDS = ["study_id", "study_uid", "series_uid", "series_numbers",
                 "series_descriptions", "file_count"]
TYPE_FIELDS = ["sequence_description", "study_count", "study_percent", "series_count"]
ISSUE_FIELDS = ["study_id", "file", "reason"]


def inspect_study(study_id, roots):
    row = dict(study_id=study_id, status="complete", series_count="unknown",
               sequence_type_count="unknown", sequence_types=[], mr_file_count=0,
               non_mr_file_count=0, issue_count=0)
    matches = [root / study_id for root in roots if (root / study_id).is_dir()]
    if len(matches) != 1:
        row["status"] = "study_not_found" if not matches else "study_ambiguous"
        row["issue_count"] = 1
        return row, [], [dict(study_id=study_id, file="", reason=row["status"])]
    groups, issues = {}, []
    for path in sorted(p for p in matches[0].rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as error:
            issues.append(dict(study_id=study_id, file=str(path),
                               reason=f"read_error: {type(error).__name__}: {error}"))
            continue
        modality = str(ds.get("Modality", "")).strip()
        if not modality:
            issues.append(dict(study_id=study_id, file=str(path), reason="missing_modality"))
            continue
        if modality != "MR":
            row["non_mr_file_count"] += 1
            continue
        row["mr_file_count"] += 1
        study_uid = str(ds.get("StudyInstanceUID", "")).strip()
        series_uid = str(ds.get("SeriesInstanceUID", "")).strip()
        if not study_uid or not series_uid:
            issues.append(dict(study_id=study_id, file=str(path), reason="missing_study_or_series_uid"))
            continue
        key = (study_uid, series_uid)
        group = groups.setdefault(key, dict(numbers=set(), descriptions=set(), file_count=0))
        group["numbers"].add(str(ds.get("SeriesNumber", "")).strip())
        group["descriptions"].add(str(ds.get("SeriesDescription", "")).strip())
        group["file_count"] += 1
    series = []
    for (study_uid, series_uid), group in sorted(groups.items()):
        descriptions = sorted(group["descriptions"])
        series.append(dict(study_id=study_id, study_uid=study_uid, series_uid=series_uid,
                           series_numbers=sorted(group["numbers"]),
                           series_descriptions=descriptions, file_count=group["file_count"]))
        if len(descriptions) != 1 or not descriptions[0]:
            issues.append(dict(study_id=study_id, file="",
                               reason=f"missing_or_inconsistent_description: {series_uid}"))
    row["issue_count"] = len(issues)
    row["status"] = "incomplete" if issues else ("complete" if series else "no_mr_series")
    # UID counts remain usable if only descriptions are missing/inconsistent.
    if not any(not issue["reason"].startswith("missing_or_inconsistent_description:") for issue in issues):
        row["series_count"] = len(series)
    if row["status"] == "complete":
        row["sequence_types"] = sorted({s["series_descriptions"][0] for s in series})
        row["sequence_type_count"] = len(row["sequence_types"])
    elif row["status"] == "no_mr_series":
        row["sequence_type_count"] = 0
    return row, series, issues


def summarize_types(studies, series):
    complete = {row["study_id"] for row in studies if row["status"] == "complete"}
    coverage, counts = defaultdict(set), Counter()
    for row in series:
        if row["study_id"] in complete:
            description = row["series_descriptions"][0]
            coverage[description].add(row["study_id"])
            counts[description] += 1
    return [dict(sequence_description=name, study_count=len(coverage[name]),
                 study_percent=100 * len(coverage[name]) / len(complete), series_count=counts[name])
            for name in sorted(coverage, key=lambda name: (-len(coverage[name]), name))]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list)
                             else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--folder-list", required=True, type=Path)
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if args.study_id_column not in (reader.fieldnames or []):
            parser.error(f"Missing CSV column: {args.study_id_column}")
        ids = [row[args.study_id_column].strip() for row in reader]
    if not ids or any(not value or value in (".", "..") or any(c in value for c in '/\\:*?"<>|') for value in ids):
        parser.error("Study IDs must be nonempty plain directory names")
    if len(ids) != len(set(ids)):
        parser.error("Duplicate Study IDs in CSV")
    roots = []
    for line in args.folder_list.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            path = Path(line.strip()).expanduser()
            roots.append((path if path.is_absolute() else args.folder_list.resolve().parent / path).resolve())
    if not roots or len(roots) != len(set(roots)) or any(not root.is_dir() for root in roots):
        parser.error("DICOM roots must be existing, unique directories")
    output = args.output_dir.resolve()
    if output.exists() or any(output == root or root in output.parents for root in roots):
        parser.error("Output must be a new directory outside DICOM roots")
    output.mkdir(parents=True)
    studies, series, issues = [], [], []
    for index, study_id in enumerate(ids, 1):
        row, detail, errors = inspect_study(study_id, roots)
        studies.append(row)
        series.extend(detail)
        issues.extend(errors)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}: {row['status']}, series={row['series_count']}", flush=True)
    types = summarize_types(studies, series)
    write_csv(output / "study_sequence_counts.csv", studies, STUDY_FIELDS)
    write_csv(output / "all_mr_series.csv", series, SERIES_FIELDS)
    write_csv(output / "sequence_study_counts.csv", types, TYPE_FIELDS)
    write_csv(output / "scan_issues.csv", issues, ISSUE_FIELDS)
    distribution = Counter(row["series_count"] for row in studies if row["series_count"] != "unknown")
    write_csv(output / "series_count_distribution.csv",
              [dict(series_count=count, study_count=distribution[count]) for count in sorted(distribution)],
              ["series_count", "study_count"])
    lines = ["ALL MR SERIES INVENTORY", f"Study CSV: {args.csv.resolve()}",
             f"DICOM root list: {args.folder_list.resolve()}", f"CSV studies: {len(ids)}",
             f"Statuses: {dict(Counter(row['status'] for row in studies))}",
             "Series identity: (StudyInstanceUID, SeriesInstanceUID), within each CSV study.",
             "Sequence type: exact SeriesDescription after trimming outer whitespace; no modality inference.",
             "All files are scanned recursively, headers only; only Modality=MR is counted.",
             "Localizers, derived series and repeated acquisitions are included; no T1/T2 selection/exceptions.",
             "Type coverage counts each study once per description, using complete studies only.",
             "study_percent denominator: complete studies with MR series.",
             "Unknown counts are not zero; issues and readable partial series remain in separate CSVs.",
             f"Sequence descriptions in complete studies: {len(types)}", "", "Series-count distribution:"]
    lines.extend(f"{count} series: {distribution[count]} studies" for count in sorted(distribution))
    lines.append("\nSequence coverage: description | studies | series")
    lines.extend(f"{row['sequence_description']} | {row['study_count']} | {row['series_count']}" for row in types)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished. Reports: {output}")


if __name__ == "__main__":
    main()
