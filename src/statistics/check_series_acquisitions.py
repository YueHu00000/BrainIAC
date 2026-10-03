"""Inspect SeriesNumber/AcquisitionNumber grouping without reading pixels."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pydicom

from summarize_dicom_sequences import write_csv


KNOWN_STUDY = "R01_Study_002904"
EXCEPTIONS = {"STUDY_0230": 8, "STUDY_0462": 9, "STUDY_0836": 9, "STUDY_1109": 4}
FILE_FIELDS = ["study_id", "known_case", "file", "study_uid", "series_uid", "modality",
               "series_number", "series_description", "acquisition_number", "instance_number",
               "number_of_frames", "echo_numbers", "temporal_position", "image_position",
               "image_orientation", "selected_t1t2", "selection_status"]
GROUP_FIELDS = ["study_id", "known_case", "series_number", "file_count", "acquisition_count",
                "acquisition_numbers", "files_per_acquisition", "missing_acquisition_files",
                "series_identity_count", "missing_uid_files", "series_uids", "series_descriptions",
                "uid_with_multiple_acquisitions", "selected_t1t2", "selected_t1_acquisition_numbers",
                "selected_t2_acquisition_numbers", "selected_multiple_acquisition_modalities", "selection_status", "scan_status"]
PAIR_FIELDS = ["study_id", "known_case", "series_number", "acquisition_number", "file_count",
               "series_identity_count", "series_uids", "series_descriptions"]
STUDY_FIELDS = ["study_id", "known_case", "status", "dicom_file_count", "series_number_count",
                "multiple_acquisition_group_count", "pair_collision_count", "selection_status", "issue_count"]
ISSUE_FIELDS = ["study_id", "file", "reason"]


def tag(ds, name):
    value = ds.get(name, "")
    return str(value).strip() if value is not None else ""


def inspect_study(study_id, roots):
    known = study_id == KNOWN_STUDY
    study = dict(study_id=study_id, known_case=known, status="complete", dicom_file_count=0,
                 series_number_count=0, multiple_acquisition_group_count=0,
                 pair_collision_count=0, selection_status="not_evaluable", issue_count=0)
    matches = [root / study_id for root in roots if (root / study_id).is_dir()]
    if len(matches) != 1:
        study["status"] = "study_not_found" if not matches else "study_ambiguous"
        study["issue_count"] = 1
        return study, [], [], [], [dict(study_id=study_id, file="", reason=study["status"])]
    directory = matches[0]
    records, issues = [], []
    selection_valid = True
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        direct = path.parent == directory
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as error:
            issues.append(dict(study_id=study_id, file=str(path), reason=f"read_error: {error}"))
            if direct:
                selection_valid = False
            continue
        study["dicom_file_count"] += 1
        row = dict(study_id=study_id, known_case=known, file=str(path),
                   study_uid=tag(ds, "StudyInstanceUID"), series_uid=tag(ds, "SeriesInstanceUID"),
                   modality=tag(ds, "Modality"), series_number=tag(ds, "SeriesNumber"),
                   series_description=tag(ds, "SeriesDescription"),
                   acquisition_number=tag(ds, "AcquisitionNumber"), instance_number=tag(ds, "InstanceNumber"),
                   number_of_frames=tag(ds, "NumberOfFrames") or "1", echo_numbers=tag(ds, "EchoNumbers"),
                   temporal_position=tag(ds, "TemporalPositionIdentifier"),
                   image_position=tag(ds, "ImagePositionPatient"), image_orientation=tag(ds, "ImageOrientationPatient"),
                   selected_t1t2=[], selection_status="not_evaluable", direct=direct)
        for field in ("series_number", "acquisition_number", "instance_number"):
            if row[field]:
                try:
                    row[field] = str(int(row[field]))
                except ValueError:
                    issues.append(dict(study_id=study_id, file=str(path), reason=f"invalid_{field}"))
                    row[field] = ""
        missing = [field for field in ("series_number", "acquisition_number", "study_uid", "series_uid") if not row[field]]
        if missing:
            issues.append(dict(study_id=study_id, file=str(path), reason="missing: " + ",".join(missing)))
        if row["number_of_frames"] != "1":
            issues.append(dict(study_id=study_id, file=str(path), reason="multiframe: per-frame acquisition not audited"))
        if direct and (not row["series_number"] or not row["series_description"] or not row["instance_number"]):
            selection_valid = False
        records.append(row)
    # Reconstruct the old direct-child (SeriesNumber, SeriesDescription) selection.
    selected = {}
    selection_states = []
    if selection_valid:
        candidates = {(int(r["series_number"]), r["series_description"]) for r in records
                      if r["direct"] and int(r["series_number"]) != EXCEPTIONS.get(study_id)}
        for modality in ("T1", "T2"):
            options = sorted(key for key in candidates if modality in key[1].upper())
            if not options:
                selection_states.append(f"{modality}=no_candidate")
                continue
            winners = [key for key in options if key[0] == max(option[0] for option in options)]
            selected[modality] = winners[0]
            selection_states.append(f"{modality}=" + ("max_number_tie" if len(winners) > 1 else "selected"))
        study["selection_status"] = ";".join(selection_states)
    grouped = defaultdict(list)
    for row in records:
        row["selected_t1t2"] = [m for m, key in selected.items() if row["direct"] and
                                (int(row["series_number"]), row["series_description"]) == key]
        row["selection_status"] = study["selection_status"]
        if row["series_number"]:
            grouped[row["series_number"]].append(row)
    groups, collisions, hit_files = [], [], []
    scan_status = "incomplete" if issues else ("complete" if records else "no_dicom_files")
    for number, files in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        acquisitions = sorted({r["acquisition_number"] for r in files if r["acquisition_number"]}, key=int)
        identities = {(r["study_uid"], r["series_uid"]) for r in files if r["study_uid"] and r["series_uid"]}
        by_uid = defaultdict(set)
        for r in files:
            if r["study_uid"] and r["series_uid"] and r["acquisition_number"]:
                by_uid[(r["study_uid"], r["series_uid"])].add(r["acquisition_number"])
        selected_acquisitions = {m: sorted({r["acquisition_number"] for r in files
                                           if m in r["selected_t1t2"] and r["acquisition_number"]}, key=int)
                                 for m in ("T1", "T2")}
        groups.append(dict(study_id=study_id, known_case=known, series_number=number, file_count=len(files),
                           acquisition_count=len(acquisitions), acquisition_numbers=acquisitions,
                           files_per_acquisition=[dict(acquisition_number=a, file_count=sum(r["acquisition_number"] == a for r in files)) for a in acquisitions],
                           missing_acquisition_files=sum(not r["acquisition_number"] for r in files),
                           series_identity_count=len(identities), missing_uid_files=sum(not r["study_uid"] or not r["series_uid"] for r in files),
                           series_uids=sorted({r["series_uid"] for r in files if r["series_uid"]}),
                           series_descriptions=sorted({r["series_description"] for r in files}),
                           uid_with_multiple_acquisitions=sum(len(values) > 1 for values in by_uid.values()),
                           selected_t1t2=sorted({m for r in files for m in r["selected_t1t2"]}),
                           selected_t1_acquisition_numbers=selected_acquisitions["T1"],
                           selected_t2_acquisition_numbers=selected_acquisitions["T2"],
                           selected_multiple_acquisition_modalities=[m for m, values in selected_acquisitions.items() if len(values) > 1],
                           selection_status=study["selection_status"], scan_status=scan_status))
        if len(acquisitions) > 1:
            hit_files.extend({key: r[key] for key in FILE_FIELDS} for r in files)
        for acquisition in acquisitions:
            members = [r for r in files if r["acquisition_number"] == acquisition]
            pair_ids = {(r["study_uid"], r["series_uid"]) for r in members if r["study_uid"] and r["series_uid"]}
            if len(pair_ids) > 1:
                collisions.append(dict(study_id=study_id, known_case=known, series_number=number,
                                       acquisition_number=acquisition, file_count=len(members),
                                       series_identity_count=len(pair_ids), series_uids=sorted({uid for _, uid in pair_ids}),
                                       series_descriptions=sorted({r["series_description"] for r in members})))
    study.update(status=scan_status, series_number_count=len(groups), issue_count=len(issues),
                 multiple_acquisition_group_count=sum(g["acquisition_count"] > 1 for g in groups),
                 pair_collision_count=len(collisions))
    return study, groups, collisions, hit_files, issues


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
            parser.error(f"Missing column: {args.study_id_column}")
        ids = [(row[args.study_id_column] or "").strip() for row in reader]
    if not ids or any(not x or x in (".", "..") or any(c in x for c in '/\\:*?"<>|') for x in ids) or len(set(ids)) != len(ids):
        parser.error("Study IDs must be unique, nonempty plain directory names")
    roots = []
    for line in args.folder_list.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            path = Path(line.strip()).expanduser()
            roots.append((path if path.is_absolute() else args.folder_list.resolve().parent / path).resolve())
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        parser.error("Folder list must contain existing, unique roots")
    output = args.output_dir.resolve()
    if output.exists() or any(output == root or root in output.parents for root in roots):
        parser.error("Output must be a new directory outside DICOM roots")
    output.mkdir(parents=True)
    studies, groups, collisions, hit_files, issues = [], [], [], [], []
    for index, study_id in enumerate(ids, 1):
        study, detail, pairs, files, errors = inspect_study(study_id, roots)
        studies.append(study)
        groups.extend(detail)
        collisions.extend(pairs)
        hit_files.extend(files)
        issues.extend(errors)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}: {study['status']}, multi-acquisition groups={study['multiple_acquisition_group_count']}", flush=True)
    hits = [g for g in groups if g["acquisition_count"] > 1]
    other_hits = [g for g in hits if not g["known_case"]]
    outputs = [("study_summary.csv", studies, STUDY_FIELDS), ("all_series_number_groups.csv", groups, GROUP_FIELDS),
               ("multiple_acquisition_groups.csv", other_hits, GROUP_FIELDS),
               ("known_case_groups.csv", [g for g in groups if g["known_case"]], GROUP_FIELDS),
               ("multiple_acquisition_files.csv", hit_files, FILE_FIELDS),
               ("pair_collisions.csv", collisions, PAIR_FIELDS), ("scan_issues.csv", issues, ISSUE_FIELDS)]
    for filename, rows, fields in outputs:
        write_csv(output / filename, rows, fields)
    lines = ["SERIES NUMBER / ACQUISITION NUMBER AUDIT", f"Study CSV: {args.csv.resolve()}",
             f"Folder list: {args.folder_list.resolve()}", f"CSV studies: {len(studies)}",
             f"Statuses: {dict(Counter(s['status'] for s in studies))}",
             "Scope: recursively read every DICOM header, all modalities; never decode pixels.",
             "Grouping: within each CSV study by SeriesNumber; missing AcquisitionNumber is unknown, not zero.",
             f"Known study (separate): {KNOWN_STUDY}; listed in CSV: {KNOWN_STUDY in ids}",
             f"Other studies with observed multiple acquisitions: {len({g['study_id'] for g in other_hits})}",
             f"Other SeriesNumber groups with observed multiple acquisitions: {len(other_hits)}",
             f"Other selected T1/T2 groups with multiple acquisitions: {sum(len(g['selected_multiple_acquisition_modalities']) for g in other_hits)}",
             f"Known-study multiple-acquisition groups: {sum(g['known_case'] for g in hits)}",
             f"Observed (SeriesNumber, AcquisitionNumber) UID collisions: {len(collisions)}",
             f"Scan issues: {len(issues)}",
             "Old T1/T2 selection uses direct-child files, (number, description), maximum number and four exceptions.",
             "Max-number ties are flagged; first sorted description is shown to match pandas group order.",
             "Nested files are audited, but not silently included in the old direct-child selection.",
             "Incomplete studies can contain definite hits; no-hit does not certify absence when fields/files are missing.",
             "No UID collision does NOT prove a correct volume or unique global name: geometry/echo/time/frame checks remain necessary.",
             "Files are DICOM objects, not necessarily slices; Enhanced MR per-frame acquisitions are not inspected.",
             "Multiple acquisitions are observations, not automatic split/merge decisions.", "", "OTHER HIT GROUPS:"]
    lines.extend(f"{g['study_id']} / SeriesNumber={g['series_number']} / acquisitions={g['acquisition_numbers']} / selected={g['selected_t1t2']}" for g in other_hits)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished. Reports: {output}")


if __name__ == "__main__":
    main()
