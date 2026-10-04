"""Summarize every MR Study_ID/SeriesNumber; keep the largest AcquisitionNumber."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from _series_statistics import METRICS, PERCENTILES, PERCENTILE_NAMES, save_csv, summarize_series


SERIES_COLUMNS = ["unique_id", "study_id", "series_number", "modality", "status", "flags",
                  "series_description", "acquisition_number", "acquisition_numbers",
                  "source_file_count", "discarded_file_count", "missing_acquisition_file_count", *METRICS]


def integer(ds, key):
    value = ds.get(key)
    return int(value) if value is not None and str(value).strip() else None


def inspect_study(study_id, roots, reject_multiframe=False):
    matches = [root / study_id for root in roots if (root / study_id).is_dir()]
    study = dict(study_id=study_id, status="complete", series_count=0, selected_file_count=0,
                 discarded_file_count=0)
    issues, groups = [], defaultdict(list)
    if len(matches) != 1:
        study["status"] = "study_not_found" if not matches else "study_ambiguous"
        return [], study, [dict(study_id=study_id, file="", reason=study["status"])]
    directory = matches[0]
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(ds.get("Modality", "")).upper() != "MR":
                continue
            number = integer(ds, "SeriesNumber")
            if number is None:
                raise ValueError("Missing SeriesNumber")
            acquisition = integer(ds, "AcquisitionNumber")
        except Exception as error:
            issues.append(dict(study_id=study_id, file=str(path), reason=f"{type(error).__name__}: {error}"))
            continue
        if reject_multiframe and int(ds.get("NumberOfFrames", 1)) > 1:
            raise ValueError(f"{study_id}_{number}: unsupported multi-frame DICOM: {path}")
        groups[number].append((path, ds, acquisition))
    rows = []
    for number, items in sorted(groups.items()):
        acquisitions = sorted({a for _, _, a in items if a is not None})
        chosen = max(acquisitions) if acquisitions else None
        selected = [(p, ds) for p, ds, a in items if a == chosen]
        selected.sort(key=lambda item: (integer(item[1], "InstanceNumber") or 0, str(item[0])))
        descriptions = sorted({str(ds.get("SeriesDescription", "")).strip() for _, ds in selected})
        row = dict(unique_id=f"{study_id}_{number}", study_id=study_id, series_number=number,
                   modality="MR", status="selected", study_directory=str(directory),
                   series_description=" | ".join(descriptions), series_descriptions=descriptions,
                   acquisition_number=chosen if chosen is not None else "unknown",
                   acquisition_numbers=acquisitions, source_file_count=len(items),
                   discarded_file_count=len(items) - len(selected),
                   missing_acquisition_file_count=sum(a is None for _, _, a in items))
        row.update(summarize_series(selected, include_frame_positions=True))
        # Paths are relative to the study directory, including nested acquisitions.
        row["file_names"] = [str(p.relative_to(directory)) for p, _ in selected]
        if len(acquisitions) > 1:
            row["flags"].append("largest_acquisition_selected")
        if row["missing_acquisition_file_count"]:
            row["flags"].append("missing_acquisition_number")
        if len(descriptions) > 1:
            row["flags"].append("multiple_series_descriptions")
        if any(integer(ds, "InstanceNumber") is None for _, ds in selected):
            row["flags"].append("missing_instance_number")
        if issues:
            row["flags"].append("partial_study_scan")
        rows.append(row)
    study.update(status="incomplete" if issues else ("complete" if rows else "no_mr_series"),
                 series_count=len(rows), selected_file_count=sum(r["file_count"] for r in rows),
                 discarded_file_count=sum(r["discarded_file_count"] for r in rows))
    return rows, study, issues


def make_distribution(rows):
    records = []
    for metric in METRICS:
        values = np.asarray([r.get(metric, np.nan) for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        percentiles = np.percentile(values, PERCENTILES, method="linear") if values.size else [np.nan] * len(PERCENTILES)
        records.append(dict(metric=metric, n_total=len(rows), n_valid=len(values),
                            n_missing=len(rows) - len(values), **dict(zip(PERCENTILE_NAMES, percentiles))))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--folder-list", type=Path, default=Path("folder_address.txt"))
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    labels = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.study_id_column not in labels:
        parser.error(f"Missing column: {args.study_id_column}")
    ids = labels[args.study_id_column].str.strip().tolist()
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
    rows, studies, issues = [], [], []
    for index, study_id in enumerate(ids, 1):
        detail, study, errors = inspect_study(study_id, roots)
        rows.extend(detail)
        studies.append(study)
        issues.extend(errors)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}: {study['status']}, series={len(detail)}", flush=True)
    output.mkdir(parents=True)
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=SERIES_COLUMNS)
    save_csv(table, output / "selected_series.csv")
    save_csv(table[table["flags"].map(bool)], output / "flagged_series.csv")
    save_csv(pd.DataFrame(studies), output / "study_summary.csv")
    save_csv(pd.DataFrame(issues, columns=["study_id", "file", "reason"]), output / "scan_issues.csv")
    distribution = make_distribution(rows)
    save_csv(distribution, output / "distribution_summary.csv")
    sequences = table.groupby("series_description", dropna=False).agg(
        study_count=("study_id", "nunique"), series_count=("unique_id", "size")).reset_index()
    save_csv(sequences, output / "sequence_study_counts.csv")
    lines = ["ALL MR SERIES SUMMARY", f"Study CSV: {args.csv.resolve()}",
             f"Folder list: {args.folder_list.resolve()}", f"CSV studies: {len(ids)}; unique series: {len(rows)}",
             f"Study statuses: {dict(Counter(s['status'] for s in studies))}",
             "Identity: {Study_ID}_{SerisNumber}, using the DICOM SeriesNumber value; scan all MR headers recursively.",
             "No T1/T2 description selection, pair requirement or old series-number exceptions.",
             "Choose the numerically largest AcquisitionNumber per series; all statistics describe that group only.",
             "The user's acquisition audit motivates this choice; larger acquisition is NOT an independent quality measurement.",
             "Missing acquisition: keep all if every value is missing; otherwise keep known maximum and discard missing-number files.",
             f"Series with multiple known acquisitions: {sum(len(r['acquisition_numbers']) > 1 for r in rows)}",
             f"Discarded files: {sum(r['discarded_file_count'] for r in rows)}; scan issues: {len(issues)}",
             "Distribution weights each unique series equally. Descriptions are labels, not inferred MRI contrasts.",
             "Missing studies appear in study_summary.csv, not as invented series rows.",
             "Coverage is first-to-last plane-centre span, NOT verified whole-brain coverage or image quality.",
             "No pixel decoding, hashes, preprocessing or embedding execution.", "",
             distribution.to_string(index=False, na_rep="unknown")]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished: {len(rows)} unique series. Reports: {output}")


if __name__ == "__main__":
    main()
