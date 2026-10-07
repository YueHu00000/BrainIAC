"""Screen UID-based selected_series.csv from summary_dicom_SeriesInstanceUID.py."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _series_statistics import combined_status, counts, position_check, save_csv, secondary_checks


NUMERIC = ("file_count", "frame_count", "unique_slice_count", "coverage_mm", "duplicate_plane_count")


def read_selected(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"unique_id", "study_id", "series_instance_uid", "series_number", "modality", "status", "flags", *NUMERIC}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}; use summary_dicom_SeriesInstanceUID.py")
    if table.unique_id.eq("").any() or table.unique_id.duplicated().any():
        raise ValueError("unique_id must be nonempty and unique")
    if table.series_instance_uid.str.strip().eq("").any() or table.series_instance_uid.eq("unknown").any():
        raise ValueError("series_instance_uid must be present; regenerate the UID-based statistics")
    expected = table.apply(lambda row: f"{row.study_id}_{row.series_instance_uid}_{int(row.series_number)}", axis=1) if len(table) else table.unique_id
    if not table.unique_id.eq(expected).all():
        raise ValueError("unique_id must equal {Study_ID}_{SeriesInstanceUID}_{SeriesNumber}")
    table["flags"] = table["flags"].map(json.loads)
    fields = [*NUMERIC, "rows", "columns", "max_in_plane_shift_mm"]
    fields += [f"{name}_{stat}_mm" for name in ("pixel_row", "pixel_column", "thickness", "tag_spacing")
               for stat in ("min", "median", "max")]
    for field in fields:
        table[field] = pd.to_numeric(table.get(field, pd.Series(index=table.index, dtype=float)), errors="coerce")
    return table


def analyze(table, coverage_threshold=100.0, error_percent=20.0):
    repeated = table.duplicate_plane_count.gt(0) | table["flags"].map(lambda values: "repeated_slice_plane" in values)
    excluded = table[repeated].copy()
    excluded["exclusion_reason"] = "repeated_slice_plane_in_this_series"
    results, slices = [], []
    for row in table[~repeated].to_dict("records"):
        coverage = row["coverage_mm"]
        known = row["status"] == "selected" and np.isfinite(coverage) and coverage >= 0
        row["coverage_status"] = ("fail" if coverage < coverage_threshold else "pass") if known else "unknown"
        position, detail = position_check(row, error_percent)
        if "missing_instance_number" in row["flags"] or "partial_study_scan" in row["flags"]:
            position.update(position_status="unknown", position_reason="incomplete_input_or_instance_order",
                            position_bad_slices=np.nan, position_assessed_slices=0)
            detail = []
        row.update(position)
        complete = "partial_study_scan" not in row["flags"]
        count_known = complete and np.isfinite(row["unique_slice_count"]) and row["unique_slice_count"] >= 1
        row["screen_status"] = combined_status([row["coverage_status"], row["position_status"],
                                               "pass" if count_known else "unknown"])
        results.append(row)
        for entry in detail:
            entry.update(unique_id=row["unique_id"], series_instance_uid=row["series_instance_uid"],
                         series_number=row["series_number"],
                         acquisition_number=row.get("acquisition_number", "unknown"))
        slices.extend(detail)
    columns = [*table.columns, "coverage_status", "position_status", "position_reason",
               "position_threshold_mm", "position_max_error_mm", "position_max_error_percent",
               "position_bad_slices", "position_assessed_slices", "screen_status"]
    result = pd.DataFrame(results, columns=columns)
    for column in (*NUMERIC, "rows", "columns", "position_bad_slices", "position_assessed_slices"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    slice_columns = ["unique_id", "study_id", "series_instance_uid", "series_number", "acquisition_number", "modality", "slice_index",
                     "file_name", "instance_number", "actual_position_mm", "equal_spacing_position_mm",
                     "error_mm", "error_percent", "threshold_mm", "exceeds_threshold"]
    return result, excluded, pd.DataFrame(slices, columns=slice_columns)


def count_tables(results):
    exact, thresholds = [], []
    for count, group in results.groupby("unique_slice_count", dropna=False, sort=True):
        exact.append(dict(unique_slice_count=count, **counts(group)))
    known = results[np.isfinite(results.unique_slice_count)]
    for threshold in sorted({int(n) + 1 for n in known.unique_slice_count}):
        for side, group in (("below", known[known.unique_slice_count < threshold]),
                            ("at_or_above", known[known.unique_slice_count >= threshold])):
            thresholds.append(dict(threshold=threshold, side=side,
                                   unknown_slice_count=len(results) - len(known), **counts(group)))
    fields = list(counts(results))
    return (pd.DataFrame(exact, columns=["unique_slice_count", *fields]),
            pd.DataFrame(thresholds, columns=["threshold", "side", "unknown_slice_count", *fields]))


def study_table(table, results, excluded):
    records = []
    for study_id, group in table.groupby("study_id", sort=False):
        selected = results[results.study_id == study_id]
        records.append(dict(study_id=study_id, n_series=len(group), n_retained=len(selected),
                            n_excluded=int(excluded.study_id.eq(study_id).sum()),
                            n_pass=int(selected.screen_status.eq("pass").sum()),
                            n_fail=int(selected.screen_status.eq("fail").sum()),
                            n_unknown=int(selected.screen_status.eq("unknown").sum())))
    return pd.DataFrame(records, columns=["study_id", "n_series", "n_retained", "n_excluded", "n_pass", "n_fail", "n_unknown"])


def export_reports(table, results, excluded, slices, studies, exact, thresholds, secondary, output):
    study_indices = {key: i + 1 for i, key in enumerate(table.study_id.drop_duplicates())}
    series_indices = {key: i + 1 for i, key in enumerate(table.unique_id)}
    for directory in ("csv", "csv_with_abnormal_study_ids"):
        (output / directory).mkdir()
    reports = [("series_quality_v2", results, results.screen_status.eq("fail")),
               ("excluded_repeated_series", excluded, pd.Series(True, index=excluded.index)),
               ("study_quality_v2", studies, studies.n_fail.gt(0) | studies.n_excluded.gt(0)),
               ("slice_position_deviations", slices, slices.exceeds_threshold.eq(True))]
    for name, data, failures in reports:
        public = data.drop(columns=[key for key in ("study_id", "unique_id", "file_name", "study_directory",
                         "file_names", "series_description", "series_descriptions", "study_instance_uids",
                         "series_instance_uid", "series_instance_uids", "frame_of_reference_uids", "image_types", "protocol_names")
                         if key in data]).copy()
        public.insert(0, "study_index", data.study_id.map(study_indices))
        if "unique_id" in data:
            public.insert(1, "series_index", data.unique_id.map(series_indices))
        save_csv(public, output / "csv" / f"{name}.csv")
        public["abnormal_study_ids"] = [[s] if bad else "" for s, bad in zip(data.study_id, failures)]
        if "unique_id" in data:
            public["abnormal_unique_ids"] = [[s] if bad else "" for s, bad in zip(data.unique_id, failures)]
        save_csv(public, output / "csv_with_abnormal_study_ids" / f"{name}.csv")
    for name, data in (("slice_count_analysis", exact), ("slice_count_thresholds", thresholds),
                       ("secondary_checks", secondary)):
        public = data.drop(columns="_abnormal_ids").copy()
        save_csv(public, output / "csv" / f"{name}.csv")
        public["abnormal_study_ids"] = data["_abnormal_ids"]
        save_csv(public, output / "csv_with_abnormal_study_ids" / f"{name}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coverage-threshold-mm", type=float, default=100.0)
    parser.add_argument("--position-error-percent", type=float, default=20.0)
    args = parser.parse_args()
    if not np.isfinite(args.coverage_threshold_mm) or args.coverage_threshold_mm <= 0:
        parser.error("Coverage threshold must be finite and positive")
    if not np.isfinite(args.position_error_percent) or not 0 <= args.position_error_percent <= 100:
        parser.error("Position error percent must be between 0 and 100")
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new directory")
    table = read_selected(args.selected_csv)
    results, excluded, slices = analyze(table, args.coverage_threshold_mm, args.position_error_percent)
    exact, thresholds = count_tables(results)
    studies = study_table(table, results, excluded)
    secondary = secondary_checks(results)
    # Keep headers for an empty cohort, including when every series was excluded.
    if secondary.empty:
        secondary = pd.DataFrame(columns=["modality", "check", "n_total", "n_pass", "n_flagged", "n_unknown", "_abnormal_ids"])
    args.output_dir.mkdir(parents=True)
    export_reports(table, results, excluded, slices, studies, exact, thresholds, secondary, args.output_dir)
    summary = counts(results)
    lines = ["ALL UNIQUE SERIES SUMMARY V2", f"Input: {args.selected_csv.resolve()}",
             f"Input studies with series: {table.study_id.nunique()}; input series: {len(table)}",
             f"Excluded repeated-plane series: {len(excluded)}; retained series: {len(results)}",
             "Only repeated series are excluded; other series from the same study remain.",
             "Input identity: {Study_ID}_{SeriesInstanceUID}_{SeriesNumber}.",
             "Input already contains the maximum-acquisition group for every Study_ID/SeriesInstanceUID/SeriesNumber.",
             f"Coverage failure: span < {args.coverage_threshold_mm:g} mm; equality passes.",
             f"Position failure: slice error > {args.position_error_percent:g}% of series coverage; equality passes.",
             "Position model: endpoint equal spacing in exported InstanceNumber order; not a full 3D registration simulation.",
             "Multi-frame position screening, missing instance order and incomplete scans remain unknown.",
             "No T1/T2 pairing or study-wide pass requirement. Study summary counts each series outcome.",
             "Screen pass is only a header-based rule outcome, not verified brain coverage or image quality.",
             f"Retained-series counts: {summary}", "", "EXACT SLICE COUNTS:",
             exact.drop(columns="_abnormal_ids").to_string(index=False, na_rep="unknown")]
    (args.output_dir / "summary_v2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished: {len(results)} retained series; {len(excluded)} excluded series. {args.output_dir}")


if __name__ == "__main__":
    main()
