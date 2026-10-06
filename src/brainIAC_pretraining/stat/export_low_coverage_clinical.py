"""Export low-coverage series with clinical fields and previous T1/T2 selection."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_COLUMNS = ["frame_count", "coverage_mm", "is_previous_t1t2_selected",
                 "previous_selected_modality"]


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def series_numbers(values):
    numbers = pd.to_numeric(values, errors="coerce")
    if not (np.isfinite(numbers) & numbers.eq(np.floor(numbers))).all():
        raise ValueError("Selected rows must contain integer series_number values")
    return numbers.astype("int64")


def build_table(selected_csv, old_selected_csv, labels_csv, coverage_threshold=100.0):
    if not np.isfinite(coverage_threshold) or coverage_threshold <= 0:
        raise ValueError("Coverage threshold must be finite and positive")
    selected, previous, labels = map(read_csv, (selected_csv, old_selected_csv, labels_csv))
    required = {"unique_id", "study_id", "series_number", "status", "frame_count", "coverage_mm"}
    if not required.issubset(selected.columns):
        raise ValueError(f"All-series CSV missing columns: {sorted(required - set(selected.columns))}")
    required_old = {"study_id", "modality", "status", "series_number"}
    if not required_old.issubset(previous.columns):
        raise ValueError("Use the original T1/T2 selected_series.csv with study_id, "
                         "modality, status and series_number; the old clinical export is insufficient")
    if "Study_ID" not in labels or len(labels.columns) < 2:
        raise ValueError("labels.csv must have a Study_ID column and at least two columns")
    clinical_columns = [name for name in labels.columns[1:-1] if name != "Study_ID"]
    if set(clinical_columns) & {"unique_id", "series_number", *IMAGE_COLUMNS}:
        raise ValueError("Clinical column names conflict with output series columns")
    for table, column in ((selected, "study_id"), (previous, "study_id"), (labels, "Study_ID")):
        table[column] = table[column].str.strip()
        if table[column].eq("").any():
            raise ValueError(f"Empty {column} in input CSV")
    if labels.Study_ID.duplicated().any():
        raise ValueError("Duplicate Study_ID in labels.csv")
    selected["series_number"] = series_numbers(selected.series_number)
    expected_ids = selected.study_id + "_" + selected.series_number.astype(str)
    if not selected.unique_id.eq(expected_ids).all():
        raise ValueError("All-series unique_id must equal {Study_ID}_{SeriesNumber}")
    if selected.unique_id.duplicated().any():
        raise ValueError("Duplicate unique_id in all-series CSV")
    if not previous.modality.isin(["T1", "T2"]).all():
        raise ValueError("Previous CSV must be the old T1/T2 selection, not the all-series CSV")
    if previous.duplicated(["study_id", "modality"]).any():
        raise ValueError("Duplicate study/modality in previous selection CSV")

    # Use actual historical selections; do not rerun maximum-series/exception rules.
    chosen = previous[previous.status.eq("selected")].copy()
    chosen["series_number"] = series_numbers(chosen.series_number)
    chosen["unique_id"] = chosen.study_id + "_" + chosen.series_number.astype(str)
    old_modalities = chosen.groupby("unique_id", sort=False).modality.agg(
        lambda values: ";".join(m for m in ("T1", "T2") if m in set(values))).to_dict()
    known_studies = {
        study for study, rows in previous.groupby("study_id", sort=False)
        if set(rows.modality) == {"T1", "T2"}
        and rows.status.isin(["selected", "no_candidate"]).all()
    }

    coverage = pd.to_numeric(selected.coverage_mm, errors="coerce")
    low = (selected.status.eq("selected") & np.isfinite(coverage)
           & coverage.ge(0) & coverage.lt(coverage_threshold))
    retained = selected.loc[low].copy()
    retained["coverage_mm"] = coverage[low]
    retained["frame_count"] = pd.to_numeric(retained.frame_count, errors="coerce")
    output = retained[["unique_id", "study_id", "series_number"]].rename(
        columns={"study_id": "Study_ID"})
    output = output.merge(labels[["Study_ID", *clinical_columns]], on="Study_ID", how="left", sort=False)
    output[clinical_columns] = output[clinical_columns].fillna("")
    output["frame_count"] = retained.frame_count.to_numpy()
    output["coverage_mm"] = retained.coverage_mm.to_numpy()
    output["previous_selected_modality"] = output.unique_id.map(old_modalities).fillna("")
    output["is_previous_t1t2_selected"] = [
        "yes" if unique_id in old_modalities else "no" if study in known_studies else "unknown"
        for unique_id, study in zip(output.unique_id, output.Study_ID)
    ]
    output = output[["unique_id", "Study_ID", "series_number", *clinical_columns, *IMAGE_COLUMNS]]
    stats = dict(input_series=len(selected), low_coverage_series=len(output),
                 low_coverage_studies=output.Study_ID.nunique(),
                 previous_selected_series=int(output.is_previous_t1t2_selected.eq("yes").sum()),
                 previous_selection_unknown=int(output.is_previous_t1t2_selected.eq("unknown").sum()),
                 unmatched_labels=int((~output.Study_ID.isin(labels.Study_ID)).sum()))
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path, help="All-series selected_series.csv")
    parser.add_argument("--old-selected-csv", required=True, type=Path,
                        help="Original T1/T2 selected_series.csv; determines actual previous selection")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--coverage-threshold-mm", type=float, default=100.0)
    args = parser.parse_args()
    if args.output_csv.exists():
        parser.error("Output CSV already exists; choose a new filename")
    output, stats = build_table(args.selected_csv, args.old_selected_csv, args.labels_csv,
                               args.coverage_threshold_mm)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        output.to_csv(handle, index=False, na_rep="unknown", float_format="%.9g")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
