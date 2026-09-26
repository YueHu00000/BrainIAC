"""Join low-coverage studies and marked repeated-plane studies to clinical labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_dicom_t1t2_v2 import read_selected


IMAGE_COLUMNS = ["T1_unique_slice_count", "T2_unique_slice_count", "T1_coverage_mm", "T2_coverage_mm"]


def build_table(selected_csv, labels_csv, coverage_threshold=100.0):
    if not np.isfinite(coverage_threshold) or coverage_threshold <= 0:
        raise ValueError("Coverage threshold must be finite and positive")
    selected = read_selected(selected_csv)
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "Study_ID" not in labels or len(labels.columns) < 2:
        raise ValueError("labels.csv must have a Study_ID column and at least two columns")
    clinical_columns = [name for name in labels.columns[1:-1] if name != "Study_ID"]
    if set(clinical_columns) & set(IMAGE_COLUMNS):
        raise ValueError("Clinical column names conflict with the output imaging columns")
    # Match the converter's string-ID convention, keeping leading zeros and literal NA.
    labels["Study_ID"] = labels.Study_ID.str.strip()
    selected["study_id"] = selected.study_id.str.strip()
    if labels.Study_ID.eq("").any() or labels.Study_ID.duplicated().any():
        raise ValueError("labels.csv contains empty or duplicate Study_ID values")
    if selected.duplicated(["study_id", "modality"]).any():
        raise ValueError("Duplicate study/modality rows after trimming Study IDs")

    repeated = selected.duplicate_plane_count.gt(0) | selected["flags"].map(lambda v: "repeated_slice_plane" in v)
    excluded_ids = selected.loc[repeated, "study_id"].unique()
    retained = selected[~selected.study_id.isin(excluded_ids)]
    # Same coverage-failure condition as Summary_v2; missing/invalid values do not qualify.
    low = (retained.status.eq("selected") & np.isfinite(retained.coverage_mm)
           & retained.coverage_mm.ge(0) & retained.coverage_mm.lt(coverage_threshold))
    low_ids = retained.loc[low, "study_id"].unique()
    included = selected.study_id.isin(low_ids) | selected.study_id.isin(excluded_ids)
    study_ids = selected.loc[included, "study_id"].drop_duplicates().tolist()
    paired = selected.pivot(index="study_id", columns="modality", values=["unique_slice_count", "coverage_mm"])
    paired = paired.reindex(study_ids)

    # Reindex is a left join: studies missing from labels remain present with blank clinical cells.
    output = labels.set_index("Study_ID")[clinical_columns].reindex(study_ids).fillna("")
    output = output.rename_axis("Study_ID").reset_index()
    for column in IMAGE_COLUMNS:
        modality, metric = column.split("_", 1)
        output[column] = paired.get((metric, modality), pd.Series(index=paired.index, dtype=float)).to_numpy()
    stats = dict(input_studies=selected.study_id.nunique(), included_repeated_studies=len(excluded_ids),
                 nonrepeated_studies=retained.study_id.nunique(), low_coverage_studies=len(low_ids),
                 output_studies=len(output),
                 T1_low_coverage_series=int((low & retained.modality.eq("T1")).sum()),
                 T2_low_coverage_series=int((low & retained.modality.eq("T2")).sum()),
                 unmatched_labels=int((~output.Study_ID.isin(labels.Study_ID)).sum()))
    # Match clinical rows using original IDs first; only the exported display ID gets a prefix.
    marked = output.Study_ID.isin(excluded_ids)
    output.loc[marked, "Study_ID"] = "excluded_" + output.loc[marked, "Study_ID"]
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path, help="Original selected_series.csv with study_id")
    parser.add_argument("--labels-csv", required=True, type=Path, help="Clinical CSV with a Study_ID header")
    parser.add_argument("--output-csv", required=True, type=Path, help="New output CSV; existing files are not overwritten")
    parser.add_argument("--coverage-threshold-mm", type=float, default=100.0)
    args = parser.parse_args()
    if args.output_csv.exists():
        parser.error("Output CSV already exists; choose a new filename")
    output, stats = build_table(args.selected_csv, args.labels_csv, args.coverage_threshold_mm)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        output.to_csv(handle, index=False, na_rep="unknown", float_format="%.9g")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
