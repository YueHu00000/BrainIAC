"""Compare T1/T2 normal coverage, farthest-position distance and Z span."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_dicom_t1t2 import (
    PERCENTILES, PERCENTILE_NAMES, inspect_study, write_csv,
)


METRICS = ["coverage_mm", "distance_mm", "z_mm",
           "distance_minus_coverage_over_distance", "z_minus_coverage_over_distance"]


def measure_geometry(source):
    """Use image origins (first pixel centres), not image centres or corners."""
    keys = ["study_id", "modality", "status", "series_number", "series_description",
            "file_count", "frame_count", "unique_slice_count", "duplicate_plane_count", "flags", "error"]
    result = {key: source.get(key, "") for key in keys}
    result.update({metric: np.nan for metric in METRICS})
    result.update(geometry_status="unknown", farthest_frame_index_a=np.nan,
                  farthest_frame_index_b=np.nan, position_a_mm=[], position_b_mm=[])
    positions = source.get("frame_positions_mm")
    if source["status"] != "selected" or positions is None:
        return result
    positions = np.asarray(positions, dtype=float)
    # Exact projection range; unlike the older summary, do not merge nearby planes.
    result["coverage_mm"] = float(np.ptp(source["projected_positions_mm"]))
    # All pairs, with O(N) temporary memory rather than an N x N x 3 array.
    # Equal maxima keep the first pair in the converter's frame order.
    best_squared, pair = -1.0, (0, 0)
    for i in range(len(positions) - 1):
        delta = positions[i + 1:] - positions[i]
        squared = np.einsum("ij,ij->i", delta, delta)
        j = int(np.argmax(squared))
        if squared[j] > best_squared:
            best_squared, pair = float(squared[j]), (i, i + 1 + j)
    a, b = pair
    distance = float(np.sqrt(max(best_squared, 0)))
    z = float(abs(positions[b, 2] - positions[a, 2]))
    result.update(distance_mm=distance, z_mm=z, farthest_frame_index_a=a,
                  farthest_frame_index_b=b, position_a_mm=positions[a].tolist(),
                  position_b_mm=positions[b].tolist(),
                  geometry_status="ok" if distance > 0 else "zero_distance")
    if distance > 0:
        result[METRICS[3]] = (distance - result["coverage_mm"]) / distance
        result[METRICS[4]] = (z - result["coverage_mm"]) / distance
    return result


def distribution_table(rows):
    records = []
    for modality in ("T1", "T2"):
        subset = [row for row in rows if row["modality"] == modality]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            values = values[np.isfinite(values)]
            percentiles = np.percentile(values, PERCENTILES) if len(values) else [np.nan] * len(PERCENTILES)
            records.append(dict(modality=modality, metric=metric, n_total=len(subset),
                                n_valid=len(values), n_unknown=len(subset) - len(values),
                                mean=float(values.mean()) if len(values) else np.nan,
                                **dict(zip(PERCENTILE_NAMES, percentiles))))
    return pd.DataFrame(records)


def study_table(rows):
    records = {}
    for row in rows:
        record = records.setdefault(row["study_id"], {"Study_ID": row["study_id"]})
        for key in ["status", "geometry_status", *METRICS]:
            record[f"{row['modality']}_{key}"] = row[key]
    return pd.DataFrame(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="labels.csv")
    parser.add_argument("--folder-list", type=Path, default=Path("folder_address.txt"))
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory outside DICOM roots")
    args = parser.parse_args()
    table = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.study_id_column not in table:
        parser.error(f"Missing CSV column: {args.study_id_column}")
    ids = table[args.study_id_column].str.strip().tolist()
    if not ids or any(not x or x in (".", "..") or any(c in x for c in '/\\:*?"<>|') for x in ids):
        parser.error("Study IDs must be nonempty plain directory names")
    if len(set(ids)) != len(ids):
        parser.error("Duplicate Study IDs in CSV")
    roots = []
    for line in args.folder_list.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            root = Path(line.strip()).expanduser()
            roots.append((root if root.is_absolute() else args.folder_list.resolve().parent / root).resolve())
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        parser.error("Roots must be existing unique directories; relative roots resolve against the list file")
    output = args.output_dir.resolve()
    if any(output == root or root in output.parents for root in roots):
        parser.error("Output directory must be outside DICOM roots")
    if output.exists():
        parser.error("Output directory exists; choose a new directory")
    output.mkdir(parents=True)
    rows = []
    for index, study_id in enumerate(ids, 1):
        pair = inspect_study(study_id, roots, include_frame_positions=True)
        rows.extend(measure_geometry(row) for row in pair)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}", flush=True)
    distribution = distribution_table(rows)
    write_csv(rows, output / "series_geometry.csv")
    for name, data in (("study_geometry", study_table(rows)), ("distribution_summary", distribution)):
        data.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig", na_rep="unknown", float_format="%.12g")
    lines = ["DICOM T1/T2 COVERAGE, DISTANCE AND Z COMPARISON",
             f"Study CSV: {args.csv.resolve()}", f"Folder list: {args.folder_list.resolve()}",
             f"Study ID column: {args.study_id_column}", f"Studies: {len(ids)}; selected-modality slots: {len(rows)}",
             "Selection: same inspect_study as summarize_dicom_t1t2.py, including its four exceptions.",
             "All CSV studies retained, including repeated-plane studies. Each modality is assessed separately.",
             "Only headers are read; no preprocessing, pixel loading or embedding is performed.",
             "coverage_mm = max(position dot normal) - min(position dot normal), using a common unit normal.",
             "This exact range does not merge nearby planes; old coverage may differ by <=0.01 mm due to merging.",
             "distance_mm = maximum Euclidean distance across ALL pairs of ImagePositionPatient coordinates.",
             "ImagePositionPatient is the first pixel centre, not the image centre or the whole image extent.",
             "z_mm = abs(Z_b-Z_a) for that SAME farthest pair, not necessarily max(Z)-min(Z) across all images.",
             "Ratios: (distance-coverage)/distance and (z-coverage)/distance; fractions, NOT percentages.",
             "The second ratio keeps its sign. It can be positive or negative; no absolute value is applied.",
             "Farthest pair indices are zero-based flattened frame indices in converter order.",
             "Equal maximum distances keep the first pair encountered; different tied pairs can have different Z spans.",
             "Missing/inconsistent geometry or incompatible coordinate frames gives unknown, never fabricated zero.",
             "Zero distance retains distance/Z/coverage but leaves both ratios unknown.",
             "Repeated planes stay included and flagged; these measurements do not certify a valid single volume.",
             "Percentiles use linear interpolation, weighting each study/modality equally; unknowns excluded per metric."]
    for modality in ("T1", "T2"):
        subset = [r for r in rows if r["modality"] == modality]
        lines += ["", f"=== {modality} ===",
                  f"Selection statuses: {dict(Counter(r['status'] for r in subset))}",
                  f"Geometry statuses: {dict(Counter(r['geometry_status'] for r in subset))}",
                  f"Repeated-plane series: {sum('repeated_slice_plane' in r['flags'] for r in subset)}",
                  distribution[distribution.modality == modality].to_string(index=False, na_rep="unknown", float_format=lambda x: f"{x:.8g}")]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished. Reports: {output}")


if __name__ == "__main__":
    main()
