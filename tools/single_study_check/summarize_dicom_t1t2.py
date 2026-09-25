"""Summarize CSV-listed studies using the original Harvey/BrainIAC T1/T2 selection."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from inspect_dicom_t1 import EXCEPTIONS, geometry


POSITION_TOL_MM = 0.01
ORIENTATION_TOL = 1e-4
GAP_RTOL = 0.01
MEASURES = ("pixel_row", "pixel_column", "thickness", "tag_spacing")
METRICS = ["file_count", "frame_count", "unique_slice_count", "rows", "columns",
           "coverage_mm", "gap_min_mm", "gap_median_mm", "gap_max_mm",
           "duplicate_plane_count", "max_in_plane_shift_mm"]
METRICS += [f"{name}_{stat}_mm" for name in MEASURES for stat in ("min", "median", "max")]
PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
PERCENTILE_NAMES = ["min", "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]


def numbers(value, size):
    """Missing or malformed numerical metadata stays unknown."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (ValueError, TypeError):
        return None
    return array if array.size == size and np.isfinite(array).all() else None


def summarize_series(items):
    result = {metric: np.nan for metric in METRICS}
    flags = []
    result["flags"] = flags
    result["file_count"] = len(items)
    result["file_names"] = [path.name for path, _ in items]
    result["instance_numbers"] = [int(ds.InstanceNumber) for _, ds in items]
    for field, column in (("StudyInstanceUID", "study_instance_uids"),
                          ("SeriesInstanceUID", "series_instance_uids"),
                          ("FrameOfReferenceUID", "frame_of_reference_uids")):
        values = sorted({str(getattr(ds, field)) for _, ds in items if getattr(ds, field, None)})
        result[column] = values
        if len(values) > 1:
            flags.append(f"multiple_{column}")
        if any(not getattr(ds, field, None) for _, ds in items):
            flags.append(f"missing_{field}")
    result["image_types"] = sorted({str(getattr(ds, "ImageType", "unknown")) for _, ds in items})
    result["protocol_names"] = sorted({str(getattr(ds, "ProtocolName", "unknown")) for _, ds in items})
    if any("DERIVED" in getattr(ds, "ImageType", []) for _, ds in items):
        flags.append("derived_image")
    sop_uids = [str(ds.SOPInstanceUID) for _, ds in items if getattr(ds, "SOPInstanceUID", None)]
    if len(sop_uids) != len(set(sop_uids)):
        flags.append("duplicate_sop_instance_uid")
    if len(set(result["instance_numbers"])) < len(items):
        flags.append("duplicate_instance_number")
    for field in ("EchoNumbers", "TemporalPositionIdentifier"):
        if len({str(getattr(ds, field)) for _, ds in items if hasattr(ds, field)}) > 1:
            flags.append(f"multiple_{field}")
    for field, key in (("Rows", "rows"), ("Columns", "columns")):
        values = [numbers(getattr(ds, field, None), 1) for _, ds in items]
        if all(v is not None and v[0] > 0 for v in values) and len({v[0] for v in values}) == 1:
            result[key] = float(values[0][0])
        else:
            flags.append(f"missing_or_varying_{key}")

    frames = []
    frame_count_known = True
    for _, ds in items:
        count = numbers(getattr(ds, "NumberOfFrames", 1), 1)
        if count is None or count[0] < 1 or count[0] != int(count[0]):
            frame_count_known = False
            flags.append("invalid_frame_count")
            continue
        count = int(count[0])
        per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
        usable_frames = len(per_frame) == count
        if (count > 1 or per_frame) and not usable_frames:
            flags.append("missing_or_incomplete_per_frame_geometry")
        for index in range(count):
            frame = per_frame[index] if usable_frames else None
            geo = geometry(ds, frame)
            if (count > 1 or per_frame) and not usable_frames:
                # A single top-level position must not be invented for all frames.
                geo["ImagePositionPatient"] = None
            if count > 1 and usable_frames:
                plane = getattr(frame, "PlanePositionSequence", [])
                geo["ImagePositionPatient"] = getattr(plane[0], "ImagePositionPatient", None) if plane else None
            spacing = getattr(ds, "SpacingBetweenSlices", None)
            shared = list(getattr(ds, "SharedFunctionalGroupsSequence", [])[:1])
            for group in shared + ([frame] if frame is not None else []):
                measures = getattr(group, "PixelMeasuresSequence", [])
                if measures:
                    spacing = getattr(measures[0], "SpacingBetweenSlices", spacing)
            geo["SpacingBetweenSlices"] = spacing
            frames.append(geo)
    if frame_count_known:
        result["frame_count"] = len(frames)
    positions, orientations = [], []
    measurements = {key: [] for key in MEASURES}
    for geo in frames:
        positions.append(numbers(geo["ImagePositionPatient"], 3))
        orientations.append(numbers(geo["ImageOrientationPatient"], 6))
        pixel = numbers(geo["PixelSpacing"], 2)
        for index, key in enumerate(("pixel_row", "pixel_column")):
            measurements[key].append(pixel[index] if pixel is not None and pixel[index] > 0 else np.nan)
        for field, key in (("SliceThickness", "thickness"), ("SpacingBetweenSlices", "tag_spacing")):
            value = numbers(geo[field], 1)
            measurements[key].append(value[0] if value is not None and value[0] > 0 else np.nan)
    for key, values in measurements.items():
        known = int(np.isfinite(values).sum())
        result[f"{key}_known_frames"] = known
        if frame_count_known and known == len(frames) and known:
            for stat, value in zip(("min", "median", "max"), np.percentile(values, [0, 50, 100], method="linear")):
                result[f"{key}_{stat}_mm"] = float(value)
            if not np.allclose(values, values[0], atol=1e-5, rtol=1e-4):
                flags.append(f"varying_{key}")
        else:
            flags.append(f"missing_or_invalid_{key}")
    result["position_known_frames"] = sum(p is not None for p in positions)
    result["orientation_known_frames"] = sum(o is not None for o in orientations)
    complete = frame_count_known and bool(frames) and all(p is not None for p in positions) and all(o is not None for o in orientations)
    if not complete:
        flags.append("incomplete_geometry")
    elif not np.allclose(orientations, orientations[0], atol=ORIENTATION_TOL, rtol=0):
        flags.append("varying_orientation")
    elif len(result["study_instance_uids"]) > 1 or len(result["frame_of_reference_uids"]) > 1:
        flags.append("incompatible_coordinate_frames")
    else:
        row, column = orientations[0][:3], orientations[0][3:]
        valid = (abs(np.linalg.norm(row) - 1) <= ORIENTATION_TOL and
                 abs(np.linalg.norm(column) - 1) <= ORIENTATION_TOL and
                 abs(np.dot(row, column)) <= ORIENTATION_TOL)
        if not valid:
            flags.append("invalid_orientation")
        else:
            normal = np.cross(row, column)
            normal /= np.linalg.norm(normal)
            offsets = np.asarray(positions) - positions[0]
            projected = offsets @ normal
            shift = np.linalg.norm(offsets - projected[:, None] * normal, axis=1)
            result["max_in_plane_shift_mm"] = float(shift.max())
            if shift.max() > POSITION_TOL_MM:
                flags.append("in_plane_origin_shift")
            unique = []
            for position in np.sort(projected):
                if not unique or position - unique[-1] > POSITION_TOL_MM:
                    unique.append(float(position))
            gaps = np.diff(unique)
            steps = np.diff(projected)
            result.update(unique_slice_count=len(unique), duplicate_plane_count=len(frames) - len(unique),
                          coverage_mm=unique[-1] - unique[0], projected_positions_mm=projected.tolist(),
                          ordered_steps_mm=steps.tolist(), sorted_unique_gaps_mm=gaps.tolist())
            if result["duplicate_plane_count"]:
                flags.append("repeated_slice_plane")
            if np.any(steps > POSITION_TOL_MM) and np.any(steps < -POSITION_TOL_MM):
                flags.append("nonmonotonic_instance_order")
            result["nonuniform_spacing"] = "unknown"
            if gaps.size:
                result.update(gap_min_mm=float(gaps.min()), gap_median_mm=float(np.median(gaps)),
                              gap_max_mm=float(gaps.max()))
            if gaps.size >= 2:
                nonuniform = float(np.ptp(gaps)) > POSITION_TOL_MM + GAP_RTOL * float(np.median(gaps))
                result["nonuniform_spacing"] = bool(nonuniform)
                if nonuniform:
                    flags.append("nonuniform_spacing")
            tag = result["tag_spacing_median_mm"]
            if gaps.size and np.isfinite(tag) and np.any(np.abs(gaps - tag) > POSITION_TOL_MM + GAP_RTOL * tag):
                flags.append("position_gap_differs_from_tag_spacing")
    for metric in ("file_count", "frame_count", "unique_slice_count"):
        if np.isfinite(result[metric]) and result[metric] < 20:
            flags.append(f"{metric}_lt20")
    result["flags"] = list(dict.fromkeys(flags))
    return result


def inspect_study(study_id, roots):
    rows = [{"study_id": study_id, "modality": modality, "status": "pending", "flags": [],
             "excluded_series_number": EXCEPTIONS.get(study_id, "none"), "pair_selected": False,
             **{metric: np.nan for metric in METRICS}} for modality in ("T1", "T2")]
    matches = [Path(root) / study_id for root in roots if (Path(root) / study_id).is_dir()]
    if len(matches) != 1:
        status = "study_not_found" if not matches else "study_ambiguous"
        for row in rows:
            row.update(status=status, flags=[status], error="; ".join(map(str, matches)))
        return rows
    directory = matches[0]
    for row in rows:
        row["study_directory"] = str(directory)
    data, headers = [], {}
    path = directory
    try:
        for filename in os.listdir(directory):
            path = directory / filename
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            # Same mandatory fields and grouping as the original converter.
            data.append((filename, int(ds.SeriesNumber), str(ds.SeriesDescription), int(ds.InstanceNumber)))
            headers[filename] = ds
    except Exception as error:
        # Without this child's metadata the original selection cannot be certified.
        for row in rows:
            row.update(status="read_error", flags=["read_error"], error=f"{path}: {type(error).__name__}: {error}")
        return rows
    frame = pd.DataFrame(data, columns=["Filename", "SeriesNumber", "SeriesDescription", "InstanceNumber"])
    frame = frame.sort_values(["SeriesNumber", "InstanceNumber"])
    grouped = frame.groupby(["SeriesNumber", "SeriesDescription"]).agg({"Filename": list}).reset_index()
    if study_id in EXCEPTIONS:
        grouped = grouped[grouped.SeriesNumber != EXCEPTIONS[study_id]]
    for row in rows:
        candidates = grouped[grouped.SeriesDescription.str.contains(row["modality"], case=False, na=False)]
        row["candidate_group_count"] = len(candidates)
        if candidates.empty:
            row.update(status="no_candidate", flags=["no_candidate"])
            continue
        chosen = candidates.loc[candidates.SeriesNumber.idxmax()]
        row.update(series_number=int(chosen.SeriesNumber), series_description=chosen.SeriesDescription)
        row.update(summarize_series([(directory / name, headers[name]) for name in chosen.Filename]))
        row["status"] = "selected"
        if (candidates.SeriesNumber == chosen.SeriesNumber).sum() > 1:
            row["flags"].append("max_series_number_tie")
    for row in rows:
        row["pair_selected"] = all(r["status"] == "selected" for r in rows)
    return rows


def make_distribution(rows):
    records = []
    for modality in ("T1", "T2"):
        selected = [row for row in rows if row["modality"] == modality]
        for metric in METRICS:
            values = np.asarray([row.get(metric, np.nan) for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            percentiles = np.percentile(values, PERCENTILES, method="linear") if values.size else [np.nan] * len(PERCENTILES)
            records.append(dict(modality=modality, metric=metric, n_total=len(selected), n_valid=len(values),
                                n_missing=len(selected) - len(values), **dict(zip(PERCENTILE_NAMES, percentiles))))
    return pd.DataFrame(records)


def make_summary(rows, distribution, csv_path, folder_list, study_id_column):
    total = len(rows) // 2
    lines = ["SELECTED DICOM T1/T2 COHORT SUMMARY", f"Study CSV: {csv_path}",
             f"Study ID column: {study_id_column}", f"DICOM root list: {folder_list}",
             f"CSV studies: {total}; expected modality records: {total * 2}",
             "Population: ALL CSV-listed studies, including the reported one embedding failure.",
             "Embedding success was not independently checked; this is not a success-only cohort.",
             "Selection: direct children; (SeriesNumber, SeriesDescription) groups; description contains T1/T2;",
             "maximum SeriesNumber after four original exceptions. Ties follow sorted group order.",
             "A read error invalidates selection for the study; no silent file skipping or fallback series.",
             "Statistics weight each selected study/modality equally; not each slice.",
             "Unknown values are excluded per metric, with all-study valid/missing denominators retained.",
             "Voxel spacing here means DICOM in-plane PixelSpacing and position-derived inter-plane distances.",
             "No raw NIfTI, processed NIfTI, embeddings, GPU inference or pixel decoding was required.",
             f"Position/plane tolerance: {POSITION_TOL_MM} mm; orientation tolerance: {ORIENTATION_TOL}.",
             f"Nonuniformity: max(gaps)-min(gaps) > {POSITION_TOL_MM} mm + {GAP_RTOL} * median(gaps).",
             "At least 3 distinct planes are needed to assess spacing uniformity.",
             "Repeated planes are collapsed before sorted spatial gaps; original ordered steps remain in CSV.",
             "For 0,63,70 mm: gaps=63,7 mm; min=7, median=35, max=63, span=70. Median does NOT imply uniform sampling.",
             "Unique slice count is distinct plane position along a common normal; it is unknown for incomplete/varying geometry.",
             "Thickness/tag spacing summaries require a valid positive value on every frame; partial availability is reported separately.",
             "Coverage is first-to-last plane-centre span, not verified whole-brain coverage.",
             "Low-count thresholds below are descriptive, not validated model eligibility criteria.",
             "Percentiles: NumPy linear interpolation; count percentiles can be fractional.",
             "CSV numeric formatting: 9 significant digits; unknown is never replaced with zero.",
             f"Dependencies: numpy={np.__version__}, pandas={pd.__version__}, pydicom={pydicom.__version__}"]
    for modality in ("T1", "T2"):
        subset = [r for r in rows if r["modality"] == modality]
        lines += ["", f"=== {modality} ===", f"Statuses: {dict(Counter(r['status'] for r in subset))}"]
        for metric in ("file_count", "frame_count", "unique_slice_count"):
            valid = [r[metric] for r in subset if np.isfinite(r[metric])]
            for threshold in (4, 10, 20):
                count = sum(value < threshold for value in valid)
                pct = f"{100 * count / len(valid):.2f}%" if valid else "unknown"
                lines.append(f"{metric} < {threshold}: {count}/{len(valid)} valid ({pct}); unknown={total-len(valid)}; all CSV studies={total}")
        flags = Counter(flag for row in subset for flag in set(row["flags"]))
        for flag, count in sorted(flags.items()):
            lines.append(f"Flag {flag}: {count}/{total} CSV studies ({100 * count / total:.2f}%)")
        assessed = [r for r in subset if isinstance(r.get("nonuniform_spacing"), bool)]
        count = sum(r["nonuniform_spacing"] for r in assessed)
        lines.append(f"Nonuniform spacing: {count}/{len(assessed)} assessable; unknown={total-len(assessed)}")
        lines += ["", distribution[distribution.modality == modality].to_string(index=False, na_rep="unknown", float_format=lambda x: f"{x:.6g}")]
    lines += ["", "=== STUDY-LEVEL (EITHER T1 OR T2) ==="]
    pairs = list(zip(rows[::2], rows[1::2]))
    for metric in ("file_count", "frame_count", "unique_slice_count"):
        for threshold in (4, 10, 20):
            positive = sum(any(np.isfinite(r[metric]) and r[metric] < threshold for r in pair) for pair in pairs)
            negative = sum(all(np.isfinite(r[metric]) and r[metric] >= threshold for r in pair) for pair in pairs)
            lines.append(f"Either modality {metric} < {threshold}: yes={positive}, no={negative}, unknown={total-positive-negative}; denominator={total}")
    flagged = sum(any(r["flags"] or r["status"] != "selected" for r in pair) for pair in pairs)
    lines.append(f"Studies with any report flag: {flagged}/{total}. Flags include metadata limitations and derived-image labels, not only defects.")
    return "\n".join(lines) + "\n"


def write_csv(rows, path):
    serialized = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                   for key, value in row.items()} for row in rows]
    # Retain headers even if the flagged subset is empty.
    table = pd.DataFrame(serialized) if serialized else pd.DataFrame(columns=["study_id", "modality", "status", "flags", *METRICS])
    table.to_csv(path, index=False, encoding="utf-8-sig", na_rep="unknown", float_format="%.9g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Target study CSV (all rows included)")
    parser.add_argument("--folder-list", required=True, type=Path, help="UTF-8 TXT: one DICOM parent directory per line")
    parser.add_argument("--study-id-column", default="Study_ID", help="CSV ID column, default Study_ID; use STUDY_ID if needed")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory; existing directories are not overwritten")
    args = parser.parse_args()
    table = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.study_id_column not in table:
        parser.error(f"Missing CSV column: {args.study_id_column}")
    ids = table[args.study_id_column].str.strip().tolist()
    if not ids or any(not value or value in (".", "..") or any(c in value for c in '/\\:*?"<>|') for value in ids):
        parser.error("Study IDs must be nonempty plain directory names; CSV must not be empty")
    if len(set(ids)) != len(ids):
        parser.error("Duplicate Study IDs in CSV; resolve duplicates to avoid double weighting")
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
        parser.error(f"Output directory already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    rows = []
    for index, study_id in enumerate(ids, 1):
        pair = inspect_study(study_id, roots)
        rows.extend(pair)
        if index == 1 or index % 100 == 0 or index == len(ids):
            print(f"[{index}/{len(ids)}] {study_id}: " + ", ".join(f"{r['modality']}={r['status']}" for r in pair), flush=True)
    distribution = make_distribution(rows)
    write_csv(rows, output / "selected_series.csv")
    write_csv([r for r in rows if r["flags"] or r["status"] != "selected"], output / "flagged_series.csv")
    distribution.to_csv(output / "distribution_summary.csv", index=False, encoding="utf-8-sig", na_rep="unknown", float_format="%.9g")
    (output / "summary.txt").write_text(make_summary(rows, distribution, args.csv.resolve(), args.folder_list.resolve(), args.study_id_column), encoding="utf-8")
    print(f"Finished: {len(ids)} CSV studies, {len(rows)} modality records. Reports: {output}")


if __name__ == "__main__":
    main()
