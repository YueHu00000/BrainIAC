"""Geometry/statistics functions reused from code/single_study_check (2026-10-02)."""


import json
import numpy as np
import pandas as pd


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



def geometry(ds, frame=None):
    """Get classic or enhanced MR geometry, including shared/per-frame overrides."""
    names = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "SliceThickness")
    result = {name: getattr(ds, name, None) for name in names}
    shared = getattr(ds, "SharedFunctionalGroupsSequence", [])
    groups = list(shared[:1]) + ([frame] if frame is not None else [])
    for group in groups:
        for sequence, keys in (
            ("PlanePositionSequence", names[:1]),
            ("PlaneOrientationSequence", names[1:2]),
            ("PixelMeasuresSequence", names[2:]),
        ):
            items = getattr(group, sequence, [])
            if items:
                for key in keys:
                    result[key] = getattr(items[0], key, result[key])
    return result


def numbers(value, size):
    """Missing or malformed numerical metadata stays unknown."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (ValueError, TypeError):
        return None
    return array if array.size == size and np.isfinite(array).all() else None


def summarize_series(items, include_frame_positions=False):
    result = {metric: np.nan for metric in METRICS}
    flags = []
    result["flags"] = flags
    result["file_count"] = len(items)
    result["file_names"] = [path.name for path, _ in items]
    result["instance_numbers"] = [int(ds.InstanceNumber) if getattr(ds, "InstanceNumber", None) is not None else None
                                  for _, ds in items]
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
            if include_frame_positions:
                result["frame_positions_mm"] = np.asarray(positions).tolist()
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


def json_list(value):
    if value is None or (isinstance(value, str) and value in ("", "unknown")):
        return []
    return json.loads(value) if isinstance(value, str) else value


def combined_status(statuses):
    """Known failure takes precedence; pass requires every check to pass."""
    if "fail" in statuses:
        return "fail"
    return "pass" if all(value == "pass" for value in statuses) else "unknown"


def position_check(row, error_percent):
    result = dict(position_status="unknown", position_reason="unavailable_geometry",
                  position_threshold_mm=np.nan, position_max_error_mm=np.nan,
                  position_max_error_percent=np.nan, position_bad_slices=np.nan,
                  position_assessed_slices=0)
    if row["status"] != "selected":
        result["position_reason"] = "selection_not_completed"
        return result, []
    coverage = row["coverage_mm"]
    if not np.isfinite(coverage) or coverage <= 0:
        result["position_reason"] = "coverage_not_positive"
        return result, []
    result["position_threshold_mm"] = coverage * error_percent / 100
    if not np.isfinite(row["frame_count"]) or not np.isfinite(row["file_count"]):
        return result, []
    if row["frame_count"] != row["file_count"]:
        result["position_reason"] = "multi_frame_conversion_not_modelled"
        return result, []
    positions = np.asarray(json_list(row.get("projected_positions_mm")), dtype=float)
    count = int(row["frame_count"])
    if positions.ndim != 1 or len(positions) != count or count < 2 or not np.isfinite(positions).all():
        result["position_reason"] = "missing_or_incomplete_positions"
        return result, []
    unique_count = row["unique_slice_count"]
    if not np.isfinite(unique_count) or unique_count != count:
        result["position_reason"] = "nonunique_or_unknown_planes"
        return result, []
    # Keep the exported file order. Sorting would conceal ordering-related errors.
    expected = np.linspace(positions[0], positions[-1], count)
    errors = np.abs(positions - expected)
    bad = errors > result["position_threshold_mm"]
    result.update(position_status="fail" if bad.any() else "pass", position_reason="assessed",
                  position_max_error_mm=float(errors.max()),
                  position_max_error_percent=float(errors.max() / coverage * 100),
                  position_bad_slices=int(bad.sum()), position_assessed_slices=count)
    names = json_list(row.get("file_names"))
    instances = json_list(row.get("instance_numbers"))
    slices = [dict(study_id=row["study_id"], modality=row["modality"], slice_index=index,
                   file_name=names[index] if len(names) == count else "unknown",
                   instance_number=instances[index] if len(instances) == count else "unknown",
                   actual_position_mm=float(actual), equal_spacing_position_mm=float(target),
                   error_mm=float(error), error_percent=float(error / coverage * 100),
                   threshold_mm=result["position_threshold_mm"], exceeds_threshold=bool(exceeds))
              for index, (actual, target, error, exceeds) in enumerate(zip(positions, expected, errors, bad))]
    return result, slices


def counts(group):
    total = len(group)
    result = {"n_series": total}
    for check in ("coverage", "position", "screen"):
        for status in ("pass", "fail", "unknown"):
            result[f"{check}_{status}"] = int(group[f"{check}_status"].eq(status).sum())
    result["screen_pass_percent_all"] = 100 * result["screen_pass"] / total if total else np.nan
    assessed = result["screen_pass"] + result["screen_fail"]
    result["screen_pass_percent_classified"] = 100 * result["screen_pass"] / assessed if assessed else np.nan
    result["position_bad_slices"] = int(group.position_bad_slices.sum())
    result["position_assessed_slices"] = int(group.position_assessed_slices.sum())
    result["_abnormal_ids"] = group.loc[group.screen_status.eq("fail"), "study_id"].drop_duplicates().tolist()
    return result


def save_csv(table, path):
    copy = table.copy()
    for column in copy:
        copy[column] = copy[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
    copy.to_csv(path, index=False, encoding="utf-8-sig", na_rep="unknown", float_format="%.9g")


def secondary_checks(results):
    records = []
    for modality in results.modality.drop_duplicates():
        subset = results[results.modality == modality]
        for name in MEASURES:
            statuses = []
            for row in subset.to_dict("records"):
                values = np.asarray([row[f"{name}_{stat}_mm"] for stat in ("min", "median", "max")])
                if row["status"] != "selected" or not np.isfinite(values).all() or (values <= 0).any():
                    statuses.append("unknown")
                else:
                    statuses.append("pass" if np.allclose(values, values[0], atol=1e-5, rtol=1e-4) else "fail")
            records.append(dict(modality=modality, check=f"{name}_within_series_consistency",
                                n_total=len(subset), n_pass=statuses.count("pass"),
                                n_flagged=statuses.count("fail"), n_unknown=statuses.count("unknown"),
                                _abnormal_ids=subset.loc[np.asarray(statuses, dtype=object) == "fail", "study_id"].tolist()))
        for name in ("rows", "columns"):
            known = subset.status.eq("selected") & np.isfinite(subset[name]) & subset[name].gt(0)
            flagged = subset["flags"].map(lambda flags: f"missing_or_varying_{name}" in flags)
            records.append(dict(modality=modality, check=f"{name}_available_and_consistent", n_total=len(subset),
                                n_pass=int((known & ~flagged).sum()), n_flagged=int(flagged.sum()),
                                n_unknown=int((~known & ~flagged).sum()),
                                _abnormal_ids=subset.loc[flagged, "study_id"].tolist()))
        # Other flags remain observations; absence is not an independently validated pass.
        covered = {f"{prefix}_{name}" for name in MEASURES for prefix in ("varying", "missing_or_invalid")}
        covered |= {"missing_or_varying_rows", "missing_or_varying_columns",
                    "file_count_lt20", "frame_count_lt20", "unique_slice_count_lt20"}
        flags = sorted({flag for values in subset["flags"] for flag in values if flag not in covered})
        for flag in flags:
            records.append(dict(modality=modality, check=f"source_flag:{flag}", n_total=len(subset),
                                n_pass=np.nan, n_flagged=int(subset["flags"].map(lambda values: flag in values).sum()),
                                n_unknown=np.nan,
                                _abnormal_ids=subset.loc[subset["flags"].map(lambda values: flag in values), "study_id"].tolist()))
    return pd.DataFrame(records)
