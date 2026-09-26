"""Summarize slice coverage and equal-spacing displacement from selected_series.csv."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC = ("file_count", "frame_count", "unique_slice_count", "coverage_mm", "duplicate_plane_count")
MEASURES = ("pixel_row", "pixel_column", "thickness", "tag_spacing")


def read_selected(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"study_id", "modality", "status", "flags", *NUMERIC}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    if table.empty or table.study_id.str.strip().eq("").any():
        raise ValueError("Input must contain nonempty study IDs")
    if table.duplicated(["study_id", "modality"]).any():
        raise ValueError("Duplicate study/modality rows")
    if any(set(group.modality) != {"T1", "T2"} for _, group in table.groupby("study_id")):
        raise ValueError("Every study must have exactly one T1 and one T2 row, including failures")
    table["flags"] = table["flags"].map(json.loads)
    if not table["flags"].map(lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)).all():
        raise ValueError("flags must contain JSON lists of strings")
    numeric_columns = [*NUMERIC, "rows", "columns", "max_in_plane_shift_mm"]
    numeric_columns += [f"{name}_{stat}_mm" for name in MEASURES for stat in ("min", "median", "max")]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table.get(column, pd.Series(index=table.index, dtype=float)), errors="coerce")
    for key in ("file_count", "frame_count", "unique_slice_count", "duplicate_plane_count"):
        known = table[key].dropna()
        if ((known < 0) | (known % 1 != 0) | ~np.isfinite(known)).any():
            raise ValueError(f"{key} must be a nonnegative integer or unknown")
    return table


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


def analyze(table, coverage_threshold=100.0, error_percent=20.0):
    repeated = table.duplicate_plane_count.gt(0) | table["flags"].map(lambda v: "repeated_slice_plane" in v)
    excluded_ids = set(table.loc[repeated, "study_id"])
    excluded = table[table.study_id.isin(excluded_ids)].copy()
    excluded["exclusion_reason"] = "repeated_slice_plane_in_either_modality"
    results, slices = [], []
    for row in table[~table.study_id.isin(excluded_ids)].to_dict("records"):
        coverage = row["coverage_mm"]
        known = row["status"] == "selected" and np.isfinite(coverage) and coverage >= 0
        row["coverage_status"] = ("fail" if coverage < coverage_threshold else "pass") if known else "unknown"
        position, detail = position_check(row, error_percent)
        row.update(position)
        count_known = np.isfinite(row["unique_slice_count"]) and row["unique_slice_count"] >= 1
        row["screen_status"] = combined_status([row["coverage_status"], row["position_status"],
                                               "pass" if count_known else "unknown"])
        results.append(row)
        slices.extend(detail)
    columns = [*table.columns, "coverage_status", "position_status", "position_reason",
               "position_threshold_mm", "position_max_error_mm", "position_max_error_percent",
               "position_bad_slices", "position_assessed_slices", "screen_status"]
    results = pd.DataFrame(results, columns=columns)
    for column in (*NUMERIC, "rows", "columns", "position_bad_slices", "position_assessed_slices"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    return results, excluded, pd.DataFrame(slices, columns=[
        "study_id", "modality", "slice_index", "file_name", "instance_number", "actual_position_mm",
        "equal_spacing_position_mm", "error_mm", "error_percent", "threshold_mm", "exceeds_threshold"])


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


def count_tables(results):
    exact, thresholds = [], []
    for modality in ("T1", "T2"):
        subset = results[results.modality == modality]
        for count, group in subset.groupby("unique_slice_count", dropna=False, sort=True):
            exact.append(dict(modality=modality, unique_slice_count=count, **counts(group)))
        known = subset[np.isfinite(subset.unique_slice_count)]
        for threshold in sorted({int(n) + 1 for n in known.unique_slice_count}):
            for side, group in (("below", known[known.unique_slice_count < threshold]),
                                ("at_or_above", known[known.unique_slice_count >= threshold])):
                thresholds.append(dict(modality=modality, threshold=threshold, side=side,
                                       unknown_slice_count=len(subset) - len(known), **counts(group)))
    extra = list(counts(results).keys())
    return (pd.DataFrame(exact, columns=["modality", "unique_slice_count", *extra]),
            pd.DataFrame(thresholds, columns=["modality", "threshold", "side", "unknown_slice_count", *extra]))


def study_table(results):
    records = []
    for study_id, group in results.groupby("study_id", sort=False):
        pair = group.set_index("modality")
        records.append(dict(study_id=study_id, T1_status=pair.loc["T1", "screen_status"],
                            T2_status=pair.loc["T2", "screen_status"],
                            study_status=combined_status(group.screen_status.tolist())))
    return pd.DataFrame(records, columns=["study_id", "T1_status", "T2_status", "study_status"])


def secondary_checks(results):
    records = []
    for modality in ("T1", "T2"):
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


def fmt(value):
    return f"{value:.6g}" if np.isfinite(value) else "unknown"


def render_summary(source, results, excluded, studies, exact, secondary, coverage_threshold, error_percent):
    total = (len(results) + len(excluded)) // 2
    lines = ["DICOM T1/T2 SUMMARY V2 - COVERAGE AND SLICE POSITION SCREEN",
             f"Input: {source}", f"Original studies: {total}",
             f"Excluded repeated-plane studies: {excluded.study_id.nunique()}; both modalities removed.",
             f"Main-analysis studies: {len(studies)}; modality records: {len(results)}",
             f"Coverage failure: coverage_mm < {coverage_threshold:g} mm (equality passes).",
             f"Position failure: at least one slice error > {error_percent:g}% of its series coverage (equality passes).",
             "Position model: q_i = p_0 + i*(p_last-p_0)/(N-1), in original exported file order; error=abs(p_i-q_i).",
             "This is a 1D endpoint equal-spacing model, NOT measured post-registration error or a full 3D ITK simulation.",
             "It ignores in-plane shifts. Multi-frame conversion and incomplete geometry remain unknown.",
             "No minimum slice count is imposed: slice count stratifies the two screening checks.",
             "PASS = both checks pass and slice count is known; FAIL = any known failure; otherwise UNKNOWN.",
             "Pass means no failure under these user-defined rules, NOT validated whole-brain/image/model quality.",
             "A study passes only when T1 and T2 pass; either modality failure makes a study fail.",
             "Secondary consistency checks do not change the main screen. Uniform but very coarse spacing may pass.",
             "Percentages use the retained cohort; unknowns are never silently counted as passing.",
             "Embedding/preprocessing success and pixels were not checked.", "", "=== MAIN COHORT ==="]
    if len(studies):
        for status in ("pass", "fail", "unknown"):
            n = int(studies.study_status.eq(status).sum())
            lines.append(f"Study {status}: {n}/{len(studies)} ({100*n/len(studies):.2f}%)")
    else:
        lines.append("No studies remain after exclusions; percentages are unavailable.")
    for modality in ("T1", "T2"):
        subset = results[results.modality == modality]
        summary = counts(subset)
        lines += ["", f"=== {modality}: {len(subset)} retained series ==="]
        for check in ("coverage", "position", "screen"):
            lines.append(f"{check}: pass={summary[check+'_pass']}, fail={summary[check+'_fail']}, unknown={summary[check+'_unknown']}")
        lines.append(f"Screen pass: {fmt(summary['screen_pass_percent_all'])}% of all retained; "
                     f"{fmt(summary['screen_pass_percent_classified'])}% of classified (pass+fail).")
        assessed = summary["position_assessed_slices"]
        bad = summary["position_bad_slices"]
        lines.append(f"Slices above position-error threshold: {bad}/{assessed} assessed slices "
                     f"({fmt(100*bad/assessed if assessed else np.nan)}%); this is slice-weighted.")
        values = subset.unique_slice_count.dropna().to_numpy(dtype=float)
        if len(values):
            names = ("min", "P1", "P5", "P25", "median", "P75", "P95", "P99", "max")
            quantiles = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100], method="linear")
            lines.append("Unique slices: " + ", ".join(f"{key}={fmt(value)}" for key, value in zip(names, quantiles))
                         + f"; valid={len(values)}, unknown={len(subset)-len(values)}")
        else:
            lines.append(f"Unique slices: no valid counts; unknown={len(subset)}")
        for check in ("coverage", "position"):
            failures = subset[subset[f"{check}_status"] == "fail"]
            known = failures.unique_slice_count.dropna()
            if not len(failures):
                lines.append(f"{check}: no observed failures; no failure-derived slice-count boundary.")
            elif not len(known):
                lines.append(f"{check}: {len(failures)} failures, all with unknown slice count.")
            else:
                boundary = int(known.max()) + 1
                below = subset[subset.unique_slice_count < boundary]
                above = subset[subset.unique_slice_count >= boundary]
                lines.append(f"{check} failures: {len(failures)} series; known slice-count range "
                             f"{fmt(known.min())}..{fmt(known.max())}; unknown count={len(failures)-len(known)}.")
                lines.append(f"Observed boundary N < {boundary}: failures={len(known)}/{len(below)} series below; "
                             f"at/above: pass={int(above[f'{check}_status'].eq('pass').sum())}, "
                             f"fail={int(above[f'{check}_status'].eq('fail').sum())}, "
                             f"unknown={int(above[f'{check}_status'].eq('unknown').sum())}.")
                lines.append("This boundary encloses observed failures; it is NOT a learned or validated eligibility threshold.")
        display = exact.loc[exact.modality == modality, ["unique_slice_count", "n_series", "coverage_fail",
                            "position_fail", "screen_pass", "screen_fail", "screen_unknown",
                            "screen_pass_percent_all", "position_bad_slices"]]
        lines += ["", "By exact unique-slice count (overlapping coverage/position failures must not be added):",
                  display.to_string(index=False, na_rep="unknown", float_format=lambda x: f"{x:.4g}")]
    lines += ["", "=== SECONDARY CHECKS (no parameter percentile tables) ===",
              "Within-series consistency uses atol=1e-5, rtol=1e-4; consistency does not establish absolute suitability.",
              "source_flag rows show observations only; a missing flag is not proof of absence of a defect.",
              secondary.drop(columns="_abnormal_ids").to_string(index=False, na_rep="not_assessed", float_format=lambda x: f"{x:.6g}"),
              "", "=== REPEATED-PLANE EXCLUSIONS ==="]
    if excluded.empty:
        lines.append("None.")
    else:
        lines.append(excluded[["study_id", "modality", "status", "duplicate_plane_count"]].to_string(index=False))
    lines += ["", "Detailed files: series_quality_v2.csv, study_quality_v2.csv, slice_position_deviations.csv,",
              "slice_count_analysis.csv, slice_count_thresholds.csv, secondary_checks.csv, excluded_repeated_studies.csv.",
              "Two matching CSV sets: csv/ has no source study IDs; csv_with_abnormal_study_ids/ adds ONLY a final abnormal_study_ids column.",
              "Study indices are 1-based input-order surrogates. Blank abnormal IDs mean no identified failure in that row, not necessarily a pass.",
              "This TXT retains source paths and excluded study IDs for internal review.",
              "Use inspect_repeated_slice_study.py on excluded study IDs to investigate source DICOM headers."]
    return "\n".join(lines) + "\n"


def save_csv(table, path):
    copy = table.copy()
    for column in copy:
        copy[column] = copy[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
    copy.to_csv(path, index=False, encoding="utf-8-sig", na_rep="unknown", float_format="%.9g")


def write_csv_pair(table, ids, output, name):
    """The two exports differ only by their final column; no normal IDs are added."""
    plain = output / "csv"
    identified = output / "csv_with_abnormal_study_ids"
    plain.mkdir(exist_ok=True)
    identified.mkdir(exist_ok=True)
    save_csv(table, plain / f"{name}.csv")
    annotated = table.copy()
    annotated["abnormal_study_ids"] = [list(dict.fromkeys(values)) if values else "" for values in ids]
    save_csv(annotated, identified / f"{name}.csv")


def public_series(table, indices):
    # Explicit fields avoid copying source paths, filenames, UIDs or free-text errors.
    fields = ["modality", "status", "series_number", *NUMERIC, "rows", "columns", "flags",
              "gap_min_mm", "gap_median_mm", "gap_max_mm", "max_in_plane_shift_mm"]
    fields += [f"{name}_{stat}_mm" for name in MEASURES for stat in ("min", "median", "max")]
    fields += ["coverage_status", "position_status", "position_reason", "position_threshold_mm",
               "position_max_error_mm", "position_max_error_percent", "position_bad_slices",
               "position_assessed_slices", "screen_status", "exclusion_reason"]
    result = table[[field for field in fields if field in table]].copy()
    result.insert(0, "study_index", table.study_id.map(indices))
    return result


def export_reports(table, results, excluded, slices, studies, exact, thresholds, secondary, output):
    indices = {study: i + 1 for i, study in enumerate(table.study_id.drop_duplicates())}
    for name, data, failures in (
            ("series_quality_v2", results, results.screen_status.eq("fail")),
            ("excluded_repeated_studies", excluded, pd.Series(True, index=excluded.index))):
        ids = [[study] if failed else [] for study, failed in zip(data.study_id, failures)]
        write_csv_pair(public_series(data, indices), ids, output, name)
    for name, data, failures in (
            ("study_quality_v2", studies, studies.study_status.eq("fail")),
            ("slice_position_deviations", slices, slices.exceeds_threshold.eq(True))):
        public = data.drop(columns=[column for column in ("study_id", "file_name") if column in data]).copy()
        public.insert(0, "study_index", data.study_id.map(indices))
        ids = [[study] if failed else [] for study, failed in zip(data.study_id, failures)]
        write_csv_pair(public, ids, output, name)
    for name, data in (("slice_count_analysis", exact), ("slice_count_thresholds", thresholds),
                       ("secondary_checks", secondary)):
        write_csv_pair(data.drop(columns="_abnormal_ids"), data["_abnormal_ids"], output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing results are not overwritten")
    parser.add_argument("--coverage-threshold-mm", type=float, default=100.0)
    parser.add_argument("--position-error-percent", type=float, default=20.0)
    args = parser.parse_args()
    if not np.isfinite(args.coverage_threshold_mm) or args.coverage_threshold_mm <= 0:
        parser.error("coverage threshold must be finite and positive")
    if not np.isfinite(args.position_error_percent) or not 0 <= args.position_error_percent <= 100:
        parser.error("position error percent must be between 0 and 100")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    table = read_selected(args.selected_csv)
    results, excluded, slices = analyze(table, args.coverage_threshold_mm, args.position_error_percent)
    exact, thresholds = count_tables(results)
    studies = study_table(results)
    secondary = secondary_checks(results)
    report = render_summary(args.selected_csv.resolve(), results, excluded, studies, exact, secondary,
                            args.coverage_threshold_mm, args.position_error_percent)
    args.output_dir.mkdir(parents=True)
    export_reports(table, results, excluded, slices, studies, exact, thresholds, secondary, args.output_dir)
    (args.output_dir / "summary_v2.txt").write_text(report, encoding="utf-8")
    print(f"Finished: {len(studies)} retained studies; {excluded.study_id.nunique()} excluded. {args.output_dir}")


if __name__ == "__main__":
    main()
