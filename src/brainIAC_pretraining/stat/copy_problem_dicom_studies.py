"""Copy complete study folders selected by description, frame count or coverage."""

import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


FIELDS = ["unique_id", "study_id", "series_instance_uid", "series_number",
          "acquisition_number", "frame_count", "coverage_mm", "series_description",
          "reason", "study_directory"]
COPY_FIELDS = ["study_id", "source_directory", "copied_directory", "problem_series_count", "status", "error"]


def read_selected(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        required = set(FIELDS) - {"reason"}
        if rows and not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}; "
                             "use selected_series.csv from summary_dicom_SeriesInstanceUID.py")
    return rows


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def numeric(value, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or (integer and not number.is_integer()):
        return None
    return int(number) if integer else number


def select_problem_series(rows, frame_count_threshold, coverage_threshold_mm):
    if (frame_count_threshold < 1 or not math.isfinite(coverage_threshold_mm)
            or coverage_threshold_mm < 0):
        raise ValueError("Require frame_count_threshold >= 1 and finite coverage_threshold_mm >= 0")
    problems, unknown, seen, sources = [], [], set(), {}
    for row in rows:
        study, uid = row["study_id"], row["series_instance_uid"]
        if (not study or study in (".", "..") or study.rstrip(" .") != study
                or any(c in study for c in '/\\:*?"<>|')):
            raise ValueError(f"Invalid Study_ID folder name: {study!r}")
        if not uid or uid == "unknown" or row["unique_id"] != f"{study}_{uid}_{int(row['series_number'])}":
            raise ValueError("unique_id must equal {Study_ID}_{SeriesInstanceUID}_{SeriesNumber}")
        if row["unique_id"] in seen:
            raise ValueError(f"Duplicate unique_id: {row['unique_id']}")
        seen.add(row["unique_id"])
        if not row["study_directory"]:
            raise ValueError(f"Missing study_directory: {study}")
        source = Path(row["study_directory"]).resolve()
        if study in sources and sources[study] != source:
            raise ValueError(f"Multiple source directories for Study_ID: {study}")
        sources[study] = source
        frames, coverage = numeric(row["frame_count"], True), numeric(row["coverage_mm"])
        record = {key: row[key] for key in FIELDS if key != "reason"}
        record.update(frame_count=frames if frames is not None else "unknown",
                      coverage_mm=coverage if coverage is not None else "unknown")
        reasons, missing = [], []
        description = row["series_description"].lower()
        if "t1" not in description and "t2" not in description:
            reasons.append("no_t1_t2_in_description")
        if frames is None:
            missing.append("unknown_frame_count")
        elif frames < frame_count_threshold:
            reasons.append("low_frame_count")
        if coverage is None:
            missing.append("unknown_coverage")
        elif coverage < coverage_threshold_mm:
            reasons.append("low_coverage")
        if reasons:
            problems.append(dict(record, reason=";".join(reasons)))
        if missing:
            unknown.append(dict(record, reason=";".join(missing)))
    return problems, unknown


def copy_problem_studies(selected_csv, output_dir, frame_count_threshold, coverage_threshold_mm):
    rows = read_selected(selected_csv)
    problems, unknown = select_problem_series(rows, frame_count_threshold, coverage_threshold_mm)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    for row in rows:
        source = Path(row["study_directory"]).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output and source study directories must not overlap")
    counts = Counter(row["study_id"] for row in problems)
    studies = {}
    for row in problems:
        study = row["study_id"]
        source = Path(row["study_directory"]).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Study directory does not exist: {source}")
        studies[study] = dict(study_id=study, source_directory=str(source),
                              copied_directory=str(output / study), problem_series_count=counts[study],
                              status="pending", error="")
    output.mkdir(parents=True)
    write_csv(output / "problem_series.csv", problems, FIELDS)
    write_csv(output / "unknown_metrics.csv", unknown, FIELDS)
    reports = list(studies.values())
    try:
        for report in reports:
            try:
                # Publish only after the whole study, including nested folders, is copied.
                with TemporaryDirectory(prefix=".copy_", dir=output) as temporary:
                    staged = Path(temporary) / "study"
                    shutil.copytree(report["source_directory"], staged)
                    os.replace(staged, report["copied_directory"])
                report["status"] = "copied"
                print(f"[copied] {report['study_id']}", flush=True)
            except Exception as error:
                report.update(status="copy_failed", error=str(error))
                print(f"[failed] {report['study_id']}: {error}", flush=True)
            write_csv(output / "copied_studies.csv", reports, COPY_FIELDS)
    finally:
        write_csv(output / "copied_studies.csv", reports, COPY_FIELDS)
    print(f"Problem series: {len(problems)}; selected studies: {len(studies)}; "
          f"copy results: {dict(Counter(row['status'] for row in reports))}; "
          f"unknown-metric series: {len(unknown)}")
    return problems, reports, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", required=True, type=Path,
                        help="Raw selected_series.csv from summary_dicom_SeriesInstanceUID.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-count-threshold", required=True, type=int)
    parser.add_argument("--coverage-threshold-mm", required=True, type=float)
    args = parser.parse_args()
    copy_problem_studies(args.selected_csv, args.output_dir,
                         args.frame_count_threshold, args.coverage_threshold_mm)


if __name__ == "__main__":
    main()
