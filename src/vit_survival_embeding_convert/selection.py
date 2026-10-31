"""Reconstruct the legacy maximum-SeriesNumber choice before applying quality cuts."""

from collections import defaultdict
import json
import math

from .common import integer, plain_id, read_csv


REQUIRED = (
    "unique_id", "study_id", "series_instance_uid", "series_number", "series_description",
    "series_descriptions", "acquisition_number", "acquisition_numbers", "source_file_count",
    "discarded_file_count", "missing_acquisition_file_count", "file_count", "frame_count",
    "coverage_mm", "status", "flags",
)
COUNT_FIELDS = ("source_file_count", "discarded_file_count", "missing_acquisition_file_count", "file_count")
CORRECTED_STUDY = "R01_Study_002904"


def json_list(value, field, element_type):
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(type(item) is not element_type for item in parsed):
        raise ValueError(f"{field} must be a JSON list of {element_type.__name__}")
    return parsed


def read_statistics(path):
    """Read the full first-stage UID statistics, retaining acquisition evidence.

    Invalid identity/trust metadata is a table error. Invalid raw quality values
    remain attached to the row, so they cannot cause a lower series to be chosen.
    """
    groups, seen = defaultdict(list), set()
    for source in read_csv(path, REQUIRED):
        row = dict(source)
        study, uid = source["study_id"].strip(), source["series_instance_uid"].strip()
        plain_id(study)
        plain_id(uid)
        if uid.lower() == "unknown":
            raise ValueError("Statistics require an actual SeriesInstanceUID")
        number = integer(source["series_number"], "SeriesNumber")
        expected = f"{study}_{uid}_{number}"
        if source["unique_id"] != expected or expected in seen:
            raise ValueError(f"Invalid or duplicate unique_id: {source['unique_id']!r}; expected {expected}")
        seen.add(expected)
        row.update(study_id=study, series_instance_uid=uid, series_number=number,
                   series_description=source["series_description"].strip())
        for field in ("series_descriptions", "flags"):
            row[field] = json_list(source[field], field, str)
        row["acquisition_numbers"] = json_list(source["acquisition_numbers"], "acquisition_numbers", int)
        acquisition = source["acquisition_number"].strip()
        row["acquisition_number"] = (
            "unknown" if acquisition.lower() in ("", "unknown")
            else integer(acquisition, "AcquisitionNumber")
        )
        for field in COUNT_FIELDS:
            row[field] = integer(source[field], field)
        try:
            row["frame_count"] = integer(source["frame_count"], "frame_count")
        except (TypeError, ValueError):
            row["frame_count"] = None
        try:
            row["coverage_mm"] = float(source["coverage_mm"])
        except (TypeError, ValueError):
            row["coverage_mm"] = math.nan
        groups[study].append(row)
    return dict(groups)


def acquisition_reasons(row, corrected):
    known = row["acquisition_numbers"]
    chosen = row["acquisition_number"]
    source, kept, discarded, missing = (row[field] for field in (
        "source_file_count", "file_count", "discarded_file_count", "missing_acquisition_file_count"))
    if (source < 1 or kept < 1 or discarded < 0 or missing < 0 or missing > source
            or source != kept + discarded or len(known) != len(set(known))):
        return ["inconsistent_source_counts_or_acquisitions"]
    if known:
        if (chosen != max(known) or discarded < missing + len(known) - 1
                or (len(known) == 1 and discarded != missing)):
            return ["inconsistent_acquisition_selection"]
    elif chosen != "unknown" or missing != source or discarded:
        return ["inconsistent_unknown_acquisition"]
    if corrected:
        if chosen != 2 or not known or not set(known).issubset({1, 2}) or missing:
            return ["corrected_T2_requires_acquisition_2"]
        return []
    if len(known) > 1 or discarded:
        return ["legacy_acquisitions_not_preserved"]
    return []


def chosen_reasons(row, modality):
    reasons = []
    descriptions = row["series_descriptions"]
    if len(descriptions) != 1:
        reasons.append("multiple_or_missing_series_descriptions")
    elif descriptions[0].strip() != row["series_description"]:
        reasons.append("series_description_mismatch")
    description = row["series_description"].lower()
    if "t1" in description and "t2" in description:
        reasons.append("ambiguous_description")
    if row["status"] != "selected":
        reasons.append("status_not_selected")
    if "missing_instance_number" in row["flags"]:
        reasons.append("missing_instance_number")
    frames, coverage = row["frame_count"], row["coverage_mm"]
    if frames is None or not math.isfinite(coverage):
        reasons.append("invalid_quality")
    else:
        if frames < 20:
            reasons.append("low_frame_count")
        if frames > 40:
            reasons.append("high_frame_count")
        if coverage < 100:
            reasons.append("low_coverage")
    corrected = row["study_id"] == CORRECTED_STUDY and modality == "T2"
    reasons.extend(acquisition_reasons(row, corrected))
    return reasons


def select_study(rows):
    """Return actual legacy choices and rejection reasons; never substitute a loser.

    Chosen rows remain in picks when quality/trust fails, for exclusion audits.
    A tie or incomplete study cannot establish a choice and returns None there.
    """
    picks, reasons = {"T1": None, "T2": None}, []
    if any("partial_study_scan" in row["flags"] for row in rows):
        return picks, ["partial_study_scan"]
    for modality in picks:
        candidates = [row for row in rows if modality.lower() in row["series_description"].lower()]
        if not candidates:
            reasons.append(f"{modality}:missing_candidate")
            continue
        maximum = max(row["series_number"] for row in candidates)
        winners = [row for row in candidates if row["series_number"] == maximum]
        if len(winners) != 1:
            reasons.append(f"{modality}:ambiguous_max_series_number")
            continue
        chosen = dict(winners[0])
        corrected = chosen["study_id"] == CORRECTED_STUDY and modality == "T2"
        chosen["selection_basis"] = "corrected_acquisition_2" if corrected else "inferred_harvey"
        picks[modality] = chosen
        reasons.extend(f"{modality}:{reason}" for reason in chosen_reasons(chosen, modality))
    if all(picks.values()) and picks["T1"]["unique_id"] == picks["T2"]["unique_id"]:
        reasons.append("same_series_for_T1_T2")
    return picks, reasons
