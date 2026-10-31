"""Select paired T1/T2 series and build three matched classification manifests."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CBI_WMD_predict.common import (
    FOLDS, MODALITIES, load_features, read_csv, write_csv, write_json,
)


SERIES_FIELDS = (
    "unique_id", "series_instance_uid", "series_number", "acquisition_number",
    "series_description", "frame_count", "coverage_mm",
)
FIELDS = ["patient_id", "study_id", "site_id", "CBI", "WMD", "embedding_path", "modality"] + [
    f"{modality}_{field}" for modality in ("t1", "t2") for field in SERIES_FIELDS
]
AUDIT_FIELDS = ["study_id", *SERIES_FIELDS, "modality", "eligible", "selected", "reason"]
SITES = set().union(*(set(sites) for fold in FOLDS.values() for sites in fold.values()))


def index_rows(rows, key, source):
    result = {}
    for row in rows:
        value = row[key].strip()
        if not value or value in result:
            raise ValueError(f"{source}: empty or duplicate {key}: {value!r}")
        result[value] = row
    return result


def integer(value, name):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer: {value!r}")
    return int(number)


def acquisition_value(value):
    # Upstream retains series without AcquisitionNumber as "unknown".
    return "unknown" if value.strip().lower() in ("", "unknown") else integer(value, "AcquisitionNumber")


def label(value, name):
    value = value.strip()
    if not value or value.lower() == "nan":
        return ""
    number = integer(value, name)
    if number not in (0, 1):
        raise ValueError(f"{name} must be 0, 1 or blank: {value!r}")
    return str(number)


def read_labels(args):
    columns = [args.study_id_column, args.patient_id_column, args.site_id_column,
               args.cbi_column, args.wmd_column]
    rows = read_csv(args.labels_csv, columns)
    result, patients = {}, set()
    for source in rows:
        study = source[args.study_id_column].strip()
        patient = source[args.patient_id_column].strip()
        if not study or not patient or study in result:
            raise ValueError(f"Labels must have a nonempty, unique study ID: {study!r}")
        if patient in patients:
            raise ValueError(f"Multiple studies for patient {patient!r}; provide index-study labels only")
        if study in (".", "..") or "/" in study or "\\" in study:
            raise ValueError(f"Study ID is not a plain filename: {study!r}")
        patients.add(patient)
        result[study] = dict(
            patient_id=patient, study_id=study,
            site_id=integer(source[args.site_id_column], "SiteID"),
            CBI=label(source[args.cbi_column], "CBI"),
            WMD=label(source[args.wmd_column], "WMD"),
        )
    return result


def candidates(args):
    manifest = index_rows(read_csv(args.manifest, [
        "unique_id", "study_id", "series_instance_uid", "series_number", "acquisition_number",
    ]), "unique_id", "manifest")
    selected = index_rows(read_csv(args.selected_csv, [
        "unique_id", "study_id", "series_instance_uid", "series_number", "series_description",
    ]), "unique_id", "selected CSV")
    quality = index_rows(read_csv(args.quality_csv, [
        "unique_id", "frame_count", "coverage_mm",
    ]), "unique_id", "quality CSV")
    final = index_rows(read_csv(args.final_csv, ["unique_id"]), "unique_id", "final CSV")
    if not final:
        raise ValueError("final CSV contains no series")
    grouped, audit = defaultdict(lambda: defaultdict(list)), []
    for uid in final:
        if uid not in manifest or uid not in selected:
            raise ValueError(f"Final series lacks matching manifest/selected metadata: {uid}")
        source, metadata = manifest[uid], selected[uid]
        for field in ("study_id", "series_instance_uid"):
            if not source[field].strip() or source[field].strip() != metadata[field].strip():
                raise ValueError(f"{uid}: manifest/selected {field} mismatch")
        number = integer(source["series_number"], "SeriesNumber")
        if number != integer(metadata["series_number"], "selected SeriesNumber"):
            raise ValueError(f"{uid}: manifest/selected SeriesNumber mismatch")
        acquisition = acquisition_value(source["acquisition_number"])
        if "acquisition_number" in metadata and acquisition != acquisition_value(
                metadata["acquisition_number"]):
            raise ValueError(f"{uid}: manifest/selected AcquisitionNumber mismatch")
        row = dict(
            study_id=source["study_id"].strip(), unique_id=uid,
            series_instance_uid=source["series_instance_uid"].strip(), series_number=number,
            acquisition_number=acquisition,
            series_description=metadata["series_description"].strip(),
            frame_count="", coverage_mm="", modality="", eligible=0, selected=0, reason="",
        )
        reasons = []
        description = row["series_description"].lower()
        hits = [modality for modality in ("T1", "T2") if modality.lower() in description]
        if len(hits) != 1:
            reasons.append("ambiguous_description" if hits else "no_t1_t2_in_description")
        else:
            row["modality"] = hits[0]
        if uid not in quality:
            reasons.append("missing_quality")
        else:
            item = quality[uid]
            if ("series_description" in item and
                    item["series_description"].strip() != row["series_description"]):
                raise ValueError(f"{uid}: quality/selected series_description mismatch")
            row.update(frame_count=item["frame_count"], coverage_mm=item["coverage_mm"])
            try:
                frames = integer(item["frame_count"], "frame_count")
                coverage = float(item["coverage_mm"])
                if not math.isfinite(coverage):
                    raise ValueError("coverage is not finite")
                row.update(frame_count=frames, coverage_mm=coverage)
                if frames < 20:
                    reasons.append("low_frame_count")
                if frames > 40:
                    reasons.append("high_frame_count")
                if coverage < 100:
                    reasons.append("low_coverage")
            except (ValueError, TypeError):
                reasons.append("invalid_quality")
        row["reason"] = ";".join(reasons)
        if not reasons:
            row["eligible"] = 1
            grouped[row["study_id"]][row["modality"]].append(row)
        audit.append(row)
    return grouped, audit


def select_series(options, modality):
    if not options:
        return None, f"missing_{modality.lower()}"
    largest = max(row["series_number"] for row in options)
    winners = [row for row in options if row["series_number"] == largest]
    for row in options:
        row["reason"] = "not_highest_series_number"
    if len(winners) != 1:
        for row in winners:
            row["reason"] = "ambiguous_max_series_number"
        return None, f"ambiguous_{modality.lower()}_max_series_number"
    winners[0].update(selected=1, reason="selected")
    return winners[0], ""


def build(args):
    labels = read_labels(args)
    grouped, audit = candidates(args)
    paired, excluded = [], []
    for study, fields in labels.items():
        reasons, picks = [], {}
        for modality in ("T1", "T2"):
            pick, reason = select_series(grouped[study][modality], modality)
            picks[modality] = pick
            if reason:
                reasons.append(reason)
        if fields["site_id"] not in SITES:
            reasons.append("site_not_in_harvey_folds")
        row = dict(fields, embedding_path=str((args.embedding_dir / f"{study}.npz").resolve()),
                   modality="T1T2")
        if not reasons:
            for modality, pick in picks.items():
                row.update({f"{modality.lower()}_{field}": pick[field] for field in SERIES_FIELDS})
            try:
                load_features([row], "T1T2")
            except (OSError, ValueError, KeyError, TypeError) as error:
                reasons.append(f"selected_embedding_unavailable: {error}")
        if reasons:
            excluded.append(dict(fields, reason=";".join(reasons)))
        else:
            paired.append(row)
    for study in sorted({row["study_id"] for row in audit} - labels.keys()):
        excluded.append(dict(patient_id="", study_id=study, site_id="", CBI="", WMD="",
                             reason="missing_labels_row"))
    if not paired:
        raise ValueError(f"No paired eligible patients; {len(excluded)} studies excluded")

    folds = [dict(patient_id=row["patient_id"], study_id=row["study_id"], fold=fold, split=split)
             for row in paired for fold, groups in FOLDS.items()
             for split, sites in groups.items() if row["site_id"] in sites]
    counts = []
    for site in sorted({row["site_id"] for row in labels.values()}):
        included = [row for row in paired if row["site_id"] == site]
        omitted = [row for row in excluded if row["site_id"] == site]
        series = [row for row in audit if row["study_id"] in labels
                  and labels[row["study_id"]]["site_id"] == site]
        for target in ("CBI", "WMD"):
            values = Counter(row[target] for row in included)
            counts.append(dict(site_id=site, target=target, label_studies=sum(
                row["site_id"] == site for row in labels.values()), paired_studies=len(included),
                excluded_studies=len(omitted), candidate_series=len(series),
                eligible_series=sum(row["eligible"] for row in series),
                selected_series=sum(row["selected"] for row in series),
                positive=values["1"], negative=values["0"],
                missing=values[""]))
    config = dict(
        inputs={name: str(getattr(args, name).resolve()) for name in (
            "manifest", "final_csv", "selected_csv", "quality_csv", "labels_csv", "embedding_dir")},
        columns={name: getattr(args, name) for name in (
            "patient_id_column", "study_id_column", "site_id_column", "cbi_column", "wmd_column")},
        coverage_mm_min=100, frame_count_min=20, frame_count_max=40,
        selection="highest_numeric_series_number_after_quality_filter",
        cohort="paired_T1_T2_identical_patients_for_all_modalities",
        feature="final_embed", representation_dim=768, folds=FOLDS,
        patients=len(paired), excluded_studies=len(excluded),
    )
    embedding_config = args.embedding_dir / "embedding_config.json"
    if embedding_config.exists():
        config["embedding_config"] = json.loads(embedding_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for modality in MODALITIES:
        rows = []
        for source in paired:
            row = dict(source, modality=modality)
            for prefix in ("t1", "t2"):
                if modality != "T1T2" and prefix.upper() != modality:
                    row.update({f"{prefix}_{field}": "" for field in SERIES_FIELDS})
            rows.append(row)
        write_csv(args.output_dir / f"manifest_{modality.lower()}.csv", rows, FIELDS)
    write_csv(args.output_dir / "folds.csv", folds, ["patient_id", "study_id", "fold", "split"])
    write_csv(args.output_dir / "series_selection.csv", audit, AUDIT_FIELDS)
    write_csv(args.output_dir / "excluded_studies.csv", excluded,
              ["patient_id", "study_id", "site_id", "CBI", "WMD", "reason"])
    write_csv(args.output_dir / "cohort_counts.csv", counts, [
        "site_id", "target", "label_studies", "paired_studies", "excluded_studies",
        "candidate_series", "eligible_series", "selected_series", "positive", "negative", "missing",
    ])
    write_json(args.output_dir / "selection_config.json", config)
    print(f"Paired patients: {len(paired)}; excluded studies: {len(excluded)}; output: {args.output_dir}")
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "final-csv", "selected-csv", "quality-csv", "labels-csv",
                 "embedding-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--patient-id-column", default="Study_ID",
                        help="Patient identity column (default: Study_ID, confirmed for this cohort)")
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--site-id-column", default="SiteID")
    parser.add_argument("--cbi-column", default="idCBI")
    parser.add_argument("--wmd-column", default="idWMD")
    return parser.parse_args(argv)


def main(argv=None):
    return build(parse_args(argv))


if __name__ == "__main__":
    main()
