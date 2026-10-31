"""Convert existing paired NORM-A CLS embeddings to the classifier NPZ schema."""

import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vit_survival_embeding_convert import artifacts
    from vit_survival_embeding_convert.common import (
        BLOCK_NUMBERS, MANIFEST_FIELDS, MODALITIES, SERIES_FIELDS, SITES,
        integer, plain_id, read_csv, write_csv, write_json,
    )
    from vit_survival_embeding_convert.selection import read_statistics, select_study
else:
    from . import artifacts
    from .common import (
        BLOCK_NUMBERS, MANIFEST_FIELDS, MODALITIES, SERIES_FIELDS, SITES,
        integer, plain_id, read_csv, write_csv, write_json,
    )
    from .selection import read_statistics, select_study


SHARED_KEYS = ("normalization_id", "normalization_provenance", "weights_id",
               "transform_id", "vit_implementation", "single_channel_adapter")
MAPPING_FIELDS = ["study_id", "status", "reason", "source_embedding_path"] + [
    f"{prefix}_{field}" for prefix in ("t1", "t2")
    for field in (*SERIES_FIELDS, "selection_basis")
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-results-dir", type=Path, required=True,
                        help="Old results root containing embeddings/.")
    parser.add_argument("--legacy-embedding-dir", type=Path,
                        help="Optional exact embedding subtree; useful to disambiguate duplicates.")
    parser.add_argument("--selected-csv", type=Path, required=True,
                        help="Full first-stage selected_series.csv, before anonymous/quality filtering.")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A NEW directory, including in audit-only mode.")
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--site-id-column", default="SiteID")
    parser.add_argument("--cbi-column", default="idCBI")
    parser.add_argument("--wmd-column", default="idWMD")
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate vectors and write reports/config only; no NPZ or manifests.")
    return parser.parse_args(argv)


def read_labels(args):
    columns = (args.study_id_column, args.site_id_column, args.cbi_column, args.wmd_column)
    labels = {}
    for row in read_csv(args.labels_csv, columns):
        study = plain_id(row[args.study_id_column])
        if study in labels:
            raise ValueError(f"Duplicate Study_ID in labels: {study}")
        label = dict(patient_id=study, study_id=study,
                     site_id=integer(row[args.site_id_column], "SiteID"))
        for target, column in (("CBI", args.cbi_column), ("WMD", args.wmd_column)):
            value = row[column]
            if value.lower() in ("", "nan", "na", "null"):
                label[target] = ""
            else:
                label[target] = integer(value, target)
                if label[target] not in (0, 1):
                    raise ValueError(f"{study}: {target} must be 0, 1 or missing")
        labels[study] = label
    return labels


def picked_fields(picks):
    return {f"{modality.lower()}_{field}": row[field]
            for modality, row in picks.items() if row is not None
            for field in (*SERIES_FIELDS, "selection_basis")}


def write_embedding(path, picks, blocks, final):
    records = {}
    for index, modality in enumerate(("T1", "T2")):
        row = picks[modality]
        description = row["series_description"]
        record = np.empty((), dtype=[("series_description", f"U{max(1, len(description))}"),
                                     ("block_embed", "<f4", (4, 768)),
                                     ("final_embed", "<f4", (768,))])
        record["series_description"] = description
        record["block_embed"] = blocks[index]
        record["final_embed"] = final[index]
        records[row["unique_id"]] = record
    # Exclusive creation prevents accidental replacement of an existing artifact.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **records)


def run(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    labels = read_labels(args)
    statistics = read_statistics(args.selected_csv)
    sources, source_issues = artifacts.index_embeddings(
        args.legacy_results_dir, args.legacy_embedding_dir)
    mapping, candidates, shared = [], [], None
    for study, label in sorted(labels.items()):
        picks, reasons = select_study(statistics.get(study, []))
        if label["site_id"] not in SITES:
            reasons.append("site_not_in_harvey_splits")
        paths = sources.get(study, [])
        if len(paths) != 1:
            reasons.append("missing_norm_a_embedding" if not paths else "ambiguous_norm_a_embedding")
        audit = dict(study_id=study, status="skipped", reason=";".join(reasons),
                     source_embedding_path=";".join(map(str, paths)), **picked_fields(picks))
        mapping.append(audit)
        if reasons:
            continue
        metadata = artifacts.read_metadata(paths[0])
        signature = {key: metadata.get(key) for key in SHARED_KEYS}
        if shared is None:
            shared = signature
        elif signature != shared:
            raise ValueError(f"{study}: mixed encoder or NORM-A parameters; use one consistent source cohort")
        candidates.append((label, picks, paths[0], audit))
    if not candidates and not args.audit_only:
        raise ValueError("No eligible paired studies; use --audit-only to write exclusion reports")

    output.mkdir(parents=True, exist_ok=False)
    embeddings = output / "embeddings"
    embeddings.mkdir()
    manifests = []
    with (output / "source_metadata.jsonl").open("w", encoding="utf-8") as provenance:
        for label, picks, source, audit in candidates:
            try:
                blocks, final, metadata = artifacts.load_embedding(source, label["study_id"])
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
                audit["reason"] = f"invalid_embedding:{error}"
                continue
            destination = embeddings / f"{label['study_id']}.npz"
            if not args.audit_only:
                write_embedding(destination, picks, blocks, final)
            audit.update(status="eligible" if args.audit_only else "converted", reason="")
            manifests.append(dict(label, embedding_path=str(destination), **picked_fields(picks)))
            per_source = {key: value for key, value in metadata.items() if key not in SHARED_KEYS}
            provenance.write(json.dumps(dict(study_id=label["study_id"], source_path=str(source),
                                            metadata=per_source,
                                            shared_metadata="embeddings/embedding_config.json"),
                                        ensure_ascii=False, allow_nan=False) + "\n")

    config = dict(
        format="vit_survival_to_brainiac_v1", block_numbers=list(BLOCK_NUMBERS),
        block_embed_shape=[4, 768], final_embed_shape=[768],
        pooling="arithmetic mean over all stored slices, separately for T1 and T2",
        source_model_and_normalization=shared,
        normalization_note="Preserved legacy NORM-A fit cohort; not refitted for Harvey folds.",
        selection_note="Infer legacy maximum SeriesNumber before raw quality filtering; no fallback.",
        quality=dict(min_coverage_mm=100, min_frame_count=20, max_frame_count=40),
        confirmed_correction=dict(study_id="R01_Study_002904", modality="T2", acquisition_number=2),
        audit_only=args.audit_only,
    )
    write_json(embeddings / "embedding_config.json", config)
    if not args.audit_only:
        folder = output / "manifests"
        folder.mkdir()
        for modality in MODALITIES:
            write_csv(folder / f"manifest_{modality}.csv",
                      [dict(row, modality=modality) for row in manifests], MANIFEST_FIELDS)
    write_csv(output / "legacy_series_mapping.csv", mapping, MAPPING_FIELDS)
    write_csv(output / "embedding_source_issues.csv", source_issues, ["path", "study_id", "reason"])
    skipped = [row for row in mapping if row["status"] == "skipped"]
    summary = dict(
        audit_only=args.audit_only, status="complete" if manifests else "no_eligible_studies",
        counts=dict(label_studies=len(labels), eligible_studies=len(manifests),
                    converted_studies=0 if args.audit_only else len(manifests),
                    skipped_studies=len(skipped), indexed_norm_a_studies=len(sources),
                    embedding_source_issues=len(source_issues)),
        skip_reasons=dict(Counter(reason for row in skipped for reason in row["reason"].split(";"))),
        inputs=dict(legacy_results_dir=str(args.legacy_results_dir.resolve()),
                    legacy_embedding_dir=str(args.legacy_embedding_dir.resolve()) if args.legacy_embedding_dir else None,
                    selected_csv=str(args.selected_csv.resolve()), labels_csv=str(args.labels_csv.resolve())),
        output_dir=str(output),
    )
    write_json(output / "conversion_summary.json", summary)
    return summary


def main():
    args = parse_args()
    try:
        summary = run(args)
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["counts"]["eligible_studies"] or args.audit_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
