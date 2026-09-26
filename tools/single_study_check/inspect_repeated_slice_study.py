"""Inspect repeated planes in the originally selected T1/T2 groups of one study."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from inspect_dicom_t1 import geometry
from summarize_dicom_t1t2 import POSITION_TOL_MM, inspect_study, numbers
from summarize_dicom_t1t2_v2 import public_series, write_csv_pair


FIELDS = ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID", "SOPInstanceUID",
          "InstanceNumber", "SeriesNumber", "SeriesDescription", "AcquisitionNumber", "EchoNumbers",
          "EchoTime", "TemporalPositionIdentifier", "AcquisitionTime", "ContentTime", "ImageType",
          "Rows", "Columns", "NumberOfFrames")


def extract_frames(row, directory):
    """Retain file/frame order and metadata that may distinguish repeated planes."""
    records = []
    for filename in row.get("file_names", []):
        ds = pydicom.dcmread(directory / filename, stop_before_pixels=True)
        count = numbers(getattr(ds, "NumberOfFrames", 1), 1)
        if count is None or count[0] < 1 or count[0] != int(count[0]):
            # The original selected-series row retains invalid_frame_count.
            continue
        count = int(count[0])
        per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
        usable = len(per_frame) == count
        for index in range(count):
            frame = per_frame[index] if usable else None
            geo = geometry(ds, frame)
            if (count > 1 or per_frame) and not usable:
                geo["ImagePositionPatient"] = None
            if count > 1 and usable:
                plane = getattr(frame, "PlanePositionSequence", [])
                geo["ImagePositionPatient"] = getattr(plane[0], "ImagePositionPatient", None) if plane else None
            record = dict(study_id=row["study_id"], modality=row["modality"],
                          file_name=filename, frame_index_in_file=index, order_index=len(records))
            record.update({field: str(getattr(ds, field, "unknown")) for field in FIELDS})
            record.update({field: str(value) if value is not None else "unknown" for field, value in geo.items()})
            record["SpacingBetweenSlices"] = str(getattr(ds, "SpacingBetweenSlices", "unknown"))
            groups = list(getattr(ds, "SharedFunctionalGroupsSequence", [])[:1]) + ([frame] if frame is not None else [])
            for group in groups:
                for sequence, fields in (("FrameContentSequence", ("TemporalPositionIndex", "StackID", "InStackPositionNumber", "DimensionIndexValues")),
                                         ("MREchoSequence", ("EffectiveEchoTime",)),
                                         ("PixelMeasuresSequence", ("SpacingBetweenSlices",))):
                    items = getattr(group, sequence, [])
                    if items:
                        for field in fields:
                            if hasattr(items[0], field):
                                record[field] = str(getattr(items[0], field))
            records.append(record)
    projected = row.get("projected_positions_mm", [])
    for record in records:
        record.update(projected_position_mm=np.nan, plane_group=np.nan, plane_group_size=np.nan,
                      repeated_plane="unknown")
    if len(projected) != len(records) or not records:
        return records, []
    groups, representative = [], None
    for index in np.argsort(projected, kind="stable"):
        value = projected[index]
        if representative is None or value - representative > POSITION_TOL_MM:
            groups.append([])
            representative = value
        groups[-1].append(int(index))
    summaries = []
    for group_id, indices in enumerate(groups):
        for index in indices:
            records[index].update(projected_position_mm=projected[index], plane_group=group_id,
                                  plane_group_size=len(indices), repeated_plane=len(indices) > 1)
        if len(indices) < 2:
            continue
        members = [records[index] for index in indices]
        summary = dict(study_id=row["study_id"], modality=row["modality"], plane_group=group_id,
                       representative_position_mm=min(projected[index] for index in indices),
                       n_frames=len(indices), excess_frames=len(indices)-1,
                       file_names=[member["file_name"] for member in members],
                       order_indices=indices)
        for field in ("SOPInstanceUID", "SeriesInstanceUID", "InstanceNumber", "EchoNumbers", "EchoTime",
                      "AcquisitionNumber", "TemporalPositionIdentifier", "TemporalPositionIndex", "EffectiveEchoTime",
                      "StackID", "InStackPositionNumber", "DimensionIndexValues", "AcquisitionTime", "ImageType"):
            summary[field] = sorted({member.get(field, "unknown") for member in members})
        summaries.append(summary)
    return records, summaries


def build_report(directory, study_id):
    # Read the same direct-child group selected by the cohort/conversion rule.
    rows = inspect_study(study_id, [directory.parent])
    frames, duplicates = [], []
    lines = ["REPEATED SLICE PLANE INVESTIGATION", f"Study: {study_id}", f"Directory: {directory}",
             "Selection follows the original T1/T2 maximum-SeriesNumber rule; no files are deleted or regrouped.",
             "Header-only inspection. Repeated plane does not establish identical pixels or a defective acquisition.",
             "Frame indices and order indices are zero-based. Plane groups use the cohort's 0.01 mm tolerance.",
             "UID/echo/time differences are clues; the root cause may remain undetermined."]
    for row in rows:
        lines += ["", f"=== {row['modality']} ===", f"Selection status: {row['status']}",
                  f"Flags: {', '.join(row['flags']) or 'none'}"]
        if row["status"] != "selected":
            lines.append(f"Details: {row.get('error', 'no selected group')}")
            continue
        records, groups = extract_frames(row, directory)
        frames.extend(records)
        duplicates.extend(groups)
        lines += [f"Series: {row['series_number']} / {row['series_description']}",
                  f"Files={row['file_count']}; frames={row['frame_count']}; unique planes={row['unique_slice_count']}; "
                  f"excess frames={row['duplicate_plane_count']}",
                  f"Repeated plane groups: {len(groups)}"]
        if "projected_positions_mm" not in row:
            lines.append("Plane grouping unavailable: see geometry/identity flags; zero listed groups is not a negative finding.")
        for group in groups:
            lines += [f"Plane {group['plane_group']}: position={group['representative_position_mm']:.9g} mm; "
                      f"frames={group['n_frames']}; excess={group['excess_frames']}",
                      f"  files: {group['file_names']}"]
            for field in ("SOPInstanceUID", "SeriesInstanceUID", "InstanceNumber", "EchoNumbers", "EchoTime",
                          "AcquisitionNumber", "TemporalPositionIdentifier", "TemporalPositionIndex", "EffectiveEchoTime"):
                if group[field] != ["unknown"]:
                    lines.append(f"  {field}: {group[field]}")
        # Count SOP identities by file rather than by frame: a multi-frame object shares one SOP UID.
        files = {record["file_name"]: record for record in records}
        uids = Counter(record["SOPInstanceUID"] for record in files.values() if record["SOPInstanceUID"] not in ("", "unknown"))
        lines.append(f"SOP UIDs reused by different files: {sum(count > 1 for count in uids.values())}")
    lines += ["", "Review frames.csv and repeated_plane_groups.csv for which files/frames share positions.",
              "CSV sets are in csv/ and csv_with_abnormal_study_ids/; only the latter adds a final abnormal_study_ids column.",
              "CSV file/UID indices replace source filenames/UIDs. This internal TXT retains source identifiers.",
              "Different echo/time identifiers may explain legitimate multi-dimensional data; same metadata does not prove duplicate pixels.",
              "Inspect raw/processed images separately before concluding whether this study is unusable."]
    return rows, frames, duplicates, "\n".join(lines) + "\n"


def export_reports(rows, frames, groups, output):
    study_id = rows[0]["study_id"]
    excluded = any(row.get("duplicate_plane_count", 0) > 0 or "repeated_slice_plane" in row["flags"] for row in rows)
    write_csv_pair(public_series(pd.DataFrame(rows), {study_id: 1}),
                   [[study_id] if excluded else [] for row in rows], output, "selected_series")
    # Preserve equality relationships between files/UIDs without exporting their original values.
    identities = {field: {} for field in ("file_name", "SOPInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID")}
    details = []
    allowed = ["modality", "frame_index_in_file", "order_index", "InstanceNumber", "SeriesNumber",
               "Rows", "Columns", "NumberOfFrames", "ImagePositionPatient", "ImageOrientationPatient",
               "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "EchoNumbers", "EchoTime",
               "EffectiveEchoTime", "AcquisitionNumber", "TemporalPositionIdentifier", "TemporalPositionIndex",
               "InStackPositionNumber", "DimensionIndexValues", "projected_position_mm", "plane_group",
               "plane_group_size", "repeated_plane"]
    index_names = ("file_index", "sop_index", "series_uid_index", "frame_of_reference_index")
    for frame in frames:
        record = dict(study_index=1)
        for field, name in zip(identities, index_names):
            value = frame.get(field, "unknown")
            mapping = identities[field]
            record[name] = mapping.setdefault(value, len(mapping) + 1) if value not in ("", "unknown") else "unknown"
        record.update({field: frame.get(field, "unknown") for field in allowed})
        details.append(record)
    write_csv_pair(pd.DataFrame(details, columns=["study_index", *index_names, *allowed]),
                   [[study_id] if frame["repeated_plane"] is True else [] for frame in frames], output, "frames")
    fields = ["modality", "plane_group", "representative_position_mm", "n_frames", "excess_frames",
              "order_indices", "InstanceNumber", "EchoNumbers", "EchoTime", "EffectiveEchoTime",
              "AcquisitionNumber", "TemporalPositionIdentifier", "TemporalPositionIndex",
              "InStackPositionNumber", "DimensionIndexValues"]
    details = [dict(study_index=1, **{field: group.get(field, "unknown") for field in fields},
                    sop_indices=[identities["SOPInstanceUID"].get(value, "unknown") for value in group["SOPInstanceUID"]],
                    series_uid_indices=[identities["SeriesInstanceUID"].get(value, "unknown") for value in group["SeriesInstanceUID"]])
               for group in groups]
    write_csv_pair(pd.DataFrame(details, columns=["study_index", *fields, "sop_indices", "series_uid_indices"]),
                   [[study_id] for group in groups], output, "repeated_plane_groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    directory = args.study_directory.resolve()
    output = args.output_dir.resolve()
    if not directory.is_dir():
        parser.error("Study directory does not exist")
    if output.exists() or output == directory or directory in output.parents:
        parser.error("Choose a new output directory outside the study directory")
    rows, frames, groups, report = build_report(directory, directory.name)
    output.mkdir(parents=True)
    export_reports(rows, frames, groups, output)
    (output / "repeated_slice_report.txt").write_text(report, encoding="utf-8")
    print(f"Finished: {directory.name}; repeated-plane groups={len(groups)}. {output}")


if __name__ == "__main__":
    main()
