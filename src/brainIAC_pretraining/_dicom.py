"""DICOM selection and the count/coverage measurements needed by pretraining."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pydicom

from brainIAC_pretraining._common import MultiFrameDicomError


POSITION_TOL_MM = 0.01
ORIENTATION_TOL = 1e-4


def integer(ds, key):
    value = ds.get(key)
    return int(value) if value is not None and str(value).strip() else None


def reject_multiframe(ds, unique_id, path):
    if int(ds.get("NumberOfFrames", 1)) > 1:
        raise MultiFrameDicomError(f"{unique_id}: unsupported multi-frame DICOM: {path}")


def is_projection(ds):
    image_type = ds.get("ImageType", [])
    if isinstance(image_type, str):
        image_type = image_type.split("\\")
    if any(str(value).strip().upper() in ("PJN", "PROJECTION IMAGE") for value in image_type):
        return True
    description = str(ds.get("SeriesDescription", "")).strip().upper()
    return description.rsplit("/", 1)[-1].strip() == "PJN"


def inspect_study(study_id, roots):
    matches = [root / study_id for root in roots if (root / study_id).is_dir()]
    study = dict(status="complete")
    if len(matches) != 1:
        study["status"] = "study_not_found" if not matches else "study_ambiguous"
        return [], study, [dict(study_id=study_id, file="", reason=study["status"])]
    directory = matches[0]
    issues, groups = [], defaultdict(list)
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as error:
            issues.append(dict(study_id=study_id, file=str(path), reason=f"{type(error).__name__}: {error}"))
            continue
        if str(ds.get("Modality", "")).upper() != "MR":
            continue
        uid = str(ds.get("SeriesInstanceUID") or "").strip()
        # Also reject files that would later be excluded or have invalid metadata.
        reject_multiframe(ds, f"{study_id}_{uid or 'unknown'}_{ds.get('SeriesNumber', 'unknown')}", path)
        try:
            if not uid:
                raise ValueError("Missing SeriesInstanceUID")
            number = integer(ds, "SeriesNumber")
            if number is None:
                raise ValueError("Missing SeriesNumber")
            acquisition = integer(ds, "AcquisitionNumber")
        except Exception as error:
            issues.append(dict(study_id=study_id, file=str(path), reason=f"{type(error).__name__}: {error}"))
            continue
        groups[(uid, number)].append((path, ds, acquisition))
    rows = []
    for (uid, number), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        if any(is_projection(ds) for _, ds, _ in items):
            continue
        acquisitions = [acquisition for _, _, acquisition in items if acquisition is not None]
        chosen = max(acquisitions) if acquisitions else None
        selected = [(path, ds) for path, ds, acquisition in items if acquisition == chosen]
        selected.sort(key=lambda item: (integer(item[1], "InstanceNumber") or 0, str(item[0])))
        rows.append(dict(unique_id=f"{study_id}_{uid}_{number}", study_id=study_id,
                         series_instance_uid=uid, series_number=number,
                         study_directory=str(directory),
                         acquisition_number=chosen if chosen is not None else "unknown",
                         file_names=[str(path.relative_to(directory)) for path, _ in selected]))
    study["status"] = "incomplete" if issues else "complete" if rows else "no_mr_series"
    return rows, study, issues


def numbers(value, size):
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size == size and np.isfinite(array).all() else None


def geometry(ds):
    position = getattr(ds, "ImagePositionPatient", None)
    orientation = getattr(ds, "ImageOrientationPatient", None)
    shared = list(getattr(ds, "SharedFunctionalGroupsSequence", [])[:1])
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
    for group in shared + list(per_frame[:1]):
        planes = getattr(group, "PlanePositionSequence", [])
        axes = getattr(group, "PlaneOrientationSequence", [])
        if planes:
            position = getattr(planes[0], "ImagePositionPatient", position)
        if axes:
            orientation = getattr(axes[0], "ImageOrientationPatient", orientation)
    if per_frame and len(per_frame) != 1:
        position = None
    return numbers(position, 3), numbers(orientation, 6)


def measure_series(row):
    """Read only manifest-listed headers; selection is never rerun here."""
    files = [Path(row["study_directory"]) / name for name in json.loads(row["file_names"])]
    counts, positions, orientations = [], [], []
    coordinate_ids = {key: set() for key in ("StudyInstanceUID", "FrameOfReferenceUID")}
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        reject_multiframe(ds, row["unique_id"], path)
        counts.append(numbers(ds.get("NumberOfFrames", 1), 1))
        position, orientation = geometry(ds)
        positions.append(position)
        orientations.append(orientation)
        for key, values in coordinate_ids.items():
            if ds.get(key):
                values.add(str(ds.get(key)))
    result = dict(row, file_count=len(files), frame_count="", unique_slice_count="", coverage_mm="")
    if not counts or any(count is None or count[0] != 1 for count in counts):
        return result
    result["frame_count"] = len(counts)
    if (any(value is None for value in positions + orientations)
            or any(len(values) > 1 for values in coordinate_ids.values())
            or not np.allclose(orientations, orientations[0], atol=ORIENTATION_TOL, rtol=0)):
        return result
    row_axis, column_axis = orientations[0][:3], orientations[0][3:]
    if (abs(np.linalg.norm(row_axis) - 1) > ORIENTATION_TOL
            or abs(np.linalg.norm(column_axis) - 1) > ORIENTATION_TOL
            or abs(np.dot(row_axis, column_axis)) > ORIENTATION_TOL):
        return result
    normal = np.cross(row_axis, column_axis)
    normal /= np.linalg.norm(normal)
    projected = (np.asarray(positions) - positions[0]) @ normal
    unique = []
    for position in np.sort(projected):
        if not unique or position - unique[-1] > POSITION_TOL_MM:
            unique.append(float(position))
    result.update(unique_slice_count=len(unique), coverage_mm=unique[-1] - unique[0])
    return result
