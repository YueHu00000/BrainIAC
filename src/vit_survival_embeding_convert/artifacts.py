"""Read legacy NORM-A slice CLS artifacts without importing the old pipeline."""

import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


NORM_A_ID = "tufts_train_global_v1"
BLOCK_NUMBERS = (3, 6, 9, 12)
SCHEMA_LAYERS = {
    "vit_multilayer_embedding_v1": list(BLOCK_NUMBERS),
    "vit_multilayer_embedding_v2": list(range(1, 13)),
}
REQUIRED_KEYS = {"block_cls", "final_cls", "metadata_json"}


def _read_metadata(archive, path):
    if not REQUIRED_KEYS.issubset(archive.files):
        raise ValueError("not_raw_embedding: expected block_cls, final_cls, metadata_json")
    value = archive["metadata_json"]
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"{path}: metadata_json must be a text scalar")
    metadata = json.loads(value.item())
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must describe an object")
    return metadata


def read_metadata(path):
    """Read the raw artifact metadata without loading either tensor array."""
    path = Path(path)
    with path.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
        return _read_metadata(archive, path)


def _validate_metadata(metadata, path, study_id):
    if not study_id or metadata.get("study_id") != study_id or path.stem != study_id:
        raise ValueError(f"{path}: filename, requested Study_ID and metadata study_id differ")
    if metadata.get("normalization_id") != NORM_A_ID:
        raise ValueError(f"{path}: expected normalization_id={NORM_A_ID}")
    layers = SCHEMA_LAYERS.get(metadata.get("schema_version"))
    if layers is None or metadata.get("layers") != layers:
        raise ValueError(f"{path}: incompatible schema_version/layers")
    for name in ("weights_id", "transform_id"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"{path}: missing {name}")
    if not isinstance(metadata.get("source_provenance"), dict):
        raise ValueError(f"{path}: missing source_provenance")
    provenance = metadata.get("normalization_provenance")
    parameters = provenance.get("parameters") if isinstance(provenance, dict) else None
    if (not isinstance(parameters, dict) or parameters.get("normalization_id") != NORM_A_ID
            or parameters.get("fit_split") != "train"):
        raise ValueError(f"{path}: missing train-fitted NORM-A parameters")
    study_ids = parameters.get("fit_study_ids")
    if (not isinstance(study_ids, list) or not study_ids
            or any(not isinstance(value, str) or not value.strip() for value in study_ids)
            or len(study_ids) != len(set(study_ids))):
        raise ValueError(f"{path}: NORM-A fit_study_ids must identify the fitting cohort")
    mean = np.asarray(parameters.get("mean"), dtype=np.float64)
    std = np.asarray(parameters.get("std"), dtype=np.float64)
    if (mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError(f"{path}: invalid NORM-A mean/std")
    return layers


def index_embeddings(results_dir, embedding_dir=None):
    """Return Study_ID -> paths and issues; retain duplicate paths for caller audit.

    Only inspect NPZ keys and metadata here. Full tensors are loaded later for
    selected studies. Both normalization branches may coexist in the tree;
    non-NORM-A files are reported and omitted, never chosen by directory order.
    """
    root = Path(embedding_dir) if embedding_dir is not None else Path(results_dir) / "embeddings"
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Legacy embedding directory does not exist: {root}")
    index, issues = {}, []
    for path in sorted(root.rglob("*.npz")):
        metadata = {}
        try:
            metadata = read_metadata(path)
            study_id = metadata.get("study_id")
            if metadata.get("normalization_id") != NORM_A_ID:
                issues.append(dict(path=str(path), study_id=study_id or "",
                                   reason="normalization_not_norm_a"))
                continue
            _validate_metadata(metadata, path, study_id)
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
            issues.append(dict(path=str(path), study_id=metadata.get("study_id", ""),
                               reason=str(error)))
            continue
        index.setdefault(study_id, []).append(path)
    for study_id, paths in index.items():
        if len(paths) > 1:
            for path in paths:
                issues.append(dict(path=str(path), study_id=study_id,
                                   reason="duplicate_study_embedding"))
    return index, issues


def load_embedding(path, study_id):
    """Return [2,4,768] block means, [2,768] final means, and original metadata.

    Axis zero is T1 then T2. Blocks are before the encoder final LayerNorm;
    final_cls is after it. Only the slice axis is averaged.
    """
    path = Path(path)
    with path.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
        metadata = _read_metadata(archive, path)
        layers = _validate_metadata(metadata, path, study_id)
        blocks = archive["block_cls"]
        final = archive["final_cls"]
    if (blocks.dtype != np.float32 or final.dtype != np.float32
            or blocks.ndim != 4 or blocks.shape[0] != 2
            or blocks.shape[2:] != (len(layers), 768)):
        raise ValueError(f"{path}: expected float32 block_cls [2,S,{len(layers)},768]")
    slices = blocks.shape[1]
    if slices < 1 or final.shape != (2, slices, 768):
        raise ValueError(f"{path}: expected final_cls [2,S,768] with matching positive S")
    shape = metadata.get("shape")
    if (metadata.get("dtype") != "float32" or metadata.get("slice_count") != slices
            or not isinstance(shape, dict) or shape.get("block_cls") != list(blocks.shape)
            or shape.get("final_cls") != list(final.shape)):
        raise ValueError(f"{path}: tensor shape/dtype differs from metadata")
    if not np.isfinite(blocks).all() or not np.isfinite(final).all():
        raise ValueError(f"{path}: embedding contains nonfinite values")
    if len(layers) == 12:
        blocks = np.take(blocks, [number - 1 for number in BLOCK_NUMBERS], axis=2)
    block_means = np.ascontiguousarray(blocks.mean(axis=1), dtype=np.float32)
    final_means = np.ascontiguousarray(final.mean(axis=1), dtype=np.float32)
    return block_means, final_means, metadata
