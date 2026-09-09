"""Extract block 3/6/9/12 and final token-0 and mean-pooled BrainIAC embeddings."""

from __future__ import annotations

import argparse
import importlib.util
import os
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from ._resume import check_separate_roots, remove_output, validate_study_id
from .preprocess_t1t2 import MODALITIES, default_brainiac_root, discover_study_ids

BLOCK_NUMBERS = (3, 6, 9, 12)
BLOCK_INDICES = tuple(number - 1 for number in BLOCK_NUMBERS)
EXPECTED_INPUT_SHAPE = (2, 1, 96, 96, 96)
EXPECTED_TOKEN_COUNT = 216
EXPECTED_HIDDEN_SIZE = 768
POOLING_NAMES = ("token0", "mean_pooling")
MODEL_KINDS = ("backbone", "brainage", "mci")


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_public_model_module(brainiac_root: str | Path | None = None) -> ModuleType:
    root = Path(brainiac_root).resolve() if brainiac_root is not None else default_brainiac_root()
    path = root / "src" / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"BrainIAC model.py does not exist: {path}")
    return _load_source_module("_brainiac_public_model", path)


def load_public_dataset_module(brainiac_root: str | Path | None = None) -> ModuleType:
    root = Path(brainiac_root).resolve() if brainiac_root is not None else default_brainiac_root()
    path = root / "src" / "dataset.py"
    if not path.is_file():
        raise FileNotFoundError(f"BrainIAC dataset.py does not exist: {path}")
    return _load_source_module("_brainiac_public_dataset", path)


def load_encoder(
    model_kind: str,
    checkpoint_path: str | Path,
    *,
    brainiac_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    brainiac_root: str | Path | None = None,
    model_module: ModuleType | None = None,
) -> torch.nn.Module:
    """Load the raw MONAI ViT using the public BrainIAC checkpoint semantics."""
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}, found {model_kind!r}")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    public_model = model_module or load_public_model_module(brainiac_root)
    target_device = torch.device(device)

    if model_kind == "backbone":
        wrapper = public_model.ViTBackboneNet(str(checkpoint_path))
        encoder = wrapper.backbone
    else:
        if brainiac_checkpoint is None:
            raise ValueError("brainiac_checkpoint is required for a fine-tuned model")
        brainiac_checkpoint = Path(brainiac_checkpoint)
        if not brainiac_checkpoint.is_file():
            raise FileNotFoundError(f"BrainIAC checkpoint does not exist: {brainiac_checkpoint}")

        backbone = public_model.ViTBackboneNet(str(brainiac_checkpoint))
        classifier = public_model.Classifier(d_model=EXPECTED_HIDDEN_SIZE, num_classes=1)
        full_model = public_model.SingleScanModel(backbone, classifier)

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[6:]
            remapped_state_dict[key] = value
        full_model.load_state_dict(remapped_state_dict, strict=True)
        encoder = full_model.backbone.backbone

    encoder.to(target_device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def extract_selected_embeddings(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return token0 and mean over spatial tokens 1: for blocks and final LN."""
    if tuple(images.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"images must have shape {EXPECTED_INPUT_SHAPE}, found {tuple(images.shape)}")
    if not torch.isfinite(images).all():
        raise ValueError("images contain NaN or Inf")

    if device is None:
        try:
            target_device = next(encoder.parameters()).device
        except StopIteration:
            target_device = images.device
    else:
        target_device = torch.device(device)
        encoder.to(target_device)

    encoder.eval()
    with torch.inference_mode():
        final_tokens, hidden_states = encoder(images.to(device=target_device, dtype=torch.float32))

    if len(hidden_states) != 12:
        raise ValueError(f"BrainIAC ViT must return 12 hidden states, found {len(hidden_states)}")
    expected_token_shape = (2, EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN_SIZE)
    if tuple(final_tokens.shape) != expected_token_shape:
        raise ValueError(f"final token shape must be {expected_token_shape}, found {tuple(final_tokens.shape)}")
    for index, state in enumerate(hidden_states):
        if tuple(state.shape) != expected_token_shape:
            raise ValueError(
                f"hidden state {index + 1} must have shape {expected_token_shape}, found {tuple(state.shape)}"
            )

    block_embed = {}
    final_embed = {}
    for pooling in POOLING_NAMES:
        if pooling == "token0":
            block_tensor = torch.stack([hidden_states[i][:, 0, :] for i in BLOCK_INDICES], dim=1)
            final_tensor = final_tokens[:, 0, :]
        else:
            block_tensor = torch.stack([hidden_states[i][:, 1:, :].mean(dim=1)
                                        for i in BLOCK_INDICES], dim=1)
            final_tensor = final_tokens[:, 1:, :].mean(dim=1)
        block_embed[pooling] = block_tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        final_embed[pooling] = final_tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(block_embed[pooling]).all() or not np.isfinite(final_embed[pooling]).all():
            raise ValueError("encoder produced NaN or Inf embeddings")
    return block_embed, final_embed


def load_study_images(
    study_id: str,
    processed_root: str | Path,
    *,
    brainiac_root: str | Path | None = None,
    transform=None,
) -> torch.Tensor:
    """Load T1 then T2 using the public BrainIAC validation transform."""
    if transform is None:
        dataset_module = load_public_dataset_module(brainiac_root)
        transform = dataset_module.get_validation_transform(image_size=(96, 96, 96))

    modality_tensors = []
    for modality in MODALITIES:
        path = Path(processed_root) / study_id / f"{modality}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"processed NIfTI does not exist: {path}")
        transformed = transform({"image": str(path)})
        tensor = torch.as_tensor(transformed["image"], dtype=torch.float32)
        if tuple(tensor.shape) != (1, 96, 96, 96):
            raise ValueError(f"{modality} transform returned shape {tuple(tensor.shape)}")
        modality_tensors.append(tensor)
    images = torch.stack(modality_tensors, dim=0)
    if not torch.isfinite(images).all():
        raise ValueError(f"transformed images contain NaN or Inf for {study_id}")
    return images


def embedding_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"block_embed", "final_embed"}:
                return False
            for key, shape in (("block_embed", (2, 4, 768)), ("final_embed", (2, 768))):
                array = data[key]
                if array.shape != shape or array.dtype.names != POOLING_NAMES:
                    return False
                for pooling in POOLING_NAMES:
                    if array[pooling].dtype != np.float32 or not np.isfinite(array[pooling]).all():
                        return False
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        return False
    return True


def save_embedding_npz(
    output_path: str | Path,
    block_embed: dict[str, np.ndarray],
    final_embed: dict[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> Path:
    """Save two structured arrays with float32 pooling fields; no pickle or metadata."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"embedding output already exists: {output_path}")
    arrays = {}
    for name, embeddings, shape in (
        ("block_embed", block_embed, (2, 4, EXPECTED_HIDDEN_SIZE)),
        ("final_embed", final_embed, (2, EXPECTED_HIDDEN_SIZE)),
    ):
        if set(embeddings) != set(POOLING_NAMES):
            raise ValueError(f"{name} must contain exactly {POOLING_NAMES}")
        packed = np.empty(shape, dtype=[(pooling, np.float32) for pooling in POOLING_NAMES])
        for pooling in POOLING_NAMES:
            value = np.asarray(embeddings[pooling], dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{name}.{pooling} must have shape {shape}, found {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name}.{pooling} contains NaN or Inf")
            packed[pooling] = value
        arrays[name] = packed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    remove_output(temporary, output_path.parent)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output_path)
    finally:
        remove_output(temporary, output_path.parent)
    return output_path


def extract_whitelist_for_checkpoint(
    study_ids: Sequence[str],
    processed_root: str | Path,
    output_root: str | Path,
    *,
    model_kind: str,
    checkpoint_path: str | Path,
    brainiac_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    brainiac_root: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    processed_root, output_root = Path(processed_root).resolve(), Path(output_root).resolve()
    check_separate_roots(processed_root, output_root)
    encoder = None
    for study_id in study_ids:
        validate_study_id(study_id)
        output_path = output_root / f"{study_id}.npz"
        if not overwrite and embedding_complete(output_path):
            print(f"[skip] {model_kind} embedding {study_id}")
            continue
        remove_output(output_path.with_name(output_path.name + ".tmp"), output_root)
        remove_output(output_path, output_root)
        print(f"[compute] {model_kind} embedding {study_id}")
        if encoder is None:
            encoder = load_encoder(
                model_kind, checkpoint_path, brainiac_checkpoint=brainiac_checkpoint,
                device=device, brainiac_root=brainiac_root,
            )
        images = load_study_images(study_id, processed_root, brainiac_root=brainiac_root)
        block_embed, final_embed = extract_selected_embeddings(encoder, images, device=device)
        save_embedding_npz(output_path, block_embed, final_embed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract BrainIAC block 3/6/9/12 and final token-0 and mean-pooled embeddings")
    parser.add_argument("--study-id", action="append", help="Optional subset; default: all Study directories in the input root")
    parser.add_argument("--processed-root", required=True, help="Root containing nested processed T1/T2 NIfTI")
    parser.add_argument("--output-root", required=True, help="Destination for one <Study_ID>.npz per Study")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--brainiac-checkpoint", help="Required for brainage and mci fine-tuned checkpoints")
    parser.add_argument("--brainiac-root", default=str(default_brainiac_root()))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Force recomputation, including completed NPZ files")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    study_ids = args.study_id if args.study_id is not None else discover_study_ids(args.processed_root)
    extract_whitelist_for_checkpoint(
        study_ids,
        args.processed_root,
        args.output_root,
        model_kind=args.model_kind,
        checkpoint_path=args.checkpoint,
        brainiac_checkpoint=args.brainiac_checkpoint,
        device=args.device,
        brainiac_root=args.brainiac_root,
        overwrite=args.overwrite,
    )
    print(f"Finished {len(study_ids)} {args.model_kind} embedding files (computed or skipped) to {args.output_root}")


if __name__ == "__main__":
    main()

