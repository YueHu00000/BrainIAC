"""Run DICOM conversion, BrainIAC preprocessing, and three embedding passes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import torch

from .convert_dicom_t1t2 import convert_whitelist
from .extract_embeddings import extract_whitelist_for_checkpoint
from .preprocess_t1t2 import default_brainiac_root, preprocess_whitelist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KPSC-style T1/T2 DICOM to BrainIAC multilayer embeddings")
    parser.add_argument("--folder-list", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--study-id-column", default="Study_ID")
    parser.add_argument("--raw-nifti-root", required=True)
    parser.add_argument("--processed-nifti-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--brainiac-checkpoint", required=True)
    parser.add_argument("--brainage-checkpoint", required=True)
    parser.add_argument("--mci-checkpoint", required=True)
    parser.add_argument("--brainiac-root", default=str(default_brainiac_root()))
    parser.add_argument("--python", default=sys.executable, help="Python executable for BrainIAC preprocessing")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    study_ids = convert_whitelist(
        args.folder_list,
        args.csv,
        args.raw_nifti_root,
        study_id_column=args.study_id_column,
        overwrite=args.overwrite,
    )
    preprocess_whitelist(
        study_ids,
        args.raw_nifti_root,
        args.processed_nifti_root,
        brainiac_root=args.brainiac_root,
        python_executable=args.python,
        overwrite=args.overwrite,
    )

    embedding_root = Path(args.embedding_root)
    model_runs = (
        ("backbone", args.brainiac_checkpoint, "brainiac_backbone"),
        ("brainage", args.brainage_checkpoint, "brainage_finetuned"),
        ("mci", args.mci_checkpoint, "mci_finetuned"),
    )
    for model_kind, checkpoint, output_name in model_runs:
        extract_whitelist_for_checkpoint(
            study_ids,
            args.processed_nifti_root,
            embedding_root / output_name,
            model_kind=model_kind,
            checkpoint_path=checkpoint,
            brainiac_checkpoint=args.brainiac_checkpoint if model_kind != "backbone" else None,
            device=args.device,
            brainiac_root=args.brainiac_root,
            overwrite=args.overwrite,
        )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)
    print(f"Completed BrainIAC embedding workflow; outputs: {args.embedding_root}")


if __name__ == "__main__":
    main()

