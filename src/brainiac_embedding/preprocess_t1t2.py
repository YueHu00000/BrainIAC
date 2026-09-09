"""Run the public BrainIAC preprocessing script for paired T1/T2 studies."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

MODALITIES = ("T1", "T2")


def default_brainiac_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_study_ids(root: str | Path) -> list[str]:
    """Discover the direct Study directories produced by conversion.

    The preprocessing scratch directory is reserved, never a Study.
    Validate both modalities before starting any expensive processing.
    """
    root = Path(root)
    studies = sorted(path.name for path in root.iterdir()
                     if path.is_dir() and path.name != ".preprocess_tmp")
    if not studies:
        raise ValueError(f"no Study directories found in {root}")
    for study_id in studies:
        for modality in MODALITIES:
            path = root / study_id / f"{modality}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(f"paired NIfTI does not exist: {path}")
    return studies


def _expected_raw_paths(raw_root: Path, study_id: str) -> dict[str, Path]:
    paths = {modality: raw_root / study_id / f"{modality}.nii.gz" for modality in MODALITIES}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"raw NIfTI does not exist: {path}")
    return paths


def preprocess_study(
    study_id: str,
    raw_root: str | Path,
    processed_root: str | Path,
    *,
    brainiac_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Run the original BrainIAC CLI and publish paired nested outputs.

    A small bridge normalizes only the spelling of paths returned by ``glob``
    so the upstream ``split('/')`` filename parsing works on Windows. The
    bridge then calls the unchanged upstream ``main`` imaging implementation.
    """
    raw_root = Path(raw_root).resolve()
    processed_root = Path(processed_root).resolve()
    _expected_raw_paths(raw_root, study_id)

    destination_dir = processed_root / study_id
    destinations = {modality: destination_dir / f"{modality}.nii.gz" for modality in MODALITIES}
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"processed output already exists: {existing[0]}")

    repo_root = Path(brainiac_root).resolve() if brainiac_root is not None else default_brainiac_root()
    preprocessing_dir = repo_root / "src" / "preprocessing"
    script = preprocessing_dir / "mri_preprocess_3d_simple.py"
    bridge = Path(__file__).resolve().with_name("_preprocess_bridge.py")
    template = preprocessing_dir / "atlases" / "temp_head.nii.gz"
    if not script.is_file():
        raise FileNotFoundError(f"BrainIAC preprocessing script does not exist: {script}")
    if not template.is_file():
        raise FileNotFoundError(f"BrainIAC template does not exist: {template}")
    if not bridge.is_file():
        raise FileNotFoundError(f"BrainIAC preprocessing bridge does not exist: {bridge}")

    processed_root.mkdir(parents=True, exist_ok=True)
    temporary_parent = processed_root / ".preprocess_tmp"
    temporary_parent.mkdir(exist_ok=True)
    executable = str(python_executable or sys.executable)

    try:
        with tempfile.TemporaryDirectory(prefix=f"{study_id}_", dir=str(temporary_parent)) as temporary:
            temporary_output = Path(temporary)
            command = [
                executable,
                str(bridge),
                "--brainiac-root",
                repo_root.as_posix(),
                "--temp_img",
                template.as_posix(),
                "--input_dir",
                (raw_root / study_id).as_posix(),
                "--output_dir",
                temporary_output.as_posix(),
            ]
            subprocess.run(command, cwd=str(preprocessing_dir), check=True)

            generated = {
                modality: temporary_output / f"{modality}_0000.nii.gz" for modality in MODALITIES
            }
            missing = [path for path in generated.values() if not path.is_file()]
            if missing:
                raise RuntimeError(f"BrainIAC preprocessing did not produce: {missing[0]}")

            destination_dir.mkdir(parents=True, exist_ok=True)
            for modality in MODALITIES:
                os.replace(str(generated[modality]), str(destinations[modality]))
    finally:
        if temporary_parent.is_dir() and not any(temporary_parent.iterdir()):
            temporary_parent.rmdir()

    return destinations["T1"], destinations["T2"]


def preprocess_whitelist(
    study_ids: Sequence[str],
    raw_root: str | Path,
    processed_root: str | Path,
    *,
    brainiac_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    for study_id in study_ids:
        preprocess_study(
            study_id,
            raw_root,
            processed_root,
            brainiac_root=brainiac_root,
            python_executable=python_executable,
            overwrite=overwrite,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run original BrainIAC preprocessing for nested T1/T2 studies")
    parser.add_argument("--study-id", action="append", help="Optional subset; default: all Study directories in the input root")
    parser.add_argument("--raw-root", required=True, help="Root containing <Study_ID>/T1.nii.gz and T2.nii.gz")
    parser.add_argument("--processed-root", required=True, help="Destination root for processed nested NIfTI")
    parser.add_argument("--brainiac-root", default=str(default_brainiac_root()), help="BrainIAC repository root")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the original preprocessing CLI")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing processed T1/T2 files")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    study_ids = args.study_id if args.study_id is not None else discover_study_ids(args.raw_root)
    preprocess_whitelist(
        study_ids,
        args.raw_root,
        args.processed_root,
        brainiac_root=args.brainiac_root,
        python_executable=args.python,
        overwrite=args.overwrite,
    )
    print(f"Preprocessed {len(study_ids)} studies to {args.processed_root}")


if __name__ == "__main__":
    main()
