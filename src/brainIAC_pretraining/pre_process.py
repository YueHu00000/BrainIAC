"""Run original BrainIAC preprocessing independently for CSV-listed series."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainIAC_pretraining._common import read_ids, scan_outputs, validate_id, write_rows


def preprocess_series(unique_id: str, raw_dir: Path, output_dir: Path,
                      python_executable: str | None = None) -> Path:
    validate_id(unique_id)
    destination = output_dir / f"{unique_id}.nii.gz"
    if destination.is_file():
        print(f"[skip] preprocess {unique_id}", flush=True)
        return destination
    repo_root = Path(__file__).resolve().parents[2]
    preprocessing_dir = repo_root / "src" / "preprocessing"
    bridge = repo_root / "src" / "brainiac_embedding" / "_preprocess_bridge.py"
    template = preprocessing_dir / "atlases" / "temp_head.nii.gz"
    scratch = output_dir / ".preprocess_tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=f"{unique_id}_", dir=scratch) as temporary:
        workspace = Path(temporary)
        inputs, outputs = workspace / "input", workspace / "output"
        inputs.mkdir()
        outputs.mkdir()
        source = raw_dir / f"{unique_id}.nii.gz"
        # Upstream truncates IDs at dots and skips names containing "_mask".
        staged = inputs / "volume.nii.gz"
        try:
            os.link(source, staged)
        except OSError:
            shutil.copyfile(source, staged)
        command = [
            str(python_executable or sys.executable), str(bridge),
            "--brainiac-root", repo_root.as_posix(), "--temp_img", template.as_posix(),
            "--input_dir", inputs.as_posix(), "--output_dir", outputs.as_posix(),
        ]
        subprocess.run(command, cwd=str(preprocessing_dir), check=True)
        result = outputs / "volume_0000.nii.gz"
        if not result.is_file():
            raise RuntimeError("preprocessing produced no final brain image; see imaging output above")
        os.replace(result, destination)
    print(f"[done] preprocess {unique_id}", flush=True)
    return destination


def preprocess_csv(converted_csv: str | Path, raw_dir: str | Path, output_dir: str | Path,
                   pre_process_csv: str | Path | None = None,
                   python_executable: str | None = None) -> list[str]:
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    if raw_dir == output_dir or raw_dir in output_dir.parents or output_dir in raw_dir.parents:
        raise ValueError("raw-dir and output-dir must not overlap")
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_process_csv = Path(pre_process_csv) if pre_process_csv else output_dir / "pre_process.csv"
    failures = []
    try:
        for unique_id in read_ids(converted_csv):
            try:
                preprocess_series(unique_id, raw_dir, output_dir, python_executable)
            except Exception as error:
                failures.append({"unique_id": unique_id, "reason": str(error)})
                print(f"[failed] preprocess {unique_id}: {error}", flush=True)
    finally:
        completed = scan_outputs(output_dir, pre_process_csv)
        write_rows(output_dir / "pre_process_errors.csv", failures, ["unique_id", "reason"])
    return completed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-csv", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pre-process-csv")
    parser.add_argument("--python-executable")
    args = parser.parse_args(argv)
    preprocess_csv(args.converted_csv, args.raw_dir, args.output_dir,
                   args.pre_process_csv, args.python_executable)


if __name__ == "__main__":
    main()
