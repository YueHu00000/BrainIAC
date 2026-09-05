"""Windows-safe bridge into the unchanged public BrainIAC preprocessing code."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _load_upstream(brainiac_root: Path):
    preprocessing_dir = brainiac_root / "src" / "preprocessing"
    script = preprocessing_dir / "mri_preprocess_3d_simple.py"
    sys.path.insert(0, str(preprocessing_dir))
    spec = importlib.util.spec_from_file_location("_brainiac_mri_preprocess_3d_simple", str(script))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load BrainIAC preprocessing from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_hd_bet_shape_fix() -> None:
    """Correct the public HD-BET array-vs-shape comparison at runtime.

    BrainIAC's bundled ``save_segmentation_nifti`` compares a 3-D array to a
    three-element size vector. The intended comparison is the array's shape;
    otherwise NumPy raises a broadcasting error for ordinary brain volumes.
    """
    from HD_BET import data_loading
    from HD_BET import run as hd_bet_run

    def save_segmentation_nifti(segmentation, dct, out_fname, order=1):
        old_size = dct.get("size_before_cropping")
        bbox = dct.get("brain_bbox")
        if bbox is not None:
            seg_old_size = np.zeros(old_size)
            for axis in range(3):
                bbox[axis][1] = np.min((bbox[axis][0] + segmentation.shape[axis], old_size[axis]))
            seg_old_size[
                bbox[0][0] : bbox[0][1],
                bbox[1][0] : bbox[1][1],
                bbox[2][0] : bbox[2][1],
            ] = segmentation
        else:
            seg_old_size = segmentation

        target_shape = np.asarray(dct["size"])[[2, 1, 0]]
        if np.any(np.asarray(seg_old_size.shape) != target_shape):
            seg_old_spacing = data_loading.resize_segmentation(seg_old_size, target_shape, order=order)
        else:
            seg_old_spacing = seg_old_size
        seg_resized_itk = sitk.GetImageFromArray(seg_old_spacing.astype(np.int32))
        seg_resized_itk.SetSpacing(np.asarray(dct["spacing"])[[0, 1, 2]])
        seg_resized_itk.SetOrigin(dct["origin"])
        seg_resized_itk.SetDirection(dct["direction"])
        sitk.WriteImage(seg_resized_itk, out_fname)

    hd_bet_run.save_segmentation_nifti = save_segmentation_nifti


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainiac-root", required=True)
    parser.add_argument("--temp_img", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    upstream = _load_upstream(Path(args.brainiac_root).resolve())
    _install_hd_bet_shape_fix()
    original_glob = upstream.glob.glob

    def forward_slash_glob(pattern):
        return [path.replace("\\", "/") for path in original_glob(pattern)]

    # Upstream extracts IDs with split('/'). Normalizing only the path spelling
    # preserves the discovered files and their order while making that parsing
    # work on Windows. The imaging operations remain in the upstream module.
    upstream.glob.glob = forward_slash_glob
    try:
        upstream.main(args.temp_img, args.input_dir, args.output_dir)
    finally:
        upstream.glob.glob = original_glob


if __name__ == "__main__":
    main()
