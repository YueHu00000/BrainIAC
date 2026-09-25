"""Report raw NIfTI geometry, metadata and voxel statistics for DICOM comparison."""

import argparse
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np


def build_report(source):
    source = Path(source).resolve()
    image = nib.load(source)
    header = image.header
    data = image.get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    lines = [
        "RAW NIFTI REPORT", f"Input: {source}", f"Image class: {type(image).__name__}",
        f"Shape (voxel axes, not necessarily anatomical axes): {image.shape}",
        f"Stored dtype: {image.get_data_dtype()}", f"Zooms: {header.get_zooms()}",
        f"Spatial/time units: {header.get_xyzt_units()}",
        f"Axis directions: {nib.aff2axcodes(image.affine)}",
        f"Reader scaling slope/intercept: {image.dataobj.slope}, {image.dataobj.inter}",
        "Affine below maps voxel indices to NIfTI RAS+ coordinates (use header spatial units).",
        "DICOM uses LPS: negate physical X and Y to compare with NIfTI RAS+.",
        "Selected affine:", np.array2string(image.affine, precision=9),
    ]
    for label in ("qform", "sform"):
        matrix, code = getattr(image, f"get_{label}")(coded=True)
        lines += [f"{label} code: {int(code)}", str(matrix)]
    if len(image.shape) >= 3:
        corners = np.asarray(list(product(*[(0, n - 1) for n in image.shape[:3]])))
        world = nib.affines.apply_affine(image.affine, corners)
        lines += [f"Voxel-centre bounds RAS: min={world.min(axis=0).tolist()}, max={world.max(axis=0).tolist()}",
                  f"First voxel centre RAS: {image.affine[:3, 3].tolist()}",
                  f"Third voxel-axis length: {image.shape[2]} (not automatically the number of acquired DICOM slices)",
                  f"Third voxel-axis step vector RAS: {image.affine[:3, 2].tolist()}"]
    lines += [f"Finite voxels: {finite.size} / {data.size}",
              f"NaN voxels: {int(np.isnan(data).sum())}", f"Inf voxels: {int(np.isinf(data).sum())}",
              f"Nonzero finite voxels: {int(np.count_nonzero(finite))}"]
    if finite.size:
        lines.append(f"Scaled voxel statistics (float32): min={finite.min()}, max={finite.max()}, mean={finite.mean(dtype=np.float64)}, std={finite.std(dtype=np.float64)}")
    lines += ["", "=== COMPLETE HEADER ===", str(header),
              "", f"=== EXTENSIONS ({len(header.extensions)}) ==="]
    for index, extension in enumerate(header.extensions, 1):
        content = extension.content
        lines.append(f"Extension {index}: code={extension.get_code()}, payload bytes={len(content)}")
        try:
            lines.append(content.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append("Binary extension; no text decoded. Inspect with a format-specific reader if needed.")
    lines += ["", "=== INTERPRETATION ===",
              "Standard NIfTI headers have no dedicated SeriesInstanceUID/SeriesNumber fields.",
              "Check descrip, aux_file and extensions for converter-specific provenance; it may be absent.",
              "T1 filename alone does not validate sequence type or identify a source series.",
              "Matching shape/spacing alone is not proof of a matching DICOM series.",
              "Use raw pre-registration images. Template-resampled dimensions cannot recover original slice count.",
              "This report does not compare DICOM pixel arrays or establish training-set inclusion."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw T1.nii.gz or .nii path")
    parser.add_argument("--output", type=Path, help="New TXT path; default: <filename>_info.txt in current directory")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    name = args.input.name.removesuffix(".gz").removesuffix(".nii")
    output = args.output or Path(f"{name}_info.txt")
    if output.exists():
        parser.error(f"Output already exists; choose another --output: {output}")
    report = build_report(args.input)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(f"Report saved: {output.resolve()}")


if __name__ == "__main__":
    main()
