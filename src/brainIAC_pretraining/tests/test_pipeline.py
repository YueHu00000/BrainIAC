"""Synthetic files through inventory, conversion, preprocessing, and exclusion."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from pydicom.uid import generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining import convert, manifest, pre_process, quality
from brainIAC_pretraining._common import read_ids, read_rows, write_rows
from brainIAC_pretraining.exclude import exclude_files
from test_processing import write_dicom


class PipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_uses_same_acquisition_and_final_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dicom_root = root / "dicom"
            for study, slices in (("S_1", 3), ("S_2", 2), ("S_3", 3)):
                study_dir = dicom_root / study
                study_dir.mkdir(parents=True)
                uid = generate_uid()
                for acquisition in (1, 2):
                    for index in range(slices):
                        write_dicom(study_dir / f"a{acquisition}_{index}.dcm", uid,
                                    acquisition, index, acquisition * 100 + index)
            labels, folders = root / "labels.csv", root / "folders.txt"
            write_rows(labels, ({"Study_ID": study} for study in ("S_1", "S_2", "S_3")), ["Study_ID"])
            folders.write_text("dicom\n", encoding="utf-8")
            scripts = Path(manifest.__file__).parent
            inventory = root / "inventory"
            subprocess.run([sys.executable, str(scripts / "manifest.py"),
                            "--csv", str(labels), "--folder-list", str(folders),
                            "--output-dir", str(inventory)], check=True, capture_output=True, text=True)
            manifest_csv = inventory / "manifest.csv"
            raw, processed = root / "raw", root / "processed"
            self.assertEqual(convert.convert_manifest(manifest_csv, raw),
                             ["S_1_10", "S_2_10", "S_3_10"])
            image = sitk.ReadImage(str(raw / "S_1_10.nii.gz"))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image)[:, 0, 0], [200, 201, 202])
            accepted, rejected = quality.build_quality(read_rows(manifest_csv))
            self.assertFalse(rejected)
            quality_csv = root / "image_number_coverage.csv"
            write_rows(quality_csv, accepted, quality.FIELDS)

            def imaging(command, **kwargs):
                inputs = Path(command[command.index("--input_dir") + 1])
                outputs = Path(command[command.index("--output_dir") + 1])
                source = next(inputs.glob("*.nii.gz"))
                if inputs.parent.name.startswith("S_3_10_"):
                    raise subprocess.CalledProcessError(1, command)
                sitk.WriteImage(sitk.ReadImage(str(source)),
                                str(outputs / "volume_0000.nii.gz"))

            with patch.object(pre_process.subprocess, "run", side_effect=imaging):
                self.assertEqual(pre_process.preprocess_csv(raw / "converted.csv", raw, processed),
                                 ["S_1_10", "S_2_10"])
            exclude_files(quality_csv, processed / "pre_process.csv", processed, root / "excluded",
                          frame_count_threshold=3, coverage_threshold_mm=4)
            self.assertEqual(read_ids(processed / "final_pre_process.csv"), ["S_1_10"])
            self.assertTrue((root / "excluded/S_2_10.nii.gz").is_file())
            self.assertEqual(read_ids(processed / "pre_process_errors.csv"), ["S_3_10"])


if __name__ == "__main__":
    unittest.main()
