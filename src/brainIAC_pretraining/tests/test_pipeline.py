"""Synthetic files through inventory, conversion, preprocessing, and exclusion."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining import convert, manifest, pre_process, quality
from brainIAC_pretraining._common import read_ids, read_rows, write_rows
from brainIAC_pretraining.exclude import exclude_files
from test_processing import write_dicom


class PipelineTests(unittest.TestCase):
    def test_same_number_distinct_uids_stay_independent_through_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dicom_root = root / "dicom"
            cases = [("S_1", "1.2.826.0.1.141", 3, 200),
                     ("S_1", "1.2.826.0.1.142", 2, 400),
                     ("S_3", "1.2.826.0.1.143", 3, 600)]
            ids = [f"{study}_{uid}_10" for study, uid, _, _ in cases]
            for study, uid, slices, value in cases:
                study_dir = dicom_root / study
                study_dir.mkdir(parents=True, exist_ok=True)
                for acquisition in (1, 2):
                    for index in range(slices):
                        write_dicom(study_dir / f"{uid}_a{acquisition}_{index}.dcm", uid,
                                    acquisition, index, acquisition * value + index)
            labels, folders = root / "labels.csv", root / "folders.txt"
            write_rows(labels, ({"Study_ID": study} for study in ("S_1", "S_3")), ["Study_ID"])
            folders.write_text("dicom\n", encoding="utf-8")
            scripts = Path(manifest.__file__).parent
            inventory = root / "inventory"
            subprocess.run([sys.executable, str(scripts / "manifest.py"),
                            "--csv", str(labels), "--folder-list", str(folders),
                            "--output-dir", str(inventory)], check=True, capture_output=True, text=True)
            manifest_csv = inventory / "manifest.csv"
            raw, processed = root / "raw", root / "processed"
            self.assertEqual(convert.convert_manifest(manifest_csv, raw), ids)
            for unique_id, (_, _, slices, value) in zip(ids, cases):
                image = sitk.ReadImage(str(raw / f"{unique_id}.nii.gz"))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(image)[:, 0, 0],
                                              np.arange(slices) + 2 * value)
            raw_timestamps = {path.name: path.stat().st_mtime_ns for path in raw.glob("*.nii.gz")}
            self.assertEqual(convert.convert_manifest(manifest_csv, raw), ids)
            self.assertEqual(raw_timestamps, {path.name: path.stat().st_mtime_ns
                                             for path in raw.glob("*.nii.gz")})
            accepted, rejected = quality.build_quality(read_rows(manifest_csv))
            self.assertFalse(rejected)
            quality_csv = root / "image_number_coverage.csv"
            write_rows(quality_csv, accepted, quality.FIELDS)

            def imaging(command, **kwargs):
                inputs = Path(command[command.index("--input_dir") + 1])
                outputs = Path(command[command.index("--output_dir") + 1])
                source = next(inputs.glob("*.nii.gz"))
                if inputs.parent.name.startswith(f"{ids[2]}_"):
                    raise subprocess.CalledProcessError(1, command)
                sitk.WriteImage(sitk.ReadImage(str(source)),
                                str(outputs / "volume_0000.nii.gz"))

            with patch.object(pre_process.subprocess, "run", side_effect=imaging):
                self.assertEqual(pre_process.preprocess_csv(raw / "converted.csv", raw, processed),
                                 ids[:2])
            processed_timestamps = {path.name: path.stat().st_mtime_ns
                                    for path in processed.glob("*.nii.gz")}
            with patch.object(pre_process.subprocess, "run", side_effect=imaging) as imaging_call:
                self.assertEqual(pre_process.preprocess_csv(raw / "converted.csv", raw, processed),
                                 ids[:2])
            self.assertEqual(imaging_call.call_count, 1)  # Only the failed series is retried.
            self.assertEqual(processed_timestamps, {path.name: path.stat().st_mtime_ns
                                                   for path in processed.glob("*.nii.gz")})
            exclude_files(quality_csv, processed / "pre_process.csv", processed, root / "excluded",
                          frame_count_threshold=3, coverage_threshold_mm=4)
            self.assertEqual(read_ids(processed / "final_pre_process.csv"), [ids[0]])
            self.assertTrue((root / "excluded" / f"{ids[1]}.nii.gz").is_file())
            self.assertEqual(read_ids(processed / "pre_process_errors.csv"), [ids[2]])


if __name__ == "__main__":
    unittest.main()
