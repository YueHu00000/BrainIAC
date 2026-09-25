"""Synthetic tests; no patient data or external service is used."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from inspect_dicom_t1 import build_report as dicom_report
from inspect_raw_nifti import build_report as nifti_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="single_study_check_")
        self.root = Path(self.temp.name)
        self.study_uid = generate_uid()

    def tearDown(self):
        self.temp.cleanup()

    def dicom(self, filename, uid, instance=1, description="T1", frames=None):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(self.root / filename), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = self.study_uid
        ds.SeriesInstanceUID = uid
        ds.SeriesNumber = 8
        ds.SeriesDescription = description
        ds.InstanceNumber = instance
        ds.Rows, ds.Columns = 4, 5
        ds.PixelSpacing = [0.8, 0.9]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [10, 20, instance * 5]
        if frames:
            ds.NumberOfFrames = frames
            ds.PerFrameFunctionalGroupsSequence = []
            for index in range(frames):
                group, plane = Dataset(), Dataset()
                plane.ImagePositionPatient = [10, 20, index * 5]
                group.PlanePositionSequence = [plane]
                ds.PerFrameFunctionalGroupsSequence.append(group)
        ds.save_as(self.root / filename, enforce_file_format=True)

    def test_three_files_and_same_number_different_uids(self):
        first, second = generate_uid(), generate_uid()
        for i in range(1, 4):
            self.dicom(f"slice{i}", first, i)
        self.dicom("another_series", second)
        self.dicom("t2", generate_uid(), description="T2")
        (self.root / "readme.txt").write_text("not DICOM", encoding="utf-8")
        report = dicom_report(self.root, "STUDY_0230")
        self.assertIn("T1 series groups: 2", report)
        self.assertIn("File count: 3\nTotal frames: 3", report)
        self.assertIn("Signed position steps in reported order (mm): [5.0, 5.0]", report)
        self.assertIn(first, report)
        self.assertIn(second, report)
        self.assertIn("readme.txt: InvalidDicomError", report)
        self.assertIn("Harvey excluded SeriesNumber for this study ID: 8", report)

    def test_file_input_scans_siblings_and_nested_multiframe(self):
        self.dicom("entry.dcm", generate_uid(), description="T2")
        self.dicom("enhanced.dcm", generate_uid(), description="t1 enhanced", frames=3)
        nested = self.root / "nested"
        nested.mkdir()
        (self.root / "enhanced.dcm").rename(nested / "enhanced.dcm")
        report = dicom_report(self.root / "entry.dcm")
        self.assertIn("File count: 1\nTotal frames: 3", report)
        self.assertIn("Frames/records with position: 3 / 3", report)
        self.assertIn("Distinct positions (rounded to 0.0001 mm): 3", report)

    def test_no_t1_is_reported_explicitly(self):
        self.dicom("t2", generate_uid(), description="T2")
        self.assertIn("T1 series groups: 0", dicom_report(self.root))

    def test_nifti_geometry_scaling_and_extension(self):
        path = self.root / "T1.nii.gz"
        image = nib.Nifti1Image(np.arange(60, dtype=np.int16).reshape(4, 5, 3),
                               np.diag([-0.9, -0.8, 5, 1]))
        image.header.set_xyzt_units("mm")
        image.header.set_slope_inter(2, 10)
        image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"source-series-example"))
        nib.save(image, path)
        report = nifti_report(path)
        self.assertIn("(4, 5, 3)", report)
        self.assertIn("Axis directions: ('L', 'P', 'S')", report)
        self.assertIn("Reader scaling slope/intercept: 2.0, 10.0", report)
        self.assertIn("min=10.0, max=128.0", report)
        self.assertIn("source-series-example", report)

    def test_all_nonfinite_voxels_and_binary_extension(self):
        path = self.root / "nonfinite.nii.gz"
        data = np.array([np.nan, np.inf], dtype=np.float32).reshape(1, 1, 2)
        image = nib.Nifti1Image(data, np.eye(4))
        image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"\xff\xfe"))
        nib.save(image, path)
        report = nifti_report(path)
        self.assertIn("Finite voxels: 0 / 2", report)
        self.assertIn("NaN voxels: 1", report)
        self.assertIn("Inf voxels: 1", report)
        self.assertIn("Binary extension", report)

    def test_cli_reports_and_refuses_overwrite(self):
        self.dicom("slice.dcm", generate_uid())
        nifti = self.root / "T1.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 3)), np.eye(4)), nifti)
        for script, source in (("inspect_dicom_t1.py", self.root), ("inspect_raw_nifti.py", nifti)):
            output = self.root / f"{script}.txt"
            cmd = [sys.executable, str(Path(__file__).parent / script), str(source), "--output", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = output.read_text(encoding="utf-8")
            self.assertIn("REPORT", before)
            again = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(before, output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
