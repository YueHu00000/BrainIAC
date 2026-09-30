"""Synthetic regression tests; no protected images or model weights required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from brainiac_embedding.single_abnormal_study.convert_single_study import convert_study
from brainiac_embedding.single_abnormal_study.preprocess_single_study import (
    preprocess_study,
)


def write_dicom(path, study_uid, series_uid, series, acquisition, z, instance, value):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.PatientName, ds.PatientID, ds.Modality = "Synthetic", "TEST", "MR"
    ds.SeriesNumber, ds.SeriesDescription = series, "AX T1" if series == 7 else "AX T2"
    ds.AcquisitionNumber, ds.InstanceNumber = acquisition, instance
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [10, 20, 30 + 7 * z]
    ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [0.5, 0.5], 5.5, 7
    ds.Rows, ds.Columns = 8, 9
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.full((8, 9), value, dtype=np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "dicom", self.root / "raw"
        self.source.mkdir()
        study_uid = generate_uid()
        for series, acquisitions in ((7, (1,)), (4, (1, 2))):
            uid = generate_uid()
            for acquisition in acquisitions:
                for z in range(23):
                    # Filenames and InstanceNumber run opposite to geometry.
                    path = self.source / f"s{series}_a{acquisition}_{22-z:02}.dcm"
                    base = 300 if series == 7 else 100 * acquisition
                    write_dicom(path, study_uid, uid, series, acquisition, z,
                                (acquisition - 1) * 23 + 23 - z, base + z)
        # A lower-numbered T2 must not replace the upstream-selected series 4.
        uid = generate_uid()
        for z in range(2):
            write_dicom(self.source / f"other{z}.dcm", study_uid, uid, 3, 1, z, z + 1, 999)
        (self.source / "notes.txt").write_text("not a DICOM", encoding="utf-8")

    def test_split_preserves_pixels_and_seven_mm_spacing(self):
        convert_study(self.source, self.output)
        for name, base in (("T1", 300), ("T2_acq1", 100), ("T2_acq2", 200)):
            image = sitk.ReadImage(str(self.output / f"{name}.nii.gz"))
            self.assertEqual(image.GetSize(), (9, 8, 23))
            np.testing.assert_allclose(image.GetSpacing(), (0.5, 0.5, 7))
            np.testing.assert_allclose(image.GetOrigin(), (10, 20, 30))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image)[:, 0, 0], base + np.arange(23))
            np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((0, 0, 22)), (10, 20, 184))

    def test_legacy_merge_has_46_slices(self):
        paths = [str(self.source / f"s4_a{a}_{22-z:02}.dcm") for a in (1, 2) for z in range(23)]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(paths)
        image = reader.Execute()
        self.assertEqual(image.GetSize()[2], 46)
        self.assertAlmostEqual(image.GetSpacing()[2], 154 / 45)

    def test_missing_acquisition_does_not_create_outputs(self):
        for path in self.source.glob("s4_a2_*.dcm"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "acquisitions 1 and 2"):
            convert_study(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_repeated_plane_within_acquisition_is_rejected(self):
        path = self.source / "s4_a1_00.dcm"
        ds = pydicom.dcmread(path)
        ds.ImagePositionPatient = [10, 20, 30]
        ds.save_as(path, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, "repeated planes"):
            convert_study(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_outputs_are_preserved(self):
        self.output.mkdir()
        existing = self.output / "T1.nii.gz"
        existing.write_bytes(b"keep original")
        with self.assertRaises(FileExistsError):
            convert_study(self.source, self.output)
        self.assertEqual(existing.read_bytes(), b"keep original")

    def test_preprocessing_cannot_overwrite_its_input(self):
        for name in ("T1.nii.gz", "T2.nii.gz"):
            (self.source / name).write_bytes(b"keep original")
        with self.assertRaises(ValueError):
            preprocess_study("STUDY", self.source, self.source)
        self.assertEqual((self.source / "T2.nii.gz").read_bytes(), b"keep original")


if __name__ == "__main__":
    unittest.main()
