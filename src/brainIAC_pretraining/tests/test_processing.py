"""Focused checks for selected conversion, independent preprocessing and resume."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brainIAC_pretraining import convert, pre_process
from brainIAC_pretraining._common import MultiFrameDicomError, read_ids, read_rows, write_rows


def write_dicom(path, series_uid, acquisition, index, value):
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    metadata.MediaStorageSOPClassUID = MRImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = metadata.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.PatientID = "synthetic"
    dataset.SeriesNumber = 10
    dataset.AcquisitionNumber = acquisition
    dataset.InstanceNumber = index + 1
    dataset.Rows = dataset.Columns = 4
    dataset.ImagePositionPatient = [0, 0, index * 2]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.PixelSpacing = [1, 1]
    dataset.SliceThickness = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.full((4, 4), value, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_convert_cli_needs_only_manifest_and_resumes_all_series(self):
        source = self.root / "dicom"
        source.mkdir()
        rows = []
        for study, value in (("first", 100), ("second", 200)):
            series_uid = generate_uid()
            names = [f"{study}_{index}.dcm" for index in range(3)]
            for index, name in enumerate(names):
                write_dicom(source / name, series_uid, 1, index, value + index)
            rows.append(dict(unique_id=f"{study}_10", study_directory=str(source),
                             file_names=json.dumps(names)))
        manifest = self.root / "manifest.csv"
        write_rows(manifest, rows, list(rows[0]))
        output = self.root / "raw"
        command = [sys.executable, str(Path(convert.__file__)),
                   "--manifest", str(manifest), "--output-dir", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_ids(output / "converted.csv"), ["first_10", "second_10"])
        for study, value in (("first", 100), ("second", 200)):
            image = sitk.ReadImage(str(output / f"{study}_10.nii.gz"))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image)[:, 0, 0],
                                          [value, value + 1, value + 2])
        timestamps = {path.name: path.stat().st_mtime_ns for path in output.glob("*.nii.gz")}
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(timestamps, {path.name: path.stat().st_mtime_ns for path in output.glob("*.nii.gz")})
        self.assertEqual(read_rows(output / "convert_errors.csv"), [])

    def test_convert_exact_selected_acquisition_failure_and_resume(self):
        dicom = self.root / "dicom"
        dicom.mkdir()
        series_uid = generate_uid()
        selected = []
        for acquisition in (1, 2):
            for index in range(3):
                name = f"a{acquisition}_{index}.dcm"
                write_dicom(dicom / name, series_uid, acquisition, index, acquisition * 100 + index)
                if acquisition == 2:
                    selected.append(name)
        fields = ["unique_id", "study_id", "series_number", "study_directory", "acquisition_number", "file_names"]
        base = dict(study_id="study", series_number="10", study_directory=str(dicom), acquisition_number="2")
        rows = [dict(base, unique_id="missing_10", file_names=json.dumps(["missing.dcm"])),
                dict(base, unique_id="study_10", file_names=json.dumps(selected)),
                dict(base, unique_id="also_missing_11", file_names=json.dumps(["also_missing.dcm"]))]
        manifest = self.root / "manifest.csv"
        write_rows(manifest, rows, fields)
        output = self.root / "raw"
        self.assertEqual(convert.convert_manifest(manifest, output), ["study_10"])
        image = sitk.ReadImage(str(output / "study_10.nii.gz"))
        self.assertEqual(image.GetSize(), (4, 4, 3))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image)[:, 0, 0], [200, 201, 202])
        self.assertEqual(read_ids(output / "convert_errors.csv"), ["missing_10", "also_missing_11"])
        self.assertFalse((output / "also_missing_11.nii.gz").exists())
        (output / "previous_1.nii.gz").write_bytes(b"existing final output")
        # Completed outputs skip without opening or validating the NIfTI.
        with patch.object(sitk.ImageSeriesReader, "Execute", side_effect=AssertionError("must skip")):
            convert.convert_series(rows[1], output)
        self.assertIn("previous_1", convert.convert_manifest(manifest, output))

    def test_preprocess_csv_only_continues_failures_and_omits_masks(self):
        raw, output = self.root / "raw", self.root / "processed"
        raw.mkdir()
        for unique_id in ("failed_1", "too_small_2", "good_3", "unlisted_4"):
            (raw / f"{unique_id}.nii.gz").write_bytes(unique_id.encode())
        converted = self.root / "converted.csv"
        write_rows(converted, [{"unique_id": name} for name in ("failed_1", "too_small_2", "good_3")], ["unique_id"])
        calls = []

        def run(command, **kwargs):
            inputs = Path(command[command.index("--input_dir") + 1])
            outputs = Path(command[command.index("--output_dir") + 1])
            names = list(inputs.glob("*.nii.gz"))
            self.assertEqual(len(names), 1)
            self.assertEqual(names[0].name, "volume.nii.gz")
            unique_id = names[0].read_bytes().decode()
            calls.append(unique_id)
            self.assertTrue(kwargs["check"])
            if unique_id == "failed_1":
                raise subprocess.CalledProcessError(1, command)
            if unique_id == "good_3":
                (outputs / "volume_0000.nii.gz").write_bytes(b"brain")
                (outputs / "volume_0000_mask.nii.gz").write_bytes(b"mask")

        with patch.object(pre_process.subprocess, "run", side_effect=run):
            completed = pre_process.preprocess_csv(converted, raw, output)
        self.assertEqual(completed, ["good_3"])
        self.assertEqual(calls, ["failed_1", "too_small_2", "good_3"])
        self.assertEqual(sorted(path.name for path in output.glob("*.nii.gz")), ["good_3.nii.gz"])
        self.assertEqual(read_ids(output / "pre_process_errors.csv"), ["failed_1", "too_small_2"])
        self.assertIn("no final brain image", read_rows(output / "pre_process_errors.csv")[1]["reason"])
        write_rows(converted, [{"unique_id": "good_3"}], ["unique_id"])
        (output / "previous_5.nii.gz").write_bytes(b"previous")
        with patch.object(pre_process.subprocess, "run", side_effect=AssertionError("must skip")):
            self.assertEqual(pre_process.preprocess_csv(converted, raw, output), ["good_3", "previous_5"])

    def test_interrupt_propagates_and_records_completed_outputs(self):
        raw, output = self.root / "raw", self.root / "processed"
        raw.mkdir()
        converted = self.root / "converted.csv"
        write_rows(converted, [{"unique_id": name} for name in ("good_1", "interrupt_2")], ["unique_id"])

        def process(unique_id, raw_dir, output_dir, python_executable):
            if unique_id == "interrupt_2":
                raise KeyboardInterrupt()
            (output_dir / f"{unique_id}.nii.gz").write_bytes(b"complete")

        with patch.object(pre_process, "preprocess_series", side_effect=process):
            with self.assertRaises(KeyboardInterrupt):
                pre_process.preprocess_csv(converted, raw, output)
        self.assertEqual(read_ids(output / "pre_process.csv"), ["good_1"])

    def test_preprocess_staging_preserves_ids_with_dots_and_mask(self):
        unique_id = "Study.1_mask_10"
        raw, output = self.root / "raw", self.root / "processed"
        raw.mkdir()
        (raw / f"{unique_id}.nii.gz").write_bytes(b"raw volume")
        converted = self.root / "converted.csv"
        write_rows(converted, [{"unique_id": unique_id}], ["unique_id"])

        def run(command, **kwargs):
            inputs = Path(command[command.index("--input_dir") + 1])
            outputs = Path(command[command.index("--output_dir") + 1])
            self.assertEqual([path.name for path in inputs.iterdir()], ["volume.nii.gz"])
            self.assertEqual((inputs / "volume.nii.gz").read_bytes(), b"raw volume")
            (outputs / "volume_0000.nii.gz").write_bytes(b"processed brain")

        with patch.object(pre_process.subprocess, "run", side_effect=run):
            self.assertEqual(pre_process.preprocess_csv(converted, raw, output), [unique_id])
        self.assertEqual((output / f"{unique_id}.nii.gz").read_bytes(), b"processed brain")
        self.assertFalse((output / "volume_0000.nii.gz").exists())

    def test_conversion_rejects_4d_and_overlapping_output_before_writing(self):
        manifest = self.root / "manifest.csv"
        source = self.root / "dicom"
        source.mkdir()
        write_dicom(source / "time.dcm", generate_uid(), 1, 0, 100)
        row = {"unique_id": "time_1", "study_directory": str(source),
               "file_names": json.dumps(["time.dcm"])}
        write_rows(manifest, [row], list(row))
        image = sitk.GetImageFromArray(np.zeros((2, 3, 4, 5)), isVector=False)
        with patch.object(sitk.ImageSeriesReader, "Execute", return_value=image):
            output = self.root / "raw"
            self.assertEqual(convert.convert_manifest(manifest, output), [])
        self.assertIn("scalar 3D", read_rows(output / "convert_errors.csv")[0]["reason"])
        overlapping = source / "raw"
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            convert.convert_manifest(manifest, overlapping)
        self.assertFalse(overlapping.exists())
        converted = self.root / "converted.csv"
        write_rows(converted, [{"unique_id": "time_1"}], ["unique_id"])
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            pre_process.preprocess_csv(converted, source, overlapping)
        self.assertFalse(overlapping.exists())

    def test_multiframe_dicom_stops_batch_and_rescans_completed_outputs(self):
        source = self.root / "dicom"
        source.mkdir()
        dicom = source / "multi.dcm"
        write_dicom(dicom, generate_uid(), 1, 0, 100)
        dataset = pydicom.dcmread(dicom)
        dataset.NumberOfFrames = 2
        dataset.PixelData *= 2
        dataset.save_as(dicom, enforce_file_format=True)
        manifest = self.root / "manifest.csv"
        row = {"unique_id": "multi_1", "study_directory": str(source),
               "file_names": json.dumps(["multi.dcm"])}
        write_rows(manifest, [row, dict(row, unique_id="later_2")], list(row))
        output = self.root / "raw"
        output.mkdir()
        (output / "previous_3.nii.gz").write_bytes(b"existing completed output")
        with patch.object(convert, "convert_series", wraps=convert.convert_series) as convert_one:
            with self.assertRaisesRegex(MultiFrameDicomError, "multi_1.*NumberOfFrames=2"):
                convert.convert_manifest(manifest, output)
        self.assertEqual(convert_one.call_count, 1)
        self.assertEqual(read_ids(output / "converted.csv"), ["previous_3"])
        errors = read_rows(output / "convert_errors.csv")
        self.assertEqual(errors[0]["unique_id"], "multi_1")
        self.assertIn(str(dicom), errors[0]["reason"])


if __name__ == "__main__":
    unittest.main()
