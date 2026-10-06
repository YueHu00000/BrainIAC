"""Synthetic metadata checks for manifest and compact quality records."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import tempfile
import unittest
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining._common import read_rows, write_rows
from brainIAC_pretraining.manifest import FIELDS as MANIFEST_FIELDS
from brainIAC_pretraining.manifest import build_manifest, read_inputs
from brainIAC_pretraining.quality import FIELDS as QUALITY_FIELDS
from brainIAC_pretraining.quality import build_quality


def dicom(root, number, acquisition, instance, description="AX T2", nested=False):
    directory = root / "Study_1"
    if nested:
        directory /= "nested"
    directory.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.3"
    ds.SeriesInstanceUID = f"1.2.3.{number}"
    ds.FrameOfReferenceUID = "1.2.4"
    ds.Modality, ds.SeriesNumber, ds.SeriesDescription = "MR", number, description
    if acquisition is not None:
        ds.AcquisitionNumber = acquisition
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = 8, 8
    ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [1, 1], 5, 7
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, instance * 7]
    path = directory / f"s{number}_a{acquisition}_i{instance}.dcm"
    ds.save_as(path, enforce_file_format=True)
    return path


def rows(unique_id="Study_1_10", **updates):
    manifest = dict(unique_id=unique_id, study_id="Study_1", series_number="10",
                    study_directory="source/Study_1", acquisition_number="2",
                    file_names=json.dumps(["nested/a.dcm", "nested/b.dcm"]))
    selected = dict(manifest, file_count="23", frame_count="23", unique_slice_count="23",
                    coverage_mm="154", flags='["largest_acquisition_selected"]',
                    duplicate_plane_count="0", position_known_frames="23")
    selected.update(updates)
    return manifest, selected


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)

    def test_maximum_acquisition_is_numeric_and_all_modalities_retained(self):
        tmp_path = self.tmp_path
        root = tmp_path / "dicom"
        for acquisition in (9, 10):
            dicom(root, 1, acquisition, 1)
        dicom(root, 2, 1, 1, description="FLAIR", nested=True)
        dicom(root, 3, 1, 1, description="LOCALIZER")
        manifest, issues = build_manifest(["Study_1"], [root])
        assert not issues
        assert [row["unique_id"] for row in manifest] == ["Study_1_1", "Study_1_2", "Study_1_3"]
        assert manifest[0]["acquisition_number"] == 10
        assert all("a10" in name for name in json.loads(manifest[0]["file_names"]))
        assert json.loads(manifest[1]["file_names"])[0].startswith("nested")
        assert list(manifest[0]) == MANIFEST_FIELDS


    def test_known_dual_23_slice_case_and_missing_acquisition(self):
        tmp_path = self.tmp_path
        root = tmp_path / "dicom"
        for acquisition in (1, 2):
            for instance in range(1, 24):
                dicom(root, 10, acquisition, instance)
        dicom(root, 11, None, 1)
        dicom(root, 12, None, 1)
        dicom(root, 12, 0, 2)
        manifest, _ = build_manifest(["Study_1"], [root])
        assert len(json.loads(manifest[0]["file_names"])) == 23
        assert manifest[0]["acquisition_number"] == 2
        assert manifest[1]["acquisition_number"] == "unknown"
        assert manifest[2]["acquisition_number"] == 0
        assert "a0" in json.loads(manifest[2]["file_names"])[0]

    def test_manifest_multiframe_in_discarded_acquisition_is_fatal(self):
        root = self.tmp_path / "dicom"
        path = dicom(root, 10, 1, 1)
        dataset = pydicom.dcmread(path)
        dataset.NumberOfFrames = 23
        dataset.save_as(path, enforce_file_format=True)
        dicom(root, 10, 2, 1)
        with self.assertRaisesRegex(ValueError, "Study_1_10: unsupported multi-frame DICOM"):
            build_manifest(["Study_1"], [root])


    def test_relative_folder_list_and_missing_study_report(self):
        tmp_path = self.tmp_path
        root = tmp_path / "dicom"
        root.mkdir()
        write_rows(tmp_path / "labels.csv", [{"Study_ID": "Missing"}], ["Study_ID"])
        folder_list = tmp_path / "folders.txt"
        folder_list.write_text("dicom\n", encoding="utf-8")
        ids, roots = read_inputs(tmp_path / "labels.csv", folder_list)
        assert roots == [root.resolve()]
        manifest, issues = build_manifest(ids, roots)
        assert manifest == []
        assert issues == [{"study_id": "Missing", "file": "", "reason": "study_not_found"}]


    def test_quality_rejects_only_affected_series(self):
        cases = [
            ({"unique_slice_count": "22"}, "count_mismatch"),
            ({"frame_count": "unknown"}, "unknown_or_invalid_count"),
            ({"frame_count": "0"}, "unknown_or_invalid_count"),
            ({"frame_count": "23.5"}, "unknown_or_invalid_count"),
            ({"frame_count": "inf"}, "unknown_or_invalid_count"),
            ({"acquisition_number": "1"}, "selected_source_mismatch"),
            ({"study_directory": "wrong"}, "selected_source_mismatch"),
            ({"file_names": '["different.dcm"]'}, "selected_source_mismatch"),
        ]
        for changes, reason in cases:
            with self.subTest(changes=changes):
                manifest1, selected1 = rows(**changes)
                manifest2, selected2 = rows("Study_1_11")
                accepted, rejected = build_quality([manifest1, manifest2], [selected1, selected2])
                self.assertEqual(accepted, [{"unique_id": "Study_1_11", "frame_count": 23, "coverage_mm": 154.0}])
                self.assertEqual(rejected, [{"unique_id": "Study_1_10", "reason": reason}])
                self.assertEqual(list(accepted[0]), QUALITY_FIELDS)

    def test_multiframe_aborts_quality_instead_of_skipping(self):
        manifest, selected = rows(file_count="1")
        with self.assertRaisesRegex(ValueError, "Study_1_10.*file_count=1, frame_count=23"):
            build_quality([manifest], [selected])

    def test_multiframe_cli_fails_without_publishing_quality_csv(self):
        root = self.tmp_path / "dicom"
        path = dicom(root, 10, 2, 1)
        manifest, _ = build_manifest(["Study_1"], [root])
        write_rows(self.tmp_path / "manifest.csv", manifest, MANIFEST_FIELDS)
        dataset = pydicom.dcmread(path)
        dataset.NumberOfFrames = 23
        dataset.save_as(path, enforce_file_format=True)
        output = self.tmp_path / "output"
        package = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(package / "quality.py"),
                                 "--manifest", str(self.tmp_path / "manifest.csv"),
                                 "--output-dir", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Study_1_10", result.stderr)
        self.assertFalse((output / "image_number_coverage.csv").exists())


    def test_missing_statistics_and_unknown_coverage(self):
        manifest1, selected1 = rows(coverage_mm="nan")
        manifest2, _ = rows("Study_1_11")
        accepted, rejected = build_quality([manifest1, manifest2], [selected1])
        assert accepted == [{"unique_id": "Study_1_10", "frame_count": 23, "coverage_mm": ""}]
        assert rejected == [{"unique_id": "Study_1_11", "reason": "missing_statistics"}]


    def test_duplicate_id_stops_instead_of_choosing_silently(self):
        manifest, selected = rows()
        with self.assertRaisesRegex(ValueError, "Duplicate unique_id"):
            build_quality([manifest, manifest], [selected])
        with self.assertRaisesRegex(ValueError, "Duplicate unique_id"):
            build_quality([manifest], [selected, selected])


    def test_inventory_cli_outputs_compact_schemas(self):
        tmp_path = self.tmp_path
        root = tmp_path / "dicom"
        dicom(root, 10, 2, 1)
        write_rows(tmp_path / "labels.csv", [{"Study_ID": "Study_1"}], ["Study_ID"])
        (tmp_path / "folders.txt").write_text("dicom\n", encoding="utf-8")
        # An isolated copy physically lacks the statistics directory.
        package = tmp_path / "src/brainIAC_pretraining"
        shutil.copytree(Path(__file__).resolve().parents[1], package,
                        ignore=shutil.ignore_patterns("stat", "tests", "__pycache__"))
        self.assertFalse((package / "stat").exists())
        for entry in ("manifest", "quality", "convert", "pre_process", "exclude", "train"):
            result = subprocess.run([sys.executable, str(package / f"{entry}.py"), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        output = tmp_path / "output"
        result = subprocess.run([sys.executable, str(package / "manifest.py"),
                                 "--csv", str(tmp_path / "labels.csv"),
                                 "--folder-list", str(tmp_path / "folders.txt"),
                                 "--output-dir", str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        manifest = read_rows(output / "manifest.csv")
        assert list(manifest[0]) == MANIFEST_FIELDS
        result = subprocess.run([sys.executable, str(package / "quality.py"),
                                 "--manifest", str(output / "manifest.csv"),
                                 "--output-dir", str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        quality = read_rows(output / "image_number_coverage.csv")
        assert list(quality[0]) == QUALITY_FIELDS
        assert quality[0]["frame_count"] == "1"
        assert quality[0]["coverage_mm"] == "0.0"

    def test_quality_measures_only_manifest_selected_acquisition(self):
        root = self.tmp_path / "dicom"
        for acquisition in (1, 2):
            for instance in range(1, 24):
                dicom(root, 10, acquisition, instance)
        manifest, _ = build_manifest(["Study_1"], [root])
        # A new larger acquisition must not change the manifest's selection.
        dicom(root, 10, 3, 1)
        accepted, rejected = build_quality(manifest)
        self.assertFalse(rejected)
        self.assertEqual(accepted, [dict(unique_id="Study_1_10", frame_count=23, coverage_mm=154.0)])

    def test_header_geometry_filters_only_the_affected_series(self):
        root = self.tmp_path / "dicom"
        for number in (10, 11, 12, 13, 14, 15):
            for instance in (1, 2):
                path = dicom(root, number, 2, instance)
                ds = pydicom.dcmread(path)
                if number == 10:
                    # Sagittal slices: coverage is normal projection, not z range.
                    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
                    ds.ImagePositionPatient = [instance * 7, 0, 0]
                elif number == 11:
                    ds.ImagePositionPatient = [0, 0, 7]
                elif number == 12 and instance == 2:
                    del ds.ImagePositionPatient
                elif number == 13 and instance == 2:
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 0, 1]
                elif number == 14 and instance == 2:
                    ds.FrameOfReferenceUID = "1.2.5"
                elif number == 15:
                    ds.ImageOrientationPatient = [2, 0, 0, 0, 1, 0]
                ds.save_as(path, enforce_file_format=True)
        manifest, _ = build_manifest(["Study_1"], [root])
        accepted, rejected = build_quality(manifest)
        self.assertEqual(accepted, [dict(unique_id="Study_1_10", frame_count=2, coverage_mm=7.0)])
        self.assertEqual(rejected[0], dict(unique_id="Study_1_11", reason="count_mismatch"))
        self.assertEqual([row["reason"] for row in rejected[1:]], ["unknown_or_invalid_count"] * 4)

    def test_single_frame_shared_geometry_and_plane_tolerance(self):
        root = self.tmp_path / "dicom"
        for instance, position in enumerate((0, 0.009, 7), 1):
            path = dicom(root, 10, 2, instance)
            ds = pydicom.dcmread(path)
            shared, frame, plane, axes = Dataset(), Dataset(), Dataset(), Dataset()
            axes.ImageOrientationPatient = ds.ImageOrientationPatient
            plane.ImagePositionPatient = [0, 0, position]
            shared.PlaneOrientationSequence = Sequence([axes])
            frame.PlanePositionSequence = Sequence([plane])
            ds.SharedFunctionalGroupsSequence = Sequence([shared])
            ds.PerFrameFunctionalGroupsSequence = Sequence([frame])
            del ds.ImagePositionPatient
            del ds.ImageOrientationPatient
            ds.save_as(path, enforce_file_format=True)
        manifest, _ = build_manifest(["Study_1"], [root])
        accepted, rejected = build_quality(manifest)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [dict(unique_id="Study_1_10", reason="count_mismatch")])
        (root / "Study_1/s10_a2_i2.dcm").unlink()
        manifest, _ = build_manifest(["Study_1"], [root])
        accepted, rejected = build_quality(manifest)
        self.assertFalse(rejected)
        self.assertEqual(accepted, [dict(unique_id="Study_1_10", frame_count=2, coverage_mm=7.0)])

if __name__ == "__main__":
    unittest.main()
