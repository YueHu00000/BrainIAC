"""Synthetic grouping audit tests; no real images, pixels or server access."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from check_series_acquisitions import KNOWN_STUDY, inspect_study


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.study_uid = generate_uid()
        self.series_uid = generate_uid()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, study, filename, number=1, acquisition=1, description="T1", uid=None, frames=1):
        path = self.data / study / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID = self.study_uid
        ds.SeriesInstanceUID = uid or self.series_uid
        ds.Modality = "MR"
        if number is not None:
            ds.SeriesNumber = number
        if acquisition is not None:
            ds.AcquisitionNumber = acquisition
        ds.SeriesDescription = description
        ds.InstanceNumber = 1
        if frames != 1:
            ds.NumberOfFrames = frames
        ds.save_as(path, enforce_file_format=True)

    def test_selected_and_unselected_acquisitions(self):
        self.write("001", "low-a", 1, 1)
        self.write("001", "low-b", 1, 2)
        self.write("001", "high-a", 9, 1)
        self.write("001", "high-b", 9, 2)
        self.write("001", "t2", 10, 1, "T2")
        study, groups, _, files, _ = inspect_study("001", [self.data])
        self.assertEqual(study["multiple_acquisition_group_count"], 2)
        by_number = {g["series_number"]: g for g in groups}
        self.assertEqual(by_number["1"]["selected_t1t2"], [])
        self.assertEqual(by_number["9"]["selected_multiple_acquisition_modalities"], ["T1"])
        self.assertEqual(by_number["9"]["uid_with_multiple_acquisitions"], 1)
        self.assertEqual(len(files), 4)

    def test_series_number_collision_and_pair_collision(self):
        self.write("001", "a", 1, 1, uid=generate_uid())
        self.write("001", "b", 1, 1, uid=generate_uid())
        _, groups, pairs, _, _ = inspect_study("001", [self.data])
        self.assertEqual(groups[0]["acquisition_count"], 1)
        self.assertEqual(groups[0]["series_identity_count"], 2)
        self.assertEqual(len(pairs), 1)

    def test_other_description_does_not_make_selected_group_multi(self):
        self.write("001", "a", 5, 1, "AX T1")
        self.write("001", "b", 5, 2, "LOCALIZER", uid=generate_uid())
        _, groups, _, _, _ = inspect_study("001", [self.data])
        self.assertEqual(groups[0]["acquisition_count"], 2)
        self.assertEqual(groups[0]["selected_t1_acquisition_numbers"], ["1"])
        self.assertEqual(groups[0]["selected_multiple_acquisition_modalities"], [])

    def test_missing_acquisition_is_unknown_not_second_value(self):
        self.write("001", "a", acquisition=0)
        self.write("001", "b", acquisition=None)
        study, groups, _, files, errors = inspect_study("001", [self.data])
        self.assertEqual(study["status"], "incomplete")
        self.assertEqual(groups[0]["acquisition_numbers"], ["0"])
        self.assertEqual(groups[0]["missing_acquisition_files"], 1)
        self.assertEqual(files, [])
        self.assertTrue(errors)

    def test_nested_and_unreadable_files(self):
        self.write("001", "a", acquisition=1)
        self.write("001", "nested/b", acquisition=2)
        (self.data / "001" / "bad.txt").write_text("not dicom")
        study, groups, _, _, _ = inspect_study("001", [self.data])
        self.assertEqual(study["selection_status"], "not_evaluable")
        self.assertEqual(groups[0]["acquisition_count"], 2)
        self.assertEqual(study["status"], "incomplete")

    def test_multiframe_missing_number_and_missing_study(self):
        self.write("001", "a", number=None, frames=2)
        study, groups, _, _, errors = inspect_study("001", [self.data])
        self.assertEqual(study["status"], "incomplete")
        self.assertEqual(groups, [])
        self.assertTrue(any("multiframe" in e["reason"] for e in errors))
        self.assertEqual(inspect_study("missing", [self.data])[0]["status"], "study_not_found")

    def test_original_exception_and_tie(self):
        self.write("STUDY_0230", "excluded", 8, 1, "T1")
        self.write("STUDY_0230", "a", 7, 1, "AX T1")
        self.write("STUDY_0230", "b", 7, 2, "BX T1")
        study, groups, _, _, _ = inspect_study("STUDY_0230", [self.data])
        self.assertIn("T1=max_number_tie", study["selection_status"])
        chosen = next(g for g in groups if g["series_number"] == "7")
        self.assertEqual(chosen["selected_t1_acquisition_numbers"], ["1"])

    def test_cli_excludes_known_case_and_preserves_ids(self):
        for study in ("001", KNOWN_STUDY):
            self.write(study, "a", acquisition=1)
            self.write(study, "b", acquisition=2)
        labels = self.root / "labels.csv"
        labels.write_text(f"Study_ID\n001\n{KNOWN_STUDY}\n", encoding="utf-8")
        folders = self.root / "folders.txt"
        folders.write_text("data\n", encoding="utf-8")
        output = self.root / "reports"
        command = [sys.executable, str(Path(__file__).with_name("check_series_acquisitions.py")),
                   "--csv", str(labels), "--folder-list", str(folders), "--output-dir", str(output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        with (output / "multiple_acquisition_groups.csv").open(encoding="utf-8-sig", newline="") as stream:
            hits = list(csv.DictReader(stream))
        self.assertEqual([r["study_id"] for r in hits], ["001"])
        self.assertEqual(json.loads(hits[0]["acquisition_numbers"]), ["1", "2"])
        self.assertIn("Other studies with observed multiple acquisitions: 1", (output / "summary.txt").read_text())
        self.assertTrue((output / "known_case_groups.csv").is_file())
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        labels.write_text("Study_ID\n001\n001\n", encoding="utf-8")
        command[-1] = str(self.root / "unused")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertFalse((self.root / "unused").exists())


if __name__ == "__main__":
    unittest.main()
