"""Synthetic DICOM counts and CLI checks; no real cohort or pixels."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from summarize_dicom_sequences import inspect_study, summarize_types


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.study_uid = generate_uid()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, study, filename, uid, description="T1", modality="MR"):
        path = self.data / study / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID = self.study_uid
        if uid is not None:
            ds.SeriesInstanceUID = uid
        if description is not None:
            ds.SeriesDescription = description
        ds.Modality = modality
        ds.SeriesNumber = 1
        ds.save_as(path, enforce_file_format=True)

    def test_distinct_series_and_deduplicated_study_coverage(self):
        uid = generate_uid()
        self.write("001", "a", uid)
        self.write("001", "nested/b", uid)
        self.write("001", "c", generate_uid())
        self.write("001", "d", generate_uid(), "FLAIR")
        self.write("001", "ct", generate_uid(), "CT", "CT")
        self.write("002", "a", generate_uid())
        a, sa, _ = inspect_study("001", [self.data])
        b, sb, _ = inspect_study("002", [self.data])
        self.assertEqual((a["series_count"], a["sequence_type_count"], a["mr_file_count"]), (3, 2, 4))
        self.assertEqual(a["non_mr_file_count"], 1)
        types = {row["sequence_description"]: row for row in summarize_types([a, b], sa + sb)}
        self.assertEqual((types["T1"]["study_count"], types["T1"]["series_count"]), (2, 3))
        self.assertEqual(types["FLAIR"]["study_percent"], 50)

    def test_missing_uid_and_unreadable_file_are_not_zero(self):
        self.write("001", "a", None)
        (self.data / "001" / "bad.txt").write_text("not dicom")
        row, series, issues = inspect_study("001", [self.data])
        self.assertEqual((row["status"], row["series_count"], len(issues)), ("incomplete", "unknown", 2))
        self.assertEqual(summarize_types([row], series), [])

    def test_missing_and_conflicting_descriptions_preserve_uid_count(self):
        uid = generate_uid()
        self.write("001", "a", uid, "T1")
        self.write("001", "b", uid, "T2")
        self.write("001", "c", generate_uid(), None)
        row, series, issues = inspect_study("001", [self.data])
        self.assertEqual(row["series_count"], 2)
        self.assertEqual(row["sequence_type_count"], "unknown")
        self.assertEqual(len(issues), 2)
        self.assertEqual(summarize_types([row], series), [])

    def test_missing_ambiguous_and_non_mr_studies(self):
        self.assertEqual(inspect_study("missing", [self.data])[0]["status"], "study_not_found")
        self.write("001", "ct", generate_uid(), "CT", "CT")
        row, _, _ = inspect_study("001", [self.data])
        self.assertEqual((row["status"], row["series_count"]), ("no_mr_series", 0))
        other = self.root / "other"
        (other / "001").mkdir(parents=True)
        self.assertEqual(inspect_study("001", [self.data, other])[0]["status"], "study_ambiguous")

    def test_cli_outputs_and_rejects_overwrite_duplicate_ids(self):
        self.write("001", "a", generate_uid(), "FLAIR")
        labels = self.root / "labels.csv"
        labels.write_text("Study_ID\n001\n", encoding="utf-8")
        folders = self.root / "folders.txt"
        folders.write_text("data\n", encoding="utf-8")
        output = self.root / "reports"
        cmd = [sys.executable, str(Path(__file__).with_name("summarize_dicom_sequences.py")),
               "--csv", str(labels), "--folder-list", str(folders), "--output-dir", str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(output.iterdir())), 6)
        with (output / "study_sequence_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual((row["study_id"], row["series_count"]), ("001", "1"))
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        labels.write_text("Study_ID\n001\n001\n", encoding="utf-8")
        cmd[-1] = str(self.root / "unused")
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        self.assertFalse((self.root / "unused").exists())


if __name__ == "__main__":
    unittest.main()
