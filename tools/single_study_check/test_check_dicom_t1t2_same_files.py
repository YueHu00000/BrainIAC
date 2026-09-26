"""Synthetic DICOM selection and file-overlap tests; no patient files are read."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from check_dicom_t1t2_same_files import compare_pair, make_summary
from inspect_dicom_t1 import EXCEPTIONS
from summarize_dicom_t1t2 import inspect_study


class FileOverlapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t1t2_overlap_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = [self.root / "root_a", self.root / "root_b"]
        for root in self.roots:
            root.mkdir()

    def write(self, study, filename, description, number=1, instance=1, root=0):
        directory = self.roots[root] / study
        directory.mkdir(exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
        ds.SeriesNumber, ds.SeriesDescription, ds.InstanceNumber = number, description, instance
        ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.2"  # Deliberately reused: UID alone must not imply same files.
        ds.Rows = ds.Columns = 16
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 3
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, 0]  # Repeated planes must not be excluded by this check.
        ds.save_as(directory / filename, enforce_file_format=True)

    def check(self, study):
        return compare_pair(inspect_study(study, self.roots))

    def test_same_group_means_same_files_in_order_even_with_repeated_planes(self):
        self.write("SAME", "second.dcm", "t1_t2 localizer", number=9, instance=2)
        self.write("SAME", "first.dcm", "t1_t2 localizer", number=9, instance=1)
        result = self.check("SAME")
        self.assertEqual(result["comparison_status"], "same_files")
        self.assertIs(result["same_file_order"], True)
        self.assertEqual(result["shared_file_count"], 2)
        self.assertEqual(result["shared_file_names"], ["first.dcm", "second.dcm"])
        self.assertIn("repeated_slice_plane", result["T1_flags"])

    def test_same_uid_and_geometry_but_different_files_are_disjoint(self):
        self.write("DIFFERENT", "t1.dcm", "T1", number=2)
        self.write("DIFFERENT", "t2.dcm", "T2", number=3)
        result = self.check("DIFFERENT")
        self.assertEqual(result["T1_series_instance_uids"], result["T2_series_instance_uids"])
        self.assertEqual(result["comparison_status"], "disjoint")
        self.assertEqual(result["shared_file_count"], 0)

    def test_shared_candidate_is_not_enough_max_number_and_ties_still_apply(self):
        self.write("MAX", "shared.dcm", "T1 T2", number=1)
        self.write("MAX", "t1.dcm", "T1", number=4)
        self.write("MAX", "t2.dcm", "T2", number=5)
        self.assertEqual(self.check("MAX")["comparison_status"], "disjoint")
        self.write("TIE", "shared.dcm", "T1 T2", number=9)
        self.write("TIE", "first.dcm", "T1 A", number=9)
        result = self.check("TIE")
        self.assertEqual(result["comparison_status"], "disjoint")
        self.assertEqual(result["T1_series_description"], "T1 A")
        self.assertEqual(result["T2_series_description"], "T1 T2")
        self.assertIn("max_series_number_tie", result["T1_flags"])

    def test_original_series_number_exceptions_are_applied(self):
        for study, excluded in EXCEPTIONS.items():
            with self.subTest(study=study):
                self.write(study, "shared.dcm", "T1 T2", number=excluded)
                self.write(study, "t1.dcm", "T1", number=1)
                self.write(study, "t2.dcm", "T2", number=2)
                result = self.check(study)
                self.assertEqual(result["comparison_status"], "disjoint")
                self.assertEqual(result["T1_series_number"], 1)

    def test_missing_ambiguous_bad_and_unpaired_studies_are_unknown(self):
        self.write("AMBIGUOUS", "a.dcm", "T1 T2")
        self.write("AMBIGUOUS", "b.dcm", "T1 T2", root=1)
        self.write("BAD", "a.dcm", "T1 T2")
        (self.roots[0] / "BAD" / "readme.txt").write_text("not DICOM")
        self.write("UNPAIRED", "a.dcm", "T1")
        (self.roots[0] / "EMPTY").mkdir()
        for study in ("MISSING", "AMBIGUOUS", "BAD", "UNPAIRED", "EMPTY"):
            with self.subTest(study=study):
                result = self.check(study)
                self.assertEqual(result["comparison_status"], "unknown")
                self.assertEqual(result["same_file_set"], "unknown")
                self.assertEqual(result["shared_file_count"], "unknown")
        report = make_summary([self.check("MISSING")], "labels.csv", "roots.txt", "Study_ID")
        self.assertIn("unknown of assessable studies", report)

    def test_comparison_distinguishes_order_from_set_and_partial_overlap(self):
        t1 = dict(study_id="S", modality="T1", status="selected", file_names=["a", "b"])
        t2 = dict(study_id="S", modality="T2", status="selected", file_names=["b", "a"])
        result = compare_pair([t1, t2])
        self.assertEqual(result["comparison_status"], "same_files")
        self.assertIs(result["same_file_order"], False)
        t2["file_names"] = ["b", "c"]
        result = compare_pair([t1, t2])
        self.assertEqual(result["comparison_status"], "partial_overlap")
        self.assertEqual(result["shared_file_names"], ["b"])

    def test_cli_multiple_roots_default_list_custom_id_and_unknowns(self):
        self.write("0001", "shared.dcm", "T1 T2")
        self.write("NA", "t1.dcm", "T1", number=1, root=1)
        self.write("NA", "t2.dcm", "T2", number=2, root=1)
        pd.DataFrame({"ID": ["0001", "NA", "MISSING"], "clinical": ["x", "y", "z"]}).to_csv(self.root / "labels.csv", index=False)
        (self.root / "folder_address.txt").write_text("root_a\n\nroot_b\n", encoding="utf-8-sig")
        output = self.root / "reports"
        command = [sys.executable, "-B", str(Path(__file__).with_name("check_dicom_t1t2_same_files.py")),
                   "--csv", "labels.csv", "--study-id-column", "ID", "--output-dir", str(output)]
        completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        all_rows = pd.read_csv(output / "study_file_overlap.csv", dtype=str, keep_default_na=False)
        self.assertEqual(all_rows.Study_ID.tolist(), ["0001", "NA", "MISSING"])
        self.assertEqual(all_rows.comparison_status.tolist(), ["same_files", "disjoint", "unknown"])
        self.assertEqual(json.loads(all_rows.iloc[0].shared_file_names), ["shared.dcm"])
        matched = pd.read_csv(output / "overlapping_studies.csv", dtype=str)
        self.assertEqual(matched.Study_ID.tolist(), ["0001"])
        unknown = pd.read_csv(output / "unknown_studies.csv", dtype=str)
        self.assertEqual(unknown.Study_ID.tolist(), ["MISSING"])
        report = (output / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("33.33% of all studies; 50.00% of assessable studies", report)
        completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual((output / "summary.txt").read_text(encoding="utf-8"), report)
        # Explicit --folder-list works from a different working directory too.
        command[command.index("labels.csv")] = str(self.root / "labels.csv")
        command[-1] = str(self.root / "reports_explicit")
        command += ["--folder-list", str(self.root / "folder_address.txt")]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_empty_overlap_subset_retains_report_headers(self):
        self.write("ONLY", "t1.dcm", "T1")
        self.write("ONLY", "t2.dcm", "T2", number=2)
        pd.DataFrame({"Study_ID": ["ONLY"]}).to_csv(self.root / "labels.csv", index=False)
        (self.root / "roots.txt").write_text(str(self.roots[0]), encoding="utf-8")
        command = [sys.executable, "-B", str(Path(__file__).with_name("check_dicom_t1t2_same_files.py")),
                   "--csv", str(self.root / "labels.csv"), "--folder-list", str(self.root / "roots.txt"),
                   "--output-dir", str(self.root / "reports")]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        empty = pd.read_csv(self.root / "reports" / "overlapping_studies.csv")
        self.assertTrue(empty.empty)
        self.assertIn("shared_file_count", empty.columns)


if __name__ == "__main__":
    unittest.main()
