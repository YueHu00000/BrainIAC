"""UID grouping regressions using synthetic headers without image pixels."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from _series_statistics import save_csv
from summary_dicom_SeriesInstanceUID import inspect_study
from summary_dicom_SeriesInstanceUID_v2 import analyze, read_selected


class UIDStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dicom"
        self.root.mkdir()
        self.uid_a, self.uid_b = generate_uid(), generate_uid()
        self.study_uid = generate_uid()
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def write(self, uid, number=3, acquisition=0, instance=1, z=0, projection=False):
        directory = self.root / "STUDY_1"
        directory.mkdir(exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = self.study_uid
        if uid is not None:
            ds.SeriesInstanceUID = uid
        ds.Modality, ds.SeriesNumber = "MR", number
        if acquisition is not None:
            ds.AcquisitionNumber = acquisition
        ds.InstanceNumber = instance
        ds.SeriesDescription = "FL:A/PJN" if projection else "AX T2"
        ds.ImageType = ["DERIVED", "SECONDARY", "PROJECTION IMAGE", "IVI"] if projection else ["ORIGINAL", "PRIMARY"]
        ds.Rows, ds.Columns = 32, 32
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 5
        ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [0, 0, z]
        self.counter += 1
        ds.save_as(directory / f"{self.counter}.dcm", enforce_file_format=True)

    def save_selected(self, rows):
        path = self.root.parent / "selected.csv"
        save_csv(pd.DataFrame(rows), path)
        return path

    def test_same_number_two_uids_have_independent_maximum_acquisitions(self):
        for uid, acquisitions in ((self.uid_a, (1, 2)), (self.uid_b, (9, 10))):
            for acquisition in acquisitions:
                for instance, z in ((1, 0), (2, 100)):
                    self.write(uid, acquisition=acquisition, instance=instance, z=z)
        rows, study, issues = inspect_study("STUDY_1", [self.root])
        by_uid = {row["series_instance_uid"]: row for row in rows}
        self.assertEqual((study["series_count"], len(issues)), (2, 0))
        for uid, maximum in ((self.uid_a, 2), (self.uid_b, 10)):
            row = by_uid[uid]
            self.assertEqual(row["unique_id"], f"STUDY_1_{uid}_3")
            self.assertEqual((row["acquisition_number"], row["source_file_count"], row["file_count"]), (maximum, 4, 2))
            self.assertEqual((row["coverage_mm"], row["duplicate_plane_count"]), (100, 0))
            self.assertEqual(row["series_instance_uids"], [uid])

    def test_same_uid_different_numbers_remain_separate(self):
        self.write(self.uid_a, number=3)
        self.write(self.uid_a, number=4)
        rows, _, _ = inspect_study("STUDY_1", [self.root])
        self.assertEqual({row["unique_id"] for row in rows}, {f"STUDY_1_{self.uid_a}_3", f"STUDY_1_{self.uid_a}_4"})

    def test_missing_uid_reports_issue_and_does_not_invent_series(self):
        self.write(None)
        self.write(self.uid_a)
        rows, study, issues = inspect_study("STUDY_1", [self.root])
        self.assertEqual(len(rows), 1)
        self.assertEqual(study["status"], "incomplete")
        self.assertIn("Missing SeriesInstanceUID", issues[0]["reason"])
        self.assertIn("partial_study_scan", rows[0]["flags"])

    def test_projection_remains_in_statistics_and_missing_acquisition_is_unknown(self):
        self.write(self.uid_a, acquisition=None, projection=True)
        rows, _, _ = inspect_study("STUDY_1", [self.root])
        self.assertEqual(rows[0]["series_description"], "FL:A/PJN")
        self.assertEqual(rows[0]["acquisition_number"], "unknown")
        self.assertIn("PROJECTION IMAGE", rows[0]["image_types"][0])

    def test_v2_validates_uid_schema_and_id(self):
        self.write(self.uid_a)
        rows, _, _ = inspect_study("STUDY_1", [self.root])
        path = self.save_selected(rows)
        self.assertEqual(read_selected(path).iloc[0].series_instance_uid, self.uid_a)
        for field, value in (("unique_id", "STUDY_1_3"), ("series_instance_uid", "")):
            bad = [dict(rows[0], **{field: value})]
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_selected(self.save_selected(bad))
        old = [{key: value for key, value in rows[0].items() if key != "series_instance_uid"}]
        with self.assertRaisesRegex(ValueError, "summary_dicom_SeriesInstanceUID"):
            read_selected(self.save_selected(old))

    def test_v2_only_excludes_repeated_uid_and_carries_uid_to_slice_records(self):
        for uid, last_z in ((self.uid_a, 0), (self.uid_b, 100)):
            self.write(uid, instance=1)
            self.write(uid, instance=2, z=last_z)
        rows, _, _ = inspect_study("STUDY_1", [self.root])
        results, excluded, slices = analyze(read_selected(self.save_selected(rows)))
        self.assertEqual(results.unique_id.tolist(), [f"STUDY_1_{self.uid_b}_3"])
        self.assertEqual(excluded.unique_id.tolist(), [f"STUDY_1_{self.uid_a}_3"])
        self.assertEqual(set(slices.series_instance_uid), {self.uid_b})

    def test_cli_normal_repeated_empty_cohorts_and_report_privacy(self):
        directory = Path(__file__).parent
        for mode in ("normal", "repeated", "empty"):
            with self.subTest(mode=mode):
                if mode == "normal":
                    self.write(self.uid_a, instance=1)
                    self.write(self.uid_a, instance=2, z=100)
                elif mode == "repeated":
                    self.write(self.uid_b, number=4, instance=1)
                    self.write(self.uid_b, number=4, instance=2)
                study = "ABSENT" if mode == "empty" else "STUDY_1"
                labels = self.root.parent / "labels.csv"
                labels.write_text(f"Study_ID\n{study}\n", encoding="utf-8")
                folders = self.root.parent / "folders.txt"
                folders.write_text(str(self.root), encoding="utf-8")
                first, second = self.root.parent / f"first_{mode}", self.root.parent / f"second_{mode}"
                subprocess.run([sys.executable, str(directory / "summary_dicom_SeriesInstanceUID.py"), "--csv", str(labels),
                                "--folder-list", str(folders), "--output-dir", str(first)], check=True, capture_output=True)
                subprocess.run([sys.executable, str(directory / "summary_dicom_SeriesInstanceUID_v2.py"), "--selected-csv",
                                str(first / "selected_series.csv"), "--output-dir", str(second)], check=True, capture_output=True)
                for path in (second / "csv").glob("*.csv"):
                    text = path.read_text(encoding="utf-8-sig")
                    self.assertNotIn("series_instance_uid", text)
                    self.assertNotIn(self.uid_a, text)
                    self.assertNotIn(self.uid_b, text)
                    self.assertNotIn("STUDY_1", text)
                if mode == "repeated":
                    failures = pd.read_csv(second / "csv_with_abnormal_study_ids" / "excluded_repeated_series.csv")
                    self.assertIn(f"STUDY_1_{self.uid_b}_4", failures.abnormal_unique_ids.iloc[0])


if __name__ == "__main__":
    unittest.main()
