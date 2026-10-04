"""Synthetic DICOM tests; no patient images or pixels are needed."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from _series_statistics import save_csv
from summariy_dicom import inspect_study, make_distribution
from summary_dicom_v2 import analyze, read_selected, study_table


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dicom"
        self.root.mkdir()
        self.study_uid = generate_uid()
        self.uids = {}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, number, acquisition, instance, z, description="AX T2", study="R01_Study_002904", nested=False):
        directory = self.root / study
        if nested:
            directory /= "nested"
        directory.mkdir(parents=True, exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = self.study_uid
        ds.SeriesInstanceUID = self.uids.setdefault((study, number), generate_uid())
        ds.Modality, ds.SeriesNumber, ds.SeriesDescription = "MR", number, description
        if acquisition is not None:
            ds.AcquisitionNumber = acquisition
        if instance is not None:
            ds.InstanceNumber = instance
        ds.Rows, ds.Columns = 256, 256
        ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [1, 1], 5, 7
        ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [0, 0, z]
        path = directory / f"s{number}_a{acquisition}_i{instance}_{z}.dcm"
        ds.save_as(path, enforce_file_format=True)
        return path

    def table(self):
        rows, _, _ = inspect_study("R01_Study_002904", [self.root])
        path = self.root / "series.csv"
        save_csv(pd.DataFrame(rows), path)
        return read_selected(path)

    def test_max_acquisition_only_and_unique_id_without_acquisition(self):
        for acquisition in (1, 2):
            for i in range(23):
                self.write(10, acquisition, i + 1, i * 7)
        rows, study, issues = inspect_study("R01_Study_002904", [self.root])
        row = rows[0]
        self.assertEqual(row["unique_id"], "R01_Study_002904_10")
        self.assertEqual(row["acquisition_number"], 2)
        self.assertEqual(row["acquisition_numbers"], [1, 2])
        self.assertEqual(row["file_count"], 23)
        self.assertEqual(row["unique_slice_count"], 23)
        self.assertEqual(row["duplicate_plane_count"], 0)
        self.assertEqual(row["discarded_file_count"], 23)
        self.assertEqual(row["coverage_mm"], 154)
        self.assertEqual(study["series_count"], 1)
        self.assertFalse(issues)
        self.assertTrue(all("a2" in name for name in row["file_names"]))

    def test_numeric_maximum_not_string_order(self):
        self.write(4, 9, 1, 0)
        self.write(4, 10, 1, 100)
        self.assertEqual(self.table().iloc[0].acquisition_number, "10")

    def test_all_numbers_and_descriptions_including_nested_files_and_old_exception(self):
        for number, description in [(1, "AX T1"), (2, "AX T1"), (3, "AX T2"), (4, "AX T2 FLAIR"), (8, "LOCALIZER")]:
            self.write(number, 1, 1, 0, description, study="STUDY_0230", nested=number == 4)
        rows, _, _ = inspect_study("STUDY_0230", [self.root])
        self.assertEqual([r["unique_id"] for r in rows], [f"STUDY_0230_{n}" for n in (1, 2, 3, 4, 8)])

    def test_missing_acquisition_is_not_zero_and_not_mixed_with_known_group(self):
        self.write(1, None, 1, 0)
        self.write(2, None, 1, 0)
        self.write(2, 0, 2, 7)
        self.write(2, 1, 3, 14)
        rows, _, _ = inspect_study("R01_Study_002904", [self.root])
        self.assertEqual(rows[0]["acquisition_number"], "unknown")
        self.assertEqual(rows[0]["file_count"], 1)
        self.assertEqual(rows[1]["acquisition_numbers"], [0, 1])
        self.assertEqual(rows[1]["acquisition_number"], 1)
        self.assertEqual(rows[1]["discarded_file_count"], 2)

    def test_repeated_series_does_not_remove_other_series_in_same_study(self):
        self.write(1, 1, 1, 0)
        self.write(1, 1, 2, 0)
        for i, z in enumerate((0, 50, 100), 1):
            self.write(2, 1, i, z)
        table = self.table()
        results, excluded, slices = analyze(table)
        self.assertEqual(results.unique_id.tolist(), ["R01_Study_002904_2"])
        self.assertEqual(excluded.unique_id.tolist(), ["R01_Study_002904_1"])
        self.assertEqual(results.iloc[0].screen_status, "pass")
        self.assertEqual(slices.unique_id.nunique(), 1)
        study = study_table(table, results, excluded).iloc[0]
        self.assertEqual((study.n_series, study.n_excluded, study.n_pass), (2, 1, 1))

    def test_missing_geometry_and_instance_order_remain_unknown(self):
        self.write(1, 1, None, 0)
        self.write(1, 1, 2, 150)
        results, _, slices = analyze(self.table())
        self.assertEqual(results.iloc[0].screen_status, "unknown")
        self.assertTrue(slices.empty)
        path = self.write(2, 1, 1, 0)
        import pydicom
        ds = pydicom.dcmread(path)
        del ds.ImagePositionPatient
        ds.save_as(path, enforce_file_format=True)
        row = self.table().query("series_number == '2'").iloc[0]
        self.assertTrue(np.isnan(row.coverage_mm))

    def test_nonuniform_geometry_retains_original_statistics(self):
        for i, z in enumerate((0, 63, 70), 1):
            self.write(1, 1, i, z)
        row = self.table().iloc[0]
        self.assertEqual((row.coverage_mm, row.unique_slice_count), (70, 3))
        self.assertEqual((float(row.gap_min_mm), float(row.gap_median_mm), float(row.gap_max_mm)), (7, 35, 63))
        rows, _, _ = inspect_study("R01_Study_002904", [self.root])
        self.assertEqual(make_distribution(rows).query("metric == 'coverage_mm'").iloc[0].n_valid, 1)

    def test_cli_end_to_end_and_all_repeated_or_empty_cohorts(self):
        for mode in ("normal", "repeated", "empty"):
            with self.subTest(mode=mode):
                study = f"STUDY_{mode}"
                if mode != "empty":
                    self.write(1, 1, 1, 0, study=study)
                    self.write(1, 2, 1, 0, study=study)
                    self.write(1, 2, 2, 0 if mode == "repeated" else 150, study=study)
                labels = self.root / f"labels_{mode}.csv"
                labels.write_text(f"Study_ID\n{study}\n", encoding="utf-8")
                folders = self.root / "folders.txt"
                folders.write_text(str(self.root), encoding="utf-8")
                # Everything stays in TemporaryDirectory, outside its DICOM root.
                first = self.root.parent / f"first_{mode}"
                second = self.root.parent / f"second_{mode}"
                directory = Path(__file__).parent
                subprocess.run([sys.executable, str(directory / "summariy_dicom.py"), "--csv", str(labels),
                                "--folder-list", str(folders), "--output-dir", str(first)], check=True, capture_output=True)
                subprocess.run([sys.executable, str(directory / "summary_dicom_v2.py"), "--selected-csv",
                                str(first / "selected_series.csv"), "--output-dir", str(second)], check=True, capture_output=True)
                self.assertTrue((second / "summary_v2.txt").is_file())
                quality = pd.read_csv(second / "csv" / "series_quality_v2.csv")
                self.assertEqual(len(quality), 1 if mode == "normal" else 0)


if __name__ == "__main__":
    unittest.main()
