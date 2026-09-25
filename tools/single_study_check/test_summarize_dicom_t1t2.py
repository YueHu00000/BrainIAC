"""Synthetic cohort checks; no patient data, pixels or external services."""

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from summarize_dicom_t1t2 import inspect_study, make_distribution, summarize_series


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dicom_cohort_test_")
        self.root = Path(self.temp.name)
        self.study_uid = generate_uid()
        self.series_uid = generate_uid()

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, z=0, instance=1, number=1, description="T1", uid=None):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = self.study_uid
        ds.SeriesInstanceUID = uid or self.series_uid
        ds.SeriesNumber, ds.SeriesDescription = number, description
        ds.InstanceNumber = instance
        ds.Rows, ds.Columns = 256, 256
        ds.PixelSpacing = [0.859375, 0.859375]
        ds.SliceThickness, ds.SpacingBetweenSlices = 5, 7
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [10, 20, z]
        return ds

    def write(self, study, filename, **kwargs):
        directory = self.root / study
        directory.mkdir(exist_ok=True)
        ds = self.dataset(**kwargs)
        ds.save_as(directory / filename, enforce_file_format=True)
        return ds

    def series(self, positions):
        return [(Path(f"slice_{i}"), self.dataset(z=z, instance=i + 1))
                for i, z in enumerate(positions)]

    def test_known_three_slice_nonuniform_case(self):
        row = summarize_series(self.series([0, 63, 70]))
        for key, expected in {"file_count": 3, "frame_count": 3, "unique_slice_count": 3,
                              "gap_min_mm": 7, "gap_median_mm": 35,
                              "gap_max_mm": 63, "coverage_mm": 70}.items():
            self.assertAlmostEqual(row[key], expected, msg=key)
        self.assertTrue(row["flags"])

    def test_duplicate_positions_do_not_inflate_slice_count(self):
        row = summarize_series(self.series([0, 0, 7, 7]))
        self.assertEqual(row["frame_count"], 4)
        self.assertEqual(row["unique_slice_count"], 2)
        self.assertEqual(row["gap_min_mm"], 7)
        self.assertEqual(row["gap_max_mm"], 7)
        self.assertTrue(row["flags"])

    def test_reverse_and_nonmonotonic_order_keep_spatial_gaps(self):
        for positions in ([14, 7, 0], [0, 14, 7]):
            with self.subTest(positions=positions):
                row = summarize_series(self.series(positions))
                self.assertEqual(row["unique_slice_count"], 3)
                self.assertEqual(row["gap_min_mm"], 7)
                self.assertEqual(row["gap_max_mm"], 7)
                self.assertEqual(row["coverage_mm"], 14)

    def test_missing_or_varying_geometry_stays_unknown(self):
        for mode in ("missing", "varying"):
            items = self.series([0, 7, 14])
            if mode == "missing":
                del items[1][1].ImagePositionPatient
            else:
                items[1][1].ImageOrientationPatient = [1, 0, 0, 0, 0, 1]
            row = summarize_series(items)
            self.assertTrue(row["gap_median_mm"] is None or np.isnan(row["gap_median_mm"]))
            self.assertTrue(row["flags"])

    def test_enhanced_frames_and_incomplete_geometry(self):
        ds = self.dataset()
        ds.NumberOfFrames = 3
        ds.PerFrameFunctionalGroupsSequence = []
        for z in (0, 7, 14):
            group, position = Dataset(), Dataset()
            position.ImagePositionPatient = [10, 20, z]
            group.PlanePositionSequence = [position]
            ds.PerFrameFunctionalGroupsSequence.append(group)
        row = summarize_series([(Path("enhanced"), ds)])
        self.assertEqual(row["file_count"], 1)
        self.assertEqual(row["frame_count"], 3)
        self.assertEqual(row["unique_slice_count"], 3)
        self.assertEqual(row["gap_median_mm"], 7)
        # A populated list does not prove all frames have explicit positions.
        del ds.PerFrameFunctionalGroupsSequence[1].PlanePositionSequence
        row = summarize_series([(Path("enhanced"), ds)])
        self.assertTrue(row["unique_slice_count"] is None or np.isnan(row["unique_slice_count"]))
        del ds.PerFrameFunctionalGroupsSequence
        row = summarize_series([(Path("enhanced"), ds)])
        self.assertEqual(row["frame_count"], 3)
        self.assertTrue(row["unique_slice_count"] is None or np.isnan(row["unique_slice_count"]))

    def test_invalid_orientation_and_mixed_coordinate_frames(self):
        items = self.series([0, 7, 14])
        for _, ds in items:
            ds.ImageOrientationPatient = [0, 0, 0, 0, 1, 0]
        row = summarize_series(items)
        self.assertTrue(np.isnan(row["unique_slice_count"]))
        self.assertIn("invalid_orientation", row["flags"])
        items = self.series([0, 7, 14])
        items[1][1].StudyInstanceUID = generate_uid()
        row = summarize_series(items)
        self.assertTrue(np.isnan(row["gap_median_mm"]))
        self.assertIn("incompatible_coordinate_frames", row["flags"])

    def test_max_number_and_lexical_description_tie(self):
        self.write("TEST", "low", number=1, description="T1 complete")
        self.write("TEST", "z", number=9, description="T1 Z")
        self.write("TEST", "a", number=9, description="T1 A")
        self.write("TEST", "t2", number=7, description="t2 axial")
        rows = {r["modality"]: r for r in inspect_study("TEST", [self.root])}
        self.assertEqual(rows["T1"]["series_number"], 9)
        self.assertEqual(rows["T1"]["series_description"], "T1 A")
        self.assertEqual(rows["T2"]["series_number"], 7)

    def test_all_original_exceptions(self):
        for study, excluded in (("STUDY_0230", 8), ("STUDY_0462", 9),
                                ("STUDY_0836", 9), ("STUDY_1109", 4)):
            self.write(study, "excluded", number=excluded, description="T1 T2")
            self.write(study, "kept", number=1, description="T1 T2")
            rows = inspect_study(study, [self.root])
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["series_number"] == 1 for row in rows))

    def test_mixed_uids_remain_one_harvey_group(self):
        self.write("MIXED", "one", description="T1 T2", z=0)
        self.write("MIXED", "two", description="T1 T2", z=7, instance=2, uid=generate_uid())
        rows = inspect_study("MIXED", [self.root])
        self.assertTrue(all(row["file_count"] == 2 for row in rows))
        self.assertTrue(all(row["flags"] for row in rows))

    def test_bad_direct_child_prevents_partial_selection(self):
        self.write("BAD", "good", description="T1 T2")
        (self.root / "BAD" / "readme.txt").write_text("not DICOM", encoding="utf-8")
        rows = inspect_study("BAD", [self.root])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.get("file_count") is None or np.isnan(row["file_count"]) for row in rows))

    def test_empty_missing_and_ambiguous_studies(self):
        (self.root / "EMPTY").mkdir()
        for study in ("EMPTY", "ABSENT"):
            rows = inspect_study(study, [self.root])
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["status"] != "selected" for row in rows))
        other = self.root / "another_root"
        other.mkdir()
        (other / "EMPTY").mkdir()
        self.assertTrue(all(row["status"] == "study_ambiguous"
                            for row in inspect_study("EMPTY", [self.root, other])))

    def test_cli_studies_in_different_roots_default_and_explicit_folder_list(self):
        roots = [self.root / "part1", self.root / "part2"]
        for root, study, count in zip(roots, ("0001", "0002"), (3, 5)):
            directory = root / study
            directory.mkdir(parents=True)
            for modality, number in (("T1", 1), ("T2", 2)):
                for index in range(count):
                    self.dataset(z=7 * index, instance=index + 1, number=number,
                                 description=modality).save_as(
                        directory / f"{study}_{modality}_{index}.dcm", enforce_file_format=True)
        csv_path = self.root / "studies.csv"
        csv_path.write_text("Study_ID\n0001\n0002\n", encoding="utf-8")
        folder_list = self.root / "folder_address.txt"
        folder_list.write_text("\n".join(map(str, roots)) + "\n", encoding="utf-8")
        script = Path(__file__).with_name("summarize_dicom_t1t2.py")
        for explicit in (False, True):
            output = self.root / ("reports_explicit" if explicit else "reports_default")
            command = [sys.executable, "-B", str(script), "--csv", str(csv_path),
                       "--output-dir", str(output)]
            if explicit:
                folder_list.rename(self.root / "custom_roots.txt")
                command += ["--folder-list", str(self.root / "custom_roots.txt")]
            completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"selected_series.csv", "flagged_series.csv", "distribution_summary.csv", "summary.txt"})
            rows = pd.read_csv(output / "selected_series.csv", dtype={"study_id": str})
            self.assertEqual(rows.study_id.tolist(), ["0001", "0001", "0002", "0002"])
            self.assertEqual(rows.status.tolist(), ["selected"] * 4)
            self.assertEqual(rows.file_count.tolist(), [3, 3, 5, 5])
            self.assertEqual(rows.unique_slice_count.tolist(), [3, 3, 5, 5])
            for row in rows.itertuples():
                expected_root = roots[int(row.study_id) - 1]
                self.assertEqual(Path(row.study_directory), expected_root / row.study_id)
                self.assertTrue(all((Path(row.study_directory) / name).is_file()
                                    for name in json.loads(row.file_names)))
            distribution = pd.read_csv(output / "distribution_summary.csv")
            counts = distribution[distribution.metric == "file_count"]
            self.assertEqual(counts.n_valid.tolist(), [2, 2])
            self.assertEqual(counts["median"].tolist(), [4, 4])

    def test_distribution_equal_series_weight_and_missing_denominator(self):
        rows = []
        for study, count in (("SMALL", 3), ("LARGE", 101)):
            row = summarize_series(self.series(list(range(count))))
            row.update(study_id=study, modality="T1", status="selected")
            rows.append(row)
        rows.append({"study_id": "MISSING", "modality": "T1", "status": "missing"})
        result = make_distribution(rows)
        row = result[(result.modality == "T1") & (result.metric == "frame_count")].iloc[0]
        self.assertEqual(row.n_total, 3)
        self.assertEqual(row.n_valid, 2)
        self.assertEqual(row.n_missing, 1)
        self.assertEqual(row["median"], 52)
        self.assertEqual(row["min"], 3)
        self.assertEqual(row["max"], 101)

    def test_selected_files_match_actual_brainiac_converter(self):
        root = Path(__file__).resolve().parents[2]
        sources = (root / "src", root / "git" / "BrainIAC" / "src")
        source = next((path for path in sources
                       if (path / "brainiac_embedding" / "convert_dicom_t1t2.py").is_file()), None)
        if source is None:
            self.skipTest("Optional BrainIAC reference checkout is unavailable")
        sys.path.insert(0, str(source))
        try:
            reference = importlib.import_module("brainiac_embedding.convert_dicom_t1t2")
        finally:
            sys.path.remove(str(source))
        for study, top in (("NORMAL", 9), ("STUDY_0230", 8), ("STUDY_0462", 9),
                           ("STUDY_0836", 9), ("STUDY_1109", 4)):
            self.write(study, "low", number=1, description="T1 T2 fallback")
            self.write(study, "latest_z", number=top, description="T1 Z")
            self.write(study, "latest_a_later", number=top, description="T1 A", instance=10, z=63)
            self.write(study, "latest_a_first", number=top, description="T1 A", instance=1, uid=generate_uid())
            self.write(study, "latest_t2", number=top, description="t2 axial")
            expected = reference.select_t1_t2_rows(reference.group_dicom_files_in_dir(self.root / study), study)
            actual = inspect_study(study, [self.root])
            for wanted, observed in zip(expected, actual):
                self.assertEqual(observed["series_number"], wanted.SeriesNumber)
                self.assertEqual(observed["series_description"], wanted.SeriesDescription)
                self.assertEqual(observed["file_names"], wanted.Filenames)


if __name__ == "__main__":
    unittest.main()
