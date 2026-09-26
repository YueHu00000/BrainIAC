"""Synthetic tests for v2 screening, cohort exclusions and repeated-plane inspection."""

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

from inspect_repeated_slice_study import build_report, extract_frames
from summarize_dicom_t1t2 import summarize_series, write_csv
from summarize_dicom_t1t2_v2 import (analyze, count_tables, read_selected, render_summary,
                                    secondary_checks, study_table, export_reports)


def check_csv_pairs(test, output, count, hidden_ids):
    plain = list((output / "csv").glob("*.csv"))
    test.assertEqual(len(plain), count)
    test.assertEqual({p.name for p in plain}, {p.name for p in (output / "csv_with_abnormal_study_ids").glob("*.csv")})
    for path in plain:
        first = pd.read_csv(path, dtype=str, keep_default_na=False)
        second = pd.read_csv(output / "csv_with_abnormal_study_ids" / path.name, dtype=str, keep_default_na=False)
        test.assertEqual(second.columns[-1], "abnormal_study_ids")
        pd.testing.assert_frame_equal(first, second.iloc[:, :-1])
        test.assertTrue({"study_id", "study_directory", "file_name", "file_names", "StudyInstanceUID"}.isdisjoint(first.columns))
        for study_id in hidden_ids:
            test.assertNotIn(study_id, path.read_text(encoding="utf-8-sig"))


def dataset(z, instance, modality="T1", series_uid="1.2.826.0.1.3680043.10.543.2"):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.10.543.1"
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.3"
    ds.InstanceNumber = instance
    ds.SeriesNumber = 1 if modality == "T1" else 2
    ds.SeriesDescription = modality
    ds.Rows = ds.Columns = 256
    ds.PixelSpacing = [.5, .5]
    ds.SliceThickness = 5
    ds.SpacingBetweenSlices = 7
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [10, 20, z]
    return ds


def record(study, modality, positions):
    items = [(Path(f"{modality}_{i}.dcm"), dataset(z, i+1, modality)) for i, z in enumerate(positions)]
    result = summarize_series(items)
    result.update(study_id=study, modality=modality, status="selected", series_number=1, series_description=modality)
    return result


class SummaryV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="summary_v2_test_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, rows):
        source = self.root / "selected.csv"
        write_csv(rows, source)
        return read_selected(source)

    def pair(self, study, positions):
        return [record(study, modality, positions) for modality in ("T1", "T2")]

    def test_three_slice_error_is_28_mm_and_40_percent(self):
        results, excluded, slices = analyze(self.load(self.pair("SPARSE", [0, 63, 70])))
        row = results.iloc[0]
        self.assertEqual(row.position_max_error_mm, 28)
        self.assertEqual(row.position_max_error_percent, 40)
        self.assertEqual(row.position_bad_slices, 1)
        self.assertEqual(row.coverage_status, "fail")
        self.assertEqual(row.position_status, "fail")
        self.assertEqual(slices[slices.modality == "T1"].equal_spacing_position_mm.tolist(), [0, 35, 70])
        self.assertTrue(excluded.empty)
        self.assertEqual(study_table(results).study_status.tolist(), ["fail"])

    def test_strict_thresholds_and_override(self):
        table = self.load(self.pair("BOUNDARY", [0, 70, 100]))
        results, _, _ = analyze(table)
        self.assertTrue(results.coverage_status.eq("pass").all())  # exactly 100 mm
        self.assertTrue(results.position_status.eq("pass").all())  # exactly 20%
        self.assertTrue(results.position_bad_slices.eq(0).all())
        results, _, _ = analyze(table, 101, 19)
        self.assertTrue(results.coverage_status.eq("fail").all())
        self.assertTrue(results.position_status.eq("fail").all())

    def test_reverse_order_and_original_nonmonotonic_order(self):
        rows = self.pair("REVERSE", [140, 70, 0]) + self.pair("NONMONOTONIC", [0, 140, 70])
        results, _, _ = analyze(self.load(rows))
        self.assertTrue(results[results.study_id == "REVERSE"].position_max_error_mm.eq(0).all())
        self.assertTrue(results[results.study_id == "NONMONOTONIC"].position_max_error_mm.eq(105).all())

    def test_repeat_in_t2_excludes_whole_study_only(self):
        rows = self.pair("KEEP", [0, 70, 140])
        rows += [record("REPEAT", "T1", [0, 70, 140]), record("REPEAT", "T2", [0, 70, 70, 140])]
        results, excluded, slices = analyze(self.load(rows))
        self.assertEqual(set(results.study_id), {"KEEP"})
        self.assertEqual(set(excluded.modality), {"T1", "T2"})
        self.assertEqual(set(slices.study_id), {"KEEP"})
        self.assertEqual(len(study_table(results)), 1)

    def test_unknown_never_becomes_pass_but_known_failure_remains(self):
        rows = self.pair("MISSING", [0, 70, 140])
        rows[0].pop("projected_positions_mm")
        rows += self.pair("SHORT", [0, 7, 14])
        rows[2].pop("projected_positions_mm")
        results, _, _ = analyze(self.load(rows))
        statuses = study_table(results).set_index("study_id").study_status.to_dict()
        self.assertEqual(statuses, {"MISSING": "unknown", "SHORT": "fail"})
        short = results[(results.study_id == "SHORT") & (results.modality == "T1")].iloc[0]
        self.assertTrue(np.isnan(short.position_bad_slices))

    def test_one_plane_and_multiframe_are_not_position_passes(self):
        rows = self.pair("ONE", [0]) + self.pair("MULTIFRAME", [0, 70, 140])
        rows[2]["file_count"] = rows[3]["file_count"] = 1
        results, _, _ = analyze(self.load(rows))
        self.assertTrue(results.position_status.eq("unknown").all())
        self.assertEqual(results.iloc[0].position_reason, "coverage_not_positive")
        self.assertEqual(results.iloc[2].position_reason, "multi_frame_conversion_not_modelled")

    def test_count_tables_are_not_monotonic_quality_assumptions(self):
        rows = self.pair("LOW_PASS", [0, 70, 140])
        rows += self.pair("HIGH_FAIL", [0, 1, 2, 3, 4])
        results, excluded, _ = analyze(self.load(rows))
        exact, thresholds = count_tables(results)
        self.assertEqual(exact.query("modality == 'T1' and unique_slice_count == 3").iloc[0].screen_pass, 1)
        self.assertEqual(exact.query("modality == 'T1' and unique_slice_count == 5").iloc[0].coverage_fail, 1)
        below = thresholds.query("modality == 'T1' and threshold == 6 and side == 'below'").iloc[0]
        self.assertEqual((below.n_series, below.coverage_fail, below.screen_pass), (2, 1, 1))
        report = render_summary("synthetic", results, excluded, study_table(results), exact,
                                secondary_checks(results), 100, 20)
        self.assertIn("Observed boundary N < 6", report)
        self.assertIn("NOT a learned or validated", report)

    def test_secondary_varying_missing_and_geometry_flags_remain_separate(self):
        rows = self.pair("A", [0, 70, 140])
        rows[0]["thickness_max_mm"] = 6
        rows[0]["flags"].append("varying_thickness")
        rows[0]["flags"].append("varying_orientation")
        rows[1]["pixel_row_min_mm"] = np.nan
        results, _, _ = analyze(self.load(rows))
        checks = secondary_checks(results).set_index(["modality", "check"])
        self.assertEqual(checks.loc[("T1", "thickness_within_series_consistency"), "n_flagged"], 1)
        self.assertEqual(checks.loc[("T2", "pixel_row_within_series_consistency"), "n_unknown"], 1)
        self.assertEqual(checks.loc[("T1", "source_flag:varying_orientation"), "n_flagged"], 1)

    def test_all_excluded_still_writes_valid_report(self):
        results, excluded, _ = analyze(self.load(self.pair("REPEAT", [0, 0, 70])))
        exact, thresholds = count_tables(results)
        report = render_summary("synthetic", results, excluded, study_table(results), exact,
                                secondary_checks(results), 100, 20)
        self.assertIn("No studies remain", report)
        self.assertIn("Excluded repeated-plane studies: 1", report)
        output = self.root / "empty"
        output.mkdir()
        table = self.load(self.pair("REPEAT", [0, 0, 70]))
        export_reports(table, results, excluded, analyze(table)[2], study_table(results), exact,
                       thresholds, secondary_checks(results), output)
        check_csv_pairs(self, output, 7, ["REPEAT"])

    def test_input_ids_and_pair_validation(self):
        rows = self.pair("0001", [0, 70, 140]) + self.pair("NA", [0, 70, 140])
        table = self.load(rows)
        self.assertEqual(table.study_id.tolist(), ["0001", "0001", "NA", "NA"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.load(rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(rows + [rows[0]])

    def test_cli_exclusion_and_artifacts(self):
        rows = self.pair("PASS", list(range(0, 141, 7)))
        rows += self.pair("SPARSE", [0, 63, 70])
        rows += [record("REPEAT", "T1", [0, 70, 140]), record("REPEAT", "T2", [0, 0, 70, 140])]
        self.load(rows)
        script = Path(__file__).with_name("summarize_dicom_t1t2_v2.py")
        output = self.root / "v2"
        command = [sys.executable, "-B", str(script), "--selected-csv", str(self.root / "selected.csv"),
                   "--output-dir", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = (output / "summary_v2.txt").read_text(encoding="utf-8")
        self.assertIn("Study pass: 1/2 (50.00%)", report)
        self.assertIn("Study fail: 1/2 (50.00%)", report)
        self.assertIn("Excluded repeated-plane studies: 1", report)
        check_csv_pairs(self, output, 7, ["PASS", "SPARSE", "REPEAT"])
        identified = output / "csv_with_abnormal_study_ids"
        excluded = pd.read_csv(identified / "excluded_repeated_studies.csv")
        self.assertEqual(excluded.modality.tolist(), ["T1", "T2"])
        self.assertEqual(excluded.abnormal_study_ids.map(json.loads).tolist(), [["REPEAT"], ["REPEAT"]])
        series = pd.read_csv(identified / "series_quality_v2.csv", keep_default_na=False)
        self.assertEqual(series.abnormal_study_ids.tolist(), ["", "", '["SPARSE"]', '["SPARSE"]'])
        exact = pd.read_csv(identified / "slice_count_analysis.csv", keep_default_na=False)
        self.assertEqual(exact.query("unique_slice_count == 3").abnormal_study_ids.tolist(), ['["SPARSE"]'] * 2)
        slices = pd.read_csv(identified / "slice_position_deviations.csv", keep_default_na=False)
        self.assertEqual(slices.abnormal_study_ids.ne("").sum(), 2)
        before = (output / "summary_v2.txt").read_text(encoding="utf-8")
        rerun = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertEqual((output / "summary_v2.txt").read_text(encoding="utf-8"), before)

    def test_export_ids_are_specific_to_each_check_and_keep_unknown_blank(self):
        rows = self.pair("ONLY_T1_BAD", [0, 70, 140]) + self.pair("UNKNOWN_CASE", [0, 70, 140])
        rows[0]["coverage_mm"] = 90
        rows[1]["thickness_max_mm"] = 6  # secondary failure, main screen passes
        rows[2].pop("projected_positions_mm")
        for row in rows:
            row["study_directory"] = f"/source/{row['study_id']}"
            row["file_names"] = [f"{row['study_id']}_{i}.dcm" for i in range(3)]
        table = self.load(rows)
        results, excluded, slices = analyze(table)
        exact, thresholds = count_tables(results)
        output = self.root / "specific"
        output.mkdir()
        export_reports(table, results, excluded, slices, study_table(results), exact, thresholds,
                       secondary_checks(results), output)
        check_csv_pairs(self, output, 7, ["ONLY_T1_BAD", "UNKNOWN_CASE"])
        identified = output / "csv_with_abnormal_study_ids"
        series = pd.read_csv(identified / "series_quality_v2.csv", keep_default_na=False)
        self.assertEqual(series.abnormal_study_ids.tolist(), ['["ONLY_T1_BAD"]', "", "", ""])
        checks = pd.read_csv(identified / "secondary_checks.csv", keep_default_na=False)
        row = checks.query("modality == 'T2' and check == 'thickness_within_series_consistency'").iloc[0]
        self.assertEqual(json.loads(row.abnormal_study_ids), ["ONLY_T1_BAD"])
        studies = pd.read_csv(identified / "study_quality_v2.csv", keep_default_na=False)
        self.assertEqual(studies.study_status.tolist(), ["fail", "unknown"])
        self.assertEqual(studies.abnormal_study_ids.tolist(), ['["ONLY_T1_BAD"]', ""])


class RepeatedStudyTests(unittest.TestCase):
    def test_repeated_groups_show_distinct_echoes_and_same_sop_files(self):
        with tempfile.TemporaryDirectory(prefix="repeated_study_") as temporary:
            study = Path(temporary) / "DEMO"
            study.mkdir()
            for modality, positions in (("T1", [0, 7, 14]), ("T2", [0, 7, 7, 14])):
                shared_sop = generate_uid()
                for index, z in enumerate(positions):
                    ds = dataset(z, index+1, modality)
                    ds.EchoNumbers = 2 if index == 2 else 1
                    if modality == "T2" and index in (1, 2):
                        ds.SOPInstanceUID = shared_sop
                        ds.file_meta.MediaStorageSOPInstanceUID = shared_sop
                    ds.save_as(study / f"{modality}_{index}.dcm", enforce_file_format=True)
            rows, frames, groups, report = build_report(study, study.name)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["EchoNumbers"], ["1", "2"])
            self.assertEqual(groups[0]["excess_frames"], 1)
            self.assertEqual(groups[0]["modality"], "T2")
            self.assertIn("SOP UIDs reused by different files: 1", report)
            self.assertEqual(sum(r["repeated_plane"] is True for r in frames), 2)
            script = Path(__file__).with_name("inspect_repeated_slice_study.py")
            output = Path(temporary) / "report"
            completed = subprocess.run([sys.executable, "-B", str(script), str(study), "--output-dir", str(output)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            check_csv_pairs(self, output, 3, ["DEMO", shared_sop])
            table = pd.read_csv(output / "csv_with_abnormal_study_ids" / "repeated_plane_groups.csv")
            self.assertEqual(json.loads(table.iloc[0].EchoNumbers), ["1", "2"])
            self.assertEqual(json.loads(table.iloc[0].abnormal_study_ids), ["DEMO"])
            frame_table = pd.read_csv(output / "csv_with_abnormal_study_ids" / "frames.csv", keep_default_na=False)
            repeated = frame_table[frame_table.abnormal_study_ids.ne("")]
            self.assertEqual(len(repeated), 2)
            self.assertEqual(repeated.sop_index.nunique(), 1)

    def test_enhanced_temporal_metadata_and_missing_geometry(self):
        with tempfile.TemporaryDirectory(prefix="repeated_enhanced_") as temporary:
            directory = Path(temporary)
            ds = dataset(0, 1)
            ds.NumberOfFrames = 3
            ds.PerFrameFunctionalGroupsSequence = []
            for index, z in enumerate([0, 0, 7]):
                group, plane, content = Dataset(), Dataset(), Dataset()
                plane.ImagePositionPatient = [10, 20, z]
                content.TemporalPositionIndex = index + 1
                group.PlanePositionSequence = [plane]
                group.FrameContentSequence = [content]
                ds.PerFrameFunctionalGroupsSequence.append(group)
            path = directory / "enhanced.dcm"
            ds.save_as(path, enforce_file_format=True)
            row = summarize_series([(path, ds)])
            row.update(study_id="DEMO", modality="T1")
            frames, groups = extract_frames(row, directory)
            self.assertEqual(groups[0]["TemporalPositionIndex"], ["1", "2"])
            self.assertEqual(len(frames), 3)
            del row["projected_positions_mm"]
            frames, groups = extract_frames(row, directory)
            self.assertEqual(groups, [])
            self.assertTrue(all(r["repeated_plane"] == "unknown" for r in frames))


if __name__ == "__main__":
    unittest.main()
