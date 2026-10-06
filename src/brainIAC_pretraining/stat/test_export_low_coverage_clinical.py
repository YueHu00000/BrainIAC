"""Series identity, historical selection, threshold boundaries and clinical join."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from export_low_coverage_clinical import IMAGE_COLUMNS, build_table


def series(study, number, coverage, status="selected"):
    return dict(unique_id=f"{study}_{number}", study_id=study, series_number=number,
                status=status, frame_count=23, coverage_mm=coverage)


def pair(study, t1=10, t2=20):
    return [dict(study_id=study, series_number=number, modality=modality, status="selected")
            for modality, number in (("T1", t1), ("T2", t2))]


class LowCoverageSeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selected, self.previous, self.labels = [self.root / name for name in
                                                   ("all.csv", "old.csv", "labels.csv")]

    def write(self, current, old, labels=None):
        pd.DataFrame(current, columns=["unique_id", "study_id", "series_number", "status",
                                       "frame_count", "coverage_mm"]).to_csv(self.selected, index=False)
        pd.DataFrame(old, columns=["study_id", "series_number", "modality", "status"]).to_csv(
            self.previous, index=False)
        if labels is None:
            labels = pd.DataFrame({"Study_ID": ["0001"], "Sex": ["NA"], "Age": ["055"], "last": ["drop"]})
        labels.to_csv(self.labels, index=False)

    def table(self, threshold=100):
        return build_table(self.selected, self.previous, self.labels, threshold)

    def test_multiple_series_actual_previous_selection_and_clinical_columns(self):
        self.write([series("0001", 10, 90), series("0001", 20, 95), series("0001", 30, 80)],
                   pair("0001"))
        result, stats = self.table()
        self.assertEqual(result.columns.tolist(), ["unique_id", "Study_ID", "series_number",
                                                   "Sex", "Age", *IMAGE_COLUMNS])
        self.assertEqual(result.unique_id.tolist(), ["0001_10", "0001_20", "0001_30"])
        self.assertEqual(result.is_previous_t1t2_selected.tolist(), ["yes", "yes", "no"])
        self.assertEqual(result.previous_selected_modality.tolist(), ["T1", "T2", ""])
        self.assertEqual(result.Age.tolist(), ["055"] * 3)
        self.assertEqual(result.Sex.tolist(), ["NA"] * 3)
        self.assertEqual((stats["low_coverage_series"], stats["low_coverage_studies"]), (3, 1))

    def test_strict_boundary_invalid_coverage_and_failed_status(self):
        values = [0, 99.9, 100, 100.5, "unknown", "inf", -1, 70]
        current = [series("0001", n, value) for n, value in enumerate(values, 1)]
        current[-1]["status"] = "read_error"
        current[2]["duplicate_plane_count"] = 5  # Does not force inclusion above threshold.
        self.write(current, pair("0001", 1, 2))
        result, _ = self.table()
        self.assertEqual(result.unique_id.tolist(), ["0001_1", "0001_2"])
        result, _ = self.table(101)
        self.assertEqual(result.unique_id.tolist(), ["0001_1", "0001_2", "0001_3", "0001_4"])
        with self.assertRaisesRegex(ValueError, "positive"):
            self.table(0)

    def test_missing_history_and_partial_history_are_unknown(self):
        old = pair("PARTIAL")
        old[1].update(status="read_error", series_number="unknown")
        self.write([series("MISSING", 10, 70), series("PARTIAL", 10, 75),
                    series("PARTIAL", 30, 80)], old)
        result, stats = self.table()
        self.assertEqual(result.is_previous_t1t2_selected.tolist(), ["unknown", "yes", "unknown"])
        self.assertEqual(result.Sex.tolist(), [""] * 3)
        self.assertEqual(stats["unmatched_labels"], 3)

    def test_same_series_can_have_both_selected_modalities(self):
        self.write([series("0001", 10, 80)], pair("0001", 10, 10))
        result, _ = self.table()
        self.assertEqual(result.previous_selected_modality.tolist(), ["T1;T2"])

    def test_ambiguous_ids_or_labels_raise(self):
        self.write([series("0001", 10, 80)] * 2, pair("0001"))
        with self.assertRaisesRegex(ValueError, "Duplicate unique_id"):
            self.table()
        row = series("0001", 10, 80)
        row["unique_id"] = "wrong"
        self.write([row], pair("0001"))
        with self.assertRaisesRegex(ValueError, "unique_id must equal"):
            self.table()
        self.write([series("0001", 10, 80)], pair("0001") + pair("0001"))
        with self.assertRaisesRegex(ValueError, "Duplicate study/modality"):
            self.table()
        labels = pd.DataFrame({"Study_ID": ["0001", " 0001 "], "Sex": ["F", "M"], "last": ["x", "x"]})
        self.write([series("0001", 10, 80)], pair("0001"), labels)
        with self.assertRaisesRegex(ValueError, "Duplicate Study_ID"):
            self.table()

    def test_cli_empty_header_and_no_overwrite(self):
        self.write([series("0001", 10, 100)], pair("0001"))
        output = self.root / "out/low.csv"
        command = [sys.executable, "-B", str(Path(__file__).with_name("export_low_coverage_clinical.py")),
                   "--selected-csv", str(self.selected), "--old-selected-csv", str(self.previous),
                   "--labels-csv", str(self.labels), "--output-csv", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(pd.read_csv(output).empty)
        before = output.read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
