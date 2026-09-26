"""Synthetic join and cohort tests; no patient data are used."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from export_low_coverage_clinical import IMAGE_COLUMNS, build_table


def pair(study_id, coverage=(90, 120), counts=(18, 24)):
    return [dict(study_id=study_id, modality=modality, status="selected", flags="[]",
                 file_count=count, frame_count=count, unique_slice_count=count,
                 coverage_mm=span, duplicate_plane_count=0)
            for modality, span, count in zip(("T1", "T2"), coverage, counts)]


class LowCoverageClinicalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selected = self.root / "selected_series.csv"
        self.labels = self.root / "labels.csv"

    def write(self, rows, labels):
        pd.DataFrame(rows).to_csv(self.selected, index=False)
        labels.to_csv(self.labels, index=False)

    def test_union_forced_inclusion_join_order_and_column_order(self):
        rows = pair("0002", (110, 85), (22, 17)) + pair("0001", (90, 120))
        rows += pair("REPEAT", (150, 160)) + pair("BOTH", (90, 95)) + pair("BOUNDARY", (100, 101))
        rows[5]["duplicate_plane_count"] = 1  # Include despite both coverages exceeding 100.
        labels = pd.DataFrame({"Study_ID": ["0001", "REPEAT", "0002", "BOTH", "BOUNDARY"],
                               "Sex": ["Female", "Female", "Male", "Female", "Male"],
                               "Age": ["055", "60", "71", "44", "50"], "drop_last": ["x"] * 5})
        self.write(rows, labels)
        result, stats = build_table(self.selected, self.labels)
        self.assertEqual(result.columns.tolist(), ["Study_ID", "Sex", "Age", *IMAGE_COLUMNS])
        self.assertEqual(result.Study_ID.tolist(), ["0002", "0001", "excluded_REPEAT", "BOTH"])
        self.assertEqual(result.Sex.tolist(), ["Male", "Female", "Female", "Female"])
        self.assertEqual(result.Age.tolist(), ["71", "055", "60", "44"])
        self.assertEqual(result.T1_unique_slice_count.tolist(), [22, 18, 18, 18])
        self.assertEqual(result.T2_unique_slice_count.tolist(), [17, 24, 24, 24])
        self.assertEqual(result.T1_coverage_mm.tolist(), [110, 90, 150, 90])
        self.assertEqual(result.T2_coverage_mm.tolist(), [85, 120, 160, 95])
        self.assertEqual((stats["included_repeated_studies"], stats["low_coverage_studies"]), (1, 3))
        self.assertEqual(stats["output_studies"], 4)
        self.assertEqual(stats["unmatched_labels"], 0)
        self.assertEqual((stats["T1_low_coverage_series"], stats["T2_low_coverage_series"]), (2, 2))

    def test_flag_inclusion_missing_labels_and_literal_na(self):
        rows = pair("NA") + pair("MISSING", (70, "unknown")) + pair("FLAG", ("unknown", "unknown"))
        rows[-1]["duplicate_plane_count"] = "unknown"
        rows[-1]["flags"] = json.dumps(["repeated_slice_plane"])
        labels = pd.DataFrame({"drop_first": ["x"], "Study_ID": ["NA"], "Sex": ["NA"], "drop_last": ["y"]})
        self.write(rows, labels)
        result, stats = build_table(self.selected, self.labels)
        self.assertEqual(result.columns.tolist(), ["Study_ID", "Sex", *IMAGE_COLUMNS])
        self.assertEqual(result.Study_ID.tolist(), ["NA", "MISSING", "excluded_FLAG"])
        self.assertEqual(result.Sex.tolist(), ["NA", "", ""])
        self.assertTrue(pd.isna(result.iloc[1].T2_coverage_mm))
        self.assertEqual(stats["unmatched_labels"], 2)
        self.assertEqual(stats["included_repeated_studies"], 1)

    def test_empty_result_invalid_coverage_and_custom_threshold(self):
        rows = pair("BOUNDARY", (100, 101)) + pair("UNKNOWN", ("unknown", "unknown"))
        rows += pair("INVALID", (-1, "inf")) + pair("NOT_SELECTED", (50, 50))
        rows[-1]["status"] = rows[-2]["status"] = "read_error"
        labels = pd.DataFrame({"Sex": ["Male"], "Study_ID": ["BOUNDARY"]})
        self.write(rows, labels)
        result, stats = build_table(self.selected, self.labels)
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["Study_ID", *IMAGE_COLUMNS])
        result, _ = build_table(self.selected, self.labels, 101)
        self.assertEqual(result.Study_ID.tolist(), ["BOUNDARY"])
        with self.assertRaisesRegex(ValueError, "positive"):
            build_table(self.selected, self.labels, 0)

    def test_all_repeated_and_zero_coverage(self):
        rows = pair("REPEAT", (0, 0))
        labels = pd.DataFrame({"Study_ID": ["REPEAT"], "last": ["x"]})
        self.write(rows, labels)
        result, _ = build_table(self.selected, self.labels)
        self.assertEqual(len(result), 1)
        rows[1]["duplicate_plane_count"] = 1
        self.write(rows, labels)
        result, stats = build_table(self.selected, self.labels)
        self.assertEqual(result.Study_ID.tolist(), ["excluded_REPEAT"])
        self.assertEqual(stats["nonrepeated_studies"], 0)
        self.assertEqual(stats["low_coverage_studies"], 0)
        self.assertEqual(stats["output_studies"], 1)

    def test_ambiguous_labels_or_selected_pairs_raise(self):
        labels = pd.DataFrame({"Study_ID": [" A", "A "], "Sex": ["F", "M"], "last": ["x", "y"]})
        self.write(pair("A"), labels)
        with self.assertRaisesRegex(ValueError, "duplicate Study_ID"):
            build_table(self.selected, self.labels)
        self.write(pair("A") + pair(" A "), labels.iloc[:1])
        with self.assertRaisesRegex(ValueError, "after trimming"):
            build_table(self.selected, self.labels)
        self.write(pair("A")[:1], labels.iloc[:1])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_table(self.selected, self.labels)

    def test_cli_header_missing_values_and_no_overwrite(self):
        self.write(pair("0001", (50, "unknown")),
                   pd.DataFrame({"Study_ID": ["OTHER"], "Sex": ["F"], "last": ["x"]}))
        output = self.root / "output" / "clinical.csv"
        command = [sys.executable, "-B", str(Path(__file__).with_name("export_low_coverage_clinical.py")),
                   "--selected-csv", str(self.selected), "--labels-csv", str(self.labels), "--output-csv", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("unmatched_labels: 1", completed.stdout)
        result = pd.read_csv(output, dtype=str, keep_default_na=False)
        self.assertEqual(result.iloc[0].Study_ID, "0001")
        self.assertEqual(result.iloc[0].Sex, "")
        self.assertEqual(result.iloc[0].T2_coverage_mm, "unknown")
        before = output.read_bytes()
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
