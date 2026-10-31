"""Legacy source selection must never silently choose an alternative series."""

import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vit_survival_embeding_convert.selection import REQUIRED, read_statistics, select_study


def series(modality, number, uid=None, **updates):
    row = dict(
        study_id="Study_A", series_instance_uid=uid or f"1.2.{number}", series_number=number,
        series_description=modality, series_descriptions=[modality], acquisition_number=1,
        acquisition_numbers=[1], source_file_count=20, discarded_file_count=0,
        missing_acquisition_file_count=0, file_count=20, frame_count=20,
        coverage_mm=100.0, status="selected", flags=[],
    )
    row.update(updates)
    row["unique_id"] = f"{row['study_id']}_{row['series_instance_uid']}_{row['series_number']}"
    return row


class SelectionTests(unittest.TestCase):
    def test_inclusive_boundaries(self):
        for frames in (20, 40):
            with self.subTest(frames=frames):
                _, reasons = select_study([series("t1 axial", 1, frame_count=frames), series("T2", 2)])
                self.assertEqual(reasons, [])

    def test_numeric_maximum_before_quality_no_fallback(self):
        picks, reasons = select_study([
            series("T1", 9), series("T1", 10, coverage_mm=99.9), series("T2", 2),
        ])
        self.assertEqual(picks["T1"]["series_number"], 10)
        self.assertIn("T1:low_coverage", reasons)

    def test_invalid_quality_keeps_winner(self):
        picks, reasons = select_study([
            series("T1", 9), series("T1", 10, frame_count=None), series("T2", 2),
        ])
        self.assertEqual(picks["T1"]["series_number"], 10)
        self.assertIn("T1:invalid_quality", reasons)

    def test_tied_maximum_with_different_uids_rejected(self):
        picks, reasons = select_study([
            series("T1", 10, "1.2.1"), series("T1", 10, "1.2.2"), series("T2", 2),
        ])
        self.assertIsNone(picks["T1"])
        self.assertIn("T1:ambiguous_max_series_number", reasons)

    def test_dual_hit_and_multiple_actual_descriptions_rejected(self):
        _, reasons = select_study([series("T1 T2", 10)])
        self.assertIn("T1:ambiguous_description", reasons)
        self.assertIn("same_series_for_T1_T2", reasons)
        _, reasons = select_study([
            series("T1", 1, series_descriptions=["T1 axial", "T1 coronal"]), series("T2", 2),
        ])
        self.assertIn("T1:multiple_or_missing_series_descriptions", reasons)

    def test_ordinary_multiple_acquisitions_rejected(self):
        _, reasons = select_study([
            series("T1", 1), series("T2", 2, acquisition_number=2,
                acquisition_numbers=[1, 2], source_file_count=40, discarded_file_count=20),
        ])
        self.assertIn("T2:legacy_acquisitions_not_preserved", reasons)

    def test_corrected_t2_acquisition_2_is_accepted(self):
        rows = [series("T1", 1, study_id="R01_Study_002904"), series(
            "T2", 2, study_id="R01_Study_002904", acquisition_number=2,
            acquisition_numbers=[1, 2], source_file_count=46, file_count=23,
            frame_count=23, discarded_file_count=23, flags=["largest_acquisition_selected"],
        )]
        picks, reasons = select_study(rows)
        self.assertEqual(reasons, [])
        self.assertEqual(picks["T2"]["selection_basis"], "corrected_acquisition_2")
        rows[1]["acquisition_number"] = 1
        self.assertTrue(select_study(rows)[1])

    def test_partial_lower_nonmatching_row_invalidates_study(self):
        picks, reasons = select_study([
            series("T1", 9), series("T2", 10), series("localizer", 1, flags=["partial_study_scan"]),
        ])
        self.assertEqual(picks, {"T1": None, "T2": None})
        self.assertEqual(reasons, ["partial_study_scan"])

    def test_impossible_acquisition_counts_are_not_accepted_as_corrected(self):
        _, reasons = select_study([
            series("T1", 1, study_id="R01_Study_002904"), series(
                "T2", 2, study_id="R01_Study_002904", acquisition_number=2,
                acquisition_numbers=[1, 2], source_file_count=20, file_count=20,
                discarded_file_count=0,
            ),
        ])
        self.assertIn("T2:inconsistent_acquisition_selection", reasons)

    def test_all_unknown_acquisition_with_all_files_kept_is_accepted(self):
        _, reasons = select_study([series("T1", 1), series("T2", 2,
            acquisition_number="unknown", acquisition_numbers=[], missing_acquisition_file_count=20,
            flags=["missing_acquisition_number"],
        )])
        self.assertEqual(reasons, [])

    def test_missing_instance_and_wrong_status_rejected(self):
        _, reasons = select_study([
            series("T1", 1, flags=["missing_instance_number"]), series("T2", 2, status="incomplete"),
        ])
        self.assertIn("T1:missing_instance_number", reasons)
        self.assertIn("T2:status_not_selected", reasons)

    def test_read_statistics_identity_duplicate_and_invalid_quality(self):
        def write(path, rows):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=REQUIRED)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list)
                                     else row[key] for key in REQUIRED})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "statistics.csv"
            rows = [series("T1", 9), series("T1", 10, frame_count="unknown"), series("T2", 2)]
            write(path, rows)
            parsed = read_statistics(path)
            picks, reasons = select_study(parsed["Study_A"])
            self.assertEqual(picks["T1"]["series_number"], 10)
            self.assertIn("T1:invalid_quality", reasons)
            bad = deepcopy(rows)
            bad[0]["unique_id"] = "Study_A_9"
            write(path, bad)
            with self.assertRaisesRegex(ValueError, "unique_id"):
                read_statistics(path)
            write(path, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_statistics(path)


if __name__ == "__main__":
    unittest.main()
