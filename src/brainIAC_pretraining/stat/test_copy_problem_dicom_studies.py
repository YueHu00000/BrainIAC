"""Problem rules and complete-study copying using synthetic source folders."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import copy_problem_dicom_studies as copy_studies


def series(study, number, directory, description="AX T2 FLAIR", frames="20", coverage="100"):
    uid = f"1.2.3.{number}"
    return dict(unique_id=f"{study}_{uid}_{number}", study_id=study, series_instance_uid=uid,
                series_number=str(number), acquisition_number="2", frame_count=frames,
                coverage_mm=coverage, series_description=description, study_directory=str(directory))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class CopyProblemStudiesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "dicom/Study_1"
        self.source.mkdir(parents=True)
        self.selected = self.root / "selected.csv"
        self.output = self.root / "review"

    def write_selected(self, rows):
        fields = list(rows[0]) if rows else list(series("Study_1", 1, self.source))
        copy_studies.write_csv(self.selected, rows, fields)

    def run_copy(self):
        return copy_studies.copy_problem_studies(self.selected, self.output, 20, 100)

    def test_or_rules_case_insensitive_and_strict_thresholds(self):
        rows = [series("Study_1", 1, self.source),
                series("Study_1", 2, self.source, description="ax t1"),
                series("Study_1", 3, self.source, description="FLAIR"),
                series("Study_1", 4, self.source, frames="19"),
                series("Study_1", 5, self.source, coverage="99.9"),
                series("Study_1", 6, self.source, description="LOCALIZER", frames="1", coverage="0")]
        problems, unknown = copy_studies.select_problem_series(rows, 20, 100)
        self.assertFalse(unknown)
        self.assertEqual([row["series_number"] for row in problems], ["3", "4", "5", "6"])
        self.assertEqual(problems[-1]["reason"],
                         "no_t1_t2_in_description;low_frame_count;low_coverage")

    def test_unknown_is_not_low_but_description_can_still_select_it(self):
        rows = [series("Study_1", 1, self.source, frames="unknown", coverage="unknown"),
                series("Study_1", 2, self.source, description="PJN", frames="5", coverage="unknown"),
                series("Study_1", 3, self.source, frames="inf", coverage="-1")]
        problems, unknown = copy_studies.select_problem_series(rows, 20, 100)
        self.assertEqual([row["series_number"] for row in problems], ["2"])
        self.assertEqual(problems[0]["reason"], "no_t1_t2_in_description;low_frame_count")
        self.assertEqual(len(unknown), 3)
        self.assertEqual(unknown[0]["reason"], "unknown_frame_count;unknown_coverage")

    def test_copy_entire_study_once_preserving_good_discarded_and_nested_files(self):
        (self.source / "nested/acquisition1").mkdir(parents=True)
        files = {"problem.dcm": b"selected problem", "good.dcm": b"normal sibling",
                 "nested/acquisition1/discarded.dcm": b"discarded acquisition", "notes.txt": b"study notes"}
        for name, content in files.items():
            (self.source / name).write_bytes(content)
        before = {name: (content, (self.source / name).stat().st_mtime_ns) for name, content in files.items()}
        self.write_selected([series("Study_1", 1, self.source, description="PJN"),
                             series("Study_1", 2, self.source, frames="10"),
                             series("Study_1", 3, self.source)])
        problems, reports, _ = self.run_copy()
        self.assertEqual(len(problems), 2)
        self.assertEqual(len(reports), 1)
        self.assertEqual((reports[0]["status"], reports[0]["problem_series_count"]), ("copied", 2))
        for name, content in files.items():
            self.assertEqual((self.output / "Study_1" / name).read_bytes(), content)
            self.assertEqual(((self.source / name).read_bytes(), (self.source / name).stat().st_mtime_ns), before[name])
        self.assertEqual(len(read_csv(self.output / "problem_series.csv")), 2)
        self.assertEqual(read_csv(self.output / "copied_studies.csv")[0]["status"], "copied")
        self.assertEqual(read_csv(self.output / "unknown_metrics.csv"), [])

    def test_copy_failure_does_not_publish_partial_study_and_continues(self):
        second = self.root / "dicom/Study_2"
        second.mkdir()
        (second / "good.dcm").write_bytes(b"complete study")
        self.write_selected([series("Study_1", 1, self.source, description="PJN"),
                             series("Study_2", 2, second, coverage="50")])
        copytree = copy_studies.shutil.copytree

        def fail_first(source, destination, *args, **kwargs):
            if Path(source) == self.source:
                Path(destination).mkdir()
                (Path(destination) / "partial.dcm").write_bytes(b"partial")
                raise OSError("synthetic copy error")
            return copytree(source, destination, *args, **kwargs)

        with patch.object(copy_studies.shutil, "copytree", side_effect=fail_first):
            _, reports, _ = self.run_copy()
        self.assertEqual([row["status"] for row in reports], ["copy_failed", "copied"])
        self.assertIn("synthetic copy error", reports[0]["error"])
        self.assertFalse((self.output / "Study_1").exists())
        self.assertEqual((self.output / "Study_2/good.dcm").read_bytes(), b"complete study")
        self.assertFalse(list(self.output.glob(".copy_*")))

    def test_existing_target_overlap_and_missing_source_fail_before_copy(self):
        self.write_selected([series("Study_1", 1, self.source, description="PJN")])
        self.output.mkdir()
        marker = self.output / "preserve.txt"
        marker.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.run_copy()
        self.assertEqual(marker.read_bytes(), b"preserve")
        nested = self.source / "review"
        with self.assertRaisesRegex(ValueError, "overlap"):
            copy_studies.copy_problem_studies(self.selected, nested, 20, 100)
        self.assertFalse(nested.exists())
        self.write_selected([series("Absent", 1, self.root / "missing", description="PJN")])
        with self.assertRaises(FileNotFoundError):
            copy_studies.copy_problem_studies(self.selected, self.root / "new_review", 20, 100)
        self.assertFalse((self.root / "new_review").exists())

    def test_duplicate_identity_conflicting_study_paths_and_legacy_input_rejected(self):
        row = series("Study_1", 1, self.source)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            copy_studies.select_problem_series([row, row], 20, 100)
        other = series("Study_1", 2, self.root / "other/Study_1")
        with self.assertRaisesRegex(ValueError, "Multiple source"):
            copy_studies.select_problem_series([row, other], 20, 100)
        with self.assertRaisesRegex(ValueError, "SeriesInstanceUID"):
            copy_studies.select_problem_series([dict(row, unique_id="Study_1_1")], 20, 100)
        self.write_selected([{"unique_id": "x", "frame_count": "20", "coverage_mm": "100"}])
        with self.assertRaisesRegex(ValueError, "summary_dicom_SeriesInstanceUID"):
            copy_studies.read_selected(self.selected)
        with self.assertRaisesRegex(ValueError, "Invalid Study_ID"):
            copy_studies.select_problem_series([dict(row, study_id="../outside")], 20, 100)

    def test_empty_cohort_and_invalid_thresholds(self):
        self.write_selected([])
        self.assertEqual(self.run_copy(), ([], [], []))
        self.assertEqual(read_csv(self.output / "problem_series.csv"), [])
        self.assertEqual(read_csv(self.output / "copied_studies.csv"), [])
        for frames, coverage in ((0, 100), (20, -1), (20, float("nan"))):
            with self.subTest(frames=frames, coverage=coverage), self.assertRaises(ValueError):
                copy_studies.select_problem_series([], frames, coverage)

    def test_cli_report_contains_full_identity_and_selected_acquisition_metrics(self):
        (self.source / "image.dcm").write_bytes(b"no image decoding needed")
        self.write_selected([series("Study_1", 3, self.source, description="FL:A/PJN", frames="5", coverage="unknown")])
        result = subprocess.run([sys.executable, str(Path(copy_studies.__file__)),
                                 "--selected-csv", str(self.selected), "--output-dir", str(self.output),
                                 "--frame-count-threshold", "20", "--coverage-threshold-mm", "100"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = read_csv(self.output / "problem_series.csv")[0]
        self.assertEqual((row["unique_id"], row["study_id"], row["series_instance_uid"], row["series_number"]),
                         ("Study_1_1.2.3.3_3", "Study_1", "1.2.3.3", "3"))
        self.assertEqual((row["acquisition_number"], row["frame_count"], row["coverage_mm"]), ("2", "5", "unknown"))
        self.assertEqual((self.output / "Study_1/image.dcm").read_bytes(), b"no image decoding needed")


if __name__ == "__main__":
    unittest.main()
