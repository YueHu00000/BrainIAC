"""Description and inclusive frame range with the existing exclusion workflow."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining import exclude_v2
from brainIAC_pretraining._common import read_ids, read_rows, write_rows


class ExcludeV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processed, self.excluded = self.root / "processed", self.root / "excluded"
        self.processed.mkdir()
        self.quality, self.ids = self.root / "quality.csv", self.processed / "pre_process.csv"

    def prepare(self, values):
        rows, ids = [], []
        for number, (description, frames, coverage) in enumerate(values, 1):
            unique_id = f"Study_1_1.2.3.{number}_{number}"
            (self.processed / f"{unique_id}.nii.gz").write_bytes(unique_id.encode())
            ids.append(dict(unique_id=unique_id))
            rows.append(dict(unique_id=unique_id, series_description=description,
                             frame_count=frames, coverage_mm=coverage))
        write_rows(self.quality, rows, ["unique_id", "series_description", "frame_count", "coverage_mm"])
        write_rows(self.ids, ids, ["unique_id"])
        return [row["unique_id"] for row in ids]

    def run_exclude(self, **kwargs):
        return exclude_v2.exclude_files(self.quality, self.ids, self.processed, self.excluded,
                                        coverage_threshold_mm=100, **kwargs)

    def test_defaults_inclusive_boundaries_and_case_insensitive_description(self):
        ids = self.prepare([("AX T1", 20, 100), ("ax t2 flair", 40, 100),
                            ("AX T2", 19, 100), ("ax t1", 41, 100),
                            ("FLAIR", 30, 100), ("", 30, 100), ("T2", 30, 99.9)])
        moved = self.run_exclude()
        self.assertEqual(read_ids(self.processed / "final_pre_process.csv"), ids[:2])
        self.assertEqual([row["reason"] for row in moved],
                         ["low_frame_count", "high_frame_count", "no_t1_t2_in_description",
                          "no_t1_t2_in_description", "low_coverage"])
        self.assertEqual((self.excluded / f"{ids[3]}.nii.gz").read_bytes(), ids[3].encode())
        self.assertEqual(self.run_exclude(), [])
        self.assertEqual(len(read_rows(self.excluded / "excluded_preprocess.csv")), 5)

    def test_configurable_limits_dry_run_and_unknown_metrics(self):
        ids = self.prepare([("T1", 10, 100), ("T2", 50, 100), ("PJN", 51, 100),
                            ("T2", "unknown", 100), ("T1", 30, "unknown")])
        moved = self.run_exclude(frame_count_min=10, frame_count_max=50, dry_run=True)
        self.assertEqual([row["unique_id"] for row in moved], ids[2:])
        self.assertEqual(moved[0]["reason"], "no_t1_t2_in_description;high_frame_count")
        self.assertFalse(self.excluded.exists())
        self.assertFalse((self.processed / "final_pre_process.csv").exists())
        self.assertTrue(all((self.processed / f"{unique_id}.nii.gz").exists() for unique_id in ids))

    def test_missing_description_column_and_target_conflicts_prevent_all_moves(self):
        ids = self.prepare([("PJN", 30, 100), ("T2", 50, 100)])
        rows = read_rows(self.quality)
        rows[1].pop("series_description")
        # A three-column input must not silently exclude every series.
        write_rows(self.quality, [{key: value for key, value in row.items() if key != "series_description"}
                                 for row in rows], ["unique_id", "frame_count", "coverage_mm"])
        with self.assertRaisesRegex(ValueError, "quality_v2.py"):
            self.run_exclude()
        self.assertTrue(all((self.processed / f"{unique_id}.nii.gz").exists() for unique_id in ids))
        self.prepare([("PJN", 30, 100), ("T2", 50, 100)])
        self.excluded.mkdir()
        (self.excluded / f"{ids[1]}.nii.gz").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.run_exclude()
        self.assertTrue((self.processed / f"{ids[0]}.nii.gz").is_file())
        self.assertEqual((self.excluded / f"{ids[1]}.nii.gz").read_bytes(), b"preserve")

    def test_invalid_ranges_fail_without_writes_and_missing_quality_is_excluded(self):
        ids = self.prepare([("T1", 30, 100)])
        for minimum, maximum in ((0, 40), (41, 40)):
            with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                self.run_exclude(frame_count_min=minimum, frame_count_max=maximum)
        write_rows(self.quality, [], ["unique_id", "series_description", "frame_count", "coverage_mm"])
        self.assertEqual(self.run_exclude(), [dict(unique_id=ids[0], reason="missing_quality")])

    def test_cli_defaults_and_default_range_filtering(self):
        ids = self.prepare([("t1", 20, 100), ("t2", 40, 100), ("T1", 41, 100)])
        arguments = ["--quality-csv", str(self.quality), "--pre-process-csv", str(self.ids),
                     "--input-dir", str(self.processed), "--excluded-dir", str(self.excluded),
                     "--coverage-threshold-mm", "100"]
        args = exclude_v2.parse_args(arguments)
        self.assertEqual((args.frame_count_min, args.frame_count_max), (20, 40))
        result = subprocess.run([sys.executable, str(Path(exclude_v2.__file__)), *arguments],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_ids(self.processed / "final_pre_process.csv"), ids[:2])
        self.assertEqual(read_rows(self.excluded / "excluded_preprocess.csv"),
                         [dict(unique_id=ids[2], reason="high_frame_count")])


if __name__ == "__main__":
    unittest.main()
