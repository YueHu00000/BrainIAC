"""Filtering boundaries, recoverable movement, and input/output separation."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining._common import read_ids, read_rows, write_rows
from brainIAC_pretraining.exclude import exclude_files


class ExcludeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "pre_process"
        self.input.mkdir()
        self.excluded = self.root / "excluded_preprocess"
        self.ids_csv = self.input / "pre_process.csv"
        self.quality = self.root / "quality.csv"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, values):
        ids = []
        rows = []
        for unique_id, metrics in values.items():
            (self.input / f"{unique_id}.nii.gz").write_bytes(unique_id.encode())
            ids.append({"unique_id": unique_id})
            if metrics is not None:
                rows.append(dict(unique_id=unique_id, frame_count=metrics[0], coverage_mm=metrics[1]))
        write_rows(self.ids_csv, ids, ["unique_id"])
        write_rows(self.quality, rows, ["unique_id", "frame_count", "coverage_mm"])

    def run_exclude(self, **kwargs):
        return exclude_files(self.quality, self.ids_csv, self.input, self.excluded,
                             frame_count_threshold=20, coverage_threshold_mm=100, **kwargs)

    def test_boundaries_unknowns_and_repeat_preserve_history(self):
        self.prepare({"S_1_1.2.826.0.1.121_10": (20, 100), "S_2_1.2.826.0.1.122_10": (19, 100), "S_3_1.2.826.0.1.123_10": (20, 99.9),
                      "S_4_1.2.826.0.1.124_10": (20, ""), "S_5_1.2.826.0.1.125_10": None})
        moved = self.run_exclude()
        self.assertEqual({row["unique_id"] for row in moved}, {"S_2_1.2.826.0.1.122_10", "S_3_1.2.826.0.1.123_10", "S_4_1.2.826.0.1.124_10", "S_5_1.2.826.0.1.125_10"})
        self.assertEqual(read_ids(self.input / "final_pre_process.csv"), ["S_1_1.2.826.0.1.121_10"])
        self.assertEqual((self.excluded / "S_2_1.2.826.0.1.122_10.nii.gz").read_bytes(), b"S_2_1.2.826.0.1.122_10")
        self.assertEqual(self.run_exclude(), [])
        self.assertEqual(len(read_rows(self.excluded / "excluded_preprocess.csv")), 4)

    def test_dry_run_does_not_write_or_move(self):
        self.prepare({"S_1_1.2.826.0.1.121_10": (10, 100)})
        self.assertEqual(len(self.run_exclude(dry_run=True)), 1)
        self.assertFalse(self.excluded.exists())
        self.assertTrue((self.input / "S_1_1.2.826.0.1.121_10.nii.gz").exists())
        self.assertFalse((self.input / "final_pre_process.csv").exists())

    def test_all_conflicts_checked_before_first_move(self):
        self.prepare({"S_1_1.2.826.0.1.121_10": (10, 100), "S_2_1.2.826.0.1.122_10": (10, 100)})
        self.excluded.mkdir()
        (self.excluded / "S_2_1.2.826.0.1.122_10.nii.gz").write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.run_exclude()
        self.assertTrue((self.input / "S_1_1.2.826.0.1.121_10.nii.gz").exists())
        self.assertEqual((self.excluded / "S_2_1.2.826.0.1.122_10.nii.gz").read_bytes(), b"existing")

    def test_nested_roots_refused_without_writes(self):
        self.prepare({"S_1_1.2.826.0.1.121_10": (10, 100)})
        self.excluded = self.input / "excluded"
        with self.assertRaises(ValueError):
            self.run_exclude()
        self.assertFalse(self.excluded.exists())

    def test_path_id_cannot_move_outside_root(self):
        self.prepare({"S_1_1.2.826.0.1.121_10": (10, 100)})
        write_rows(self.ids_csv, [{"unique_id": "../S_1_1.2.826.0.1.121_10"}], ["unique_id"])
        with self.assertRaises(ValueError):
            self.run_exclude()
        self.assertTrue((self.input / "S_1_1.2.826.0.1.121_10.nii.gz").exists())


if __name__ == "__main__":
    unittest.main()
