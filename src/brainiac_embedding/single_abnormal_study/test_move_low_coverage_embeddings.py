"""Coverage filtering and real file moves in temporary directories only."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from brainiac_embedding.single_abnormal_study.move_low_coverage_embeddings import (
    low_coverage_ids,
    move_embeddings,
)


class MoveCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.csv = self.root / "low_coverage_clinical.csv"
        self.source = self.root / "embeddings"
        self.target = self.root / "transfer"
        self.source.mkdir()

    def write_csv(self, rows):
        with self.csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Study_ID", "T1_coverage_mm", "T2_coverage_mm"))
            writer.writerows(rows)

    def put(self, relative, content=b"embedding bytes"):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_either_modality_strict_boundary_and_unknown_values(self):
        self.write_csv([
            ("T1_LOW", 100, 150), ("T2_LOW", 150, 100.9), ("AT", 101, 101),
            ("ZERO", 0, "unknown"), ("UNKNOWN_LOW", "unknown", 50),
            ("UNKNOWN", "unknown", ""), ("INVALID", -1, "inf"), ("NAN", "NaN", 150),
        ])
        self.assertEqual(low_coverage_ids(self.csv), {"T1_LOW", "T2_LOW", "ZERO", "UNKNOWN_LOW"})
        self.assertEqual(low_coverage_ids(self.csv, 100), {"ZERO", "UNKNOWN_LOW"})

    def test_exporter_prefix_dedup_and_literal_string_ids(self):
        self.write_csv([("excluded_0001", 50, 150), ("0001", 150, 50),
                        ("excluded_REPEAT", 154, 154), (" NA ", 80, 150)])
        self.assertEqual(low_coverage_ids(self.csv), {"0001", "NA"})

    def test_move_all_model_and_normalization_paths_without_reading_npz(self):
        self.write_csv([("0001", 100, 150), ("KEEP", 101, 150), ("MISSING", 70, 80)])
        relatives = ["0001.npz", "backbone/0001.npz", "mci/0001.npz",
                     "tufts_train_global_v1/0001.npz", "per_study_per_modality_v1/0001.npz"]
        for relative in relatives:
            self.put(relative, relative.encode())
        kept = [self.put("KEEP.npz"), self.put("00010.npz"), self.put("0001.npz.tmp"),
                self.put("run_summary.json")]
        before_csv = self.csv.read_bytes()
        result = move_embeddings(self.csv, self.source, self.target)
        self.assertEqual(result, {"selected_studies": 2, "matched_studies": 1,
                                  "missing_studies": 1, "files": 5, "dry_run": False})
        for relative in relatives:
            self.assertFalse((self.source / relative).exists())
            self.assertEqual((self.target / relative).read_bytes(), relative.encode())
        self.assertTrue(all(path.is_file() for path in kept))
        self.assertEqual(self.csv.read_bytes(), before_csv)
        rerun = move_embeddings(self.csv, self.source, self.target)
        self.assertEqual(rerun["files"], 0)

    def test_dry_run_creates_nothing(self):
        self.write_csv([("A", 50, 150)])
        path = self.put("nested/A.npz")
        result = move_embeddings(self.csv, self.source, self.target, dry_run=True)
        self.assertEqual(result["files"], 1)
        self.assertTrue(path.exists())
        self.assertFalse(self.target.exists())

    def test_later_destination_conflict_prevents_all_moves(self):
        self.write_csv([("A", 50, 150), ("Z", 150, 50)])
        first, last = self.put("A.npz"), self.put("model/Z.npz")
        existing = self.target / "model" / "Z.npz"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"keep destination")
        with self.assertRaises(FileExistsError):
            move_embeddings(self.csv, self.source, self.target)
        self.assertTrue(first.exists() and last.exists())
        self.assertFalse((self.target / "A.npz").exists())
        self.assertEqual(existing.read_bytes(), b"keep destination")

    def test_destination_parent_file_prevents_all_moves(self):
        self.write_csv([("A", 50, 150)])
        first = self.put("A.npz")
        self.put("model/A.npz")
        self.target.mkdir()
        (self.target / "model").write_bytes(b"not a directory")
        with self.assertRaises(NotADirectoryError):
            move_embeddings(self.csv, self.source, self.target)
        self.assertTrue(first.exists())

    def test_overlapping_roots_are_rejected(self):
        self.write_csv([("A", 50, 150)])
        self.put("A.npz")
        for target in (self.source, self.source / "inside", self.root):
            with self.assertRaises(ValueError):
                move_embeddings(self.csv, self.source, target)
        self.assertTrue((self.source / "A.npz").exists())

    def test_invalid_csv_threshold_or_id_is_rejected_before_moves(self):
        self.write_csv([("A", 50, 150)])
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                low_coverage_ids(self.csv, threshold)
        for study_id in ("", "../A", "excluded_", "A/../B"):
            self.write_csv([(study_id, 50, 150)])
            with self.assertRaises(ValueError):
                move_embeddings(self.csv, self.source, self.target)
        self.csv.write_text("Study_ID,T1_coverage_mm\nA,50\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            low_coverage_ids(self.csv)
        self.assertFalse(self.target.exists())

    def test_empty_selection_creates_no_destination(self):
        self.write_csv([("A", 154, 154)])
        self.put("A.npz")
        self.assertEqual(move_embeddings(self.csv, self.source, self.target)["files"], 0)
        self.assertFalse(self.target.exists())

    def test_cli_default_threshold_from_any_working_directory(self):
        self.write_csv([("A", 100, 150), ("B", 101, 150)])
        self.put("A.npz")
        self.put("B.npz")
        script = Path(__file__).with_name("move_low_coverage_embeddings.py").resolve()
        command = [sys.executable, str(script), "--csv", str(self.csv),
                   "--embedding-dir", str(self.source), "--transfer-dir", str(self.target)]
        result = subprocess.run(command + ["--dry-run"], cwd=self.root, capture_output=True,
                                text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.target.exists())
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.target / "A.npz").exists())
        self.assertTrue((self.source / "B.npz").exists())


if __name__ == "__main__":
    unittest.main()
