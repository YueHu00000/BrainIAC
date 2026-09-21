from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

from brainiac_embedding import preprocess_t1t2 as preprocessing


class PreprocessTests(unittest.TestCase):
    def test_existing_pair_skips_without_reading_contents_or_checking_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "processed" / "A"
            target.mkdir(parents=True)
            (target / "T1.nii.gz").touch()
            (target / "T2.nii.gz").write_bytes(b"not a gzip file")
            with (
                mock.patch("gzip.open", side_effect=AssertionError("must not decompress")),
                mock.patch.object(sitk, "ReadImage", side_effect=AssertionError("must not read images")),
                mock.patch.object(np, "isfinite", side_effect=AssertionError("must not scan voxels")),
                mock.patch.object(preprocessing, "check_separate_roots") as check_roots,
                mock.patch.object(preprocessing, "_expected_raw_paths") as check_raw,
                mock.patch.object(preprocessing, "clean_study_scratch") as clean,
                mock.patch.object(preprocessing.subprocess, "run") as run,
            ):
                result = preprocessing.preprocess_study("A", root / "absent", target.parent)
                self.assertEqual(result, (target / "T1.nii.gz", target / "T2.nii.gz"))
                for operation in (check_roots, check_raw, clean, run):
                    operation.assert_not_called()
            self.assertEqual((target / "T1.nii.gz").read_bytes(), b"")
            self.assertEqual((target / "T2.nii.gz").read_bytes(), b"not a gzip file")

    def _make_raw(self, root: Path, study_id: str) -> Path:
        raw = root / "raw" / study_id
        raw.mkdir(parents=True)
        (raw / "T1.nii.gz").write_bytes(b"raw-t1")
        (raw / "T2.nii.gz").write_bytes(b"raw-t2")
        return root / "raw"

    def test_cli_discovers_all_studies_and_allows_explicit_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self._make_raw(root, "B")
            self._make_raw(root, "A")
            (raw / "readme.txt").touch()
            args = ["--raw-root", str(raw), "--processed-root", str(root / "processed")]
            with mock.patch.object(preprocessing, "preprocess_study") as process:
                preprocessing.main(args)
                self.assertEqual([c.args[0] for c in process.call_args_list], ["A", "B"])
                process.reset_mock()
                preprocessing.main(args + ["--study-id", "B"])
                self.assertEqual([c.args[0] for c in process.call_args_list], ["B"])

    def test_discovery_rejects_empty_missing_and_incomplete_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                preprocessing.discover_study_ids(root)
            with self.assertRaises(FileNotFoundError):
                preprocessing.discover_study_ids(root / "absent")
            raw = self._make_raw(root, "A")
            (raw / "A" / "T2.nii.gz").unlink()
            with mock.patch.object(preprocessing, "preprocess_study") as process:
                with self.assertRaisesRegex(FileNotFoundError, "T2.nii.gz"):
                    preprocessing.main(["--raw-root", str(raw), "--processed-root", str(root / "processed")])
                process.assert_not_called()

    def test_original_cli_is_called_and_outputs_are_published_without_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = self._make_raw(root, "STUDY_1")
            processed_root = root / "processed"
            commands = []

            def fake_run(command, cwd, check):
                self.assertTrue(check)
                commands.append((command, cwd))
                output = Path(command[command.index("--output_dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(sitk.GetImageFromArray(np.ones((3, 4, 5), dtype=np.float32)), str(output / "T1_0000.nii.gz"))
                sitk.WriteImage(sitk.GetImageFromArray(np.full((3, 4, 5), 2, dtype=np.float32)), str(output / "T2_0000.nii.gz"))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(preprocessing.subprocess, "run", side_effect=fake_run):
                t1, t2 = preprocessing.preprocess_study("STUDY_1", raw_root, processed_root)

            self.assertEqual(float(sitk.GetArrayFromImage(sitk.ReadImage(str(t1))).mean()), 1.0)
            self.assertEqual(float(sitk.GetArrayFromImage(sitk.ReadImage(str(t2))).mean()), 2.0)
            self.assertEqual(t1.name, "T1.nii.gz")
            self.assertEqual(t2.name, "T2.nii.gz")
            self.assertFalse((processed_root / "STUDY_1" / "T1_0000.nii.gz").exists())
            command, cwd = commands[0]
            self.assertTrue(command[command.index("--input_dir") + 1].endswith("/raw/STUDY_1"))
            self.assertEqual(Path(cwd).name, "preprocessing")

    def test_missing_one_upstream_output_does_not_publish_partial_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = self._make_raw(root, "STUDY_1")
            processed_root = root / "processed"

            def fake_run(command, cwd, check):
                output = Path(command[command.index("--output_dir") + 1])
                sitk.WriteImage(sitk.GetImageFromArray(np.ones((3, 4, 5), dtype=np.float32)), str(output / "T1_0000.nii.gz"))
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(preprocessing.subprocess, "run", side_effect=fake_run),
                self.assertRaisesRegex(RuntimeError, "did not produce"),
            ):
                preprocessing.preprocess_study("STUDY_1", raw_root, processed_root)

            self.assertFalse((processed_root / "STUDY_1").exists())


if __name__ == "__main__":
    unittest.main()
