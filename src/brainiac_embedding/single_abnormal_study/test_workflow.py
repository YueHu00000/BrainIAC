"""Test single-study routing with real NIfTI transforms and synthetic encoders."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk
import torch

from brainiac_embedding import extract_embeddings as extraction
from brainiac_embedding import preprocess_t1t2
from brainiac_embedding.single_abnormal_study import embed_single_study as embedding
from brainiac_embedding.single_abnormal_study import (
    preprocess_single_study as preprocessing,
)


class SyntheticEncoder(torch.nn.Module):
    def forward(self, images):
        # Distinguish modality, token position and layer, including final LN.
        offsets = torch.arange(216, device=images.device).view(1, 216, 1)
        modalities = torch.arange(2, device=images.device).view(2, 1, 1) * 1000
        tokens = (offsets + modalities).expand(2, 216, 768).float()
        return tokens + 99, [tokens + layer for layer in range(1, 13)]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = "R01_Study_002904"
        self.raw = self.root / "raw"
        pair = self.raw / self.study
        pair.mkdir(parents=True)
        self.pixels = {}
        grid = np.arange(23 * 8 * 9, dtype=np.float32).reshape(23, 8, 9)
        for index, modality in enumerate(("T1", "T2")):
            pixels = (grid + 1) ** (index + 1)
            self.pixels[modality] = pixels
            image = sitk.GetImageFromArray(pixels)
            image.SetSpacing((0.5, 0.5, 7))
            sitk.WriteImage(image, str(pair / f"{modality}.nii.gz"))
        # The single-study entry must not attempt discovery of this incomplete study.
        (self.raw / "UNRELATED").mkdir()
        (self.raw / "UNRELATED" / "keep.txt").write_text("untouched", encoding="utf-8")
        self.processed = self.root / "processed"

    def fake_preprocess(self, command, cwd, check):
        source = Path(command[command.index("--input_dir") + 1])
        destination = Path(command[command.index("--output_dir") + 1])
        self.assertEqual(source, self.raw / self.study)
        self.assertTrue(check)
        for modality in ("T1", "T2"):
            shutil.copyfile(source / f"{modality}.nii.gz", destination / f"{modality}_0000.nii.gz")
        return subprocess.CompletedProcess(command, 0)

    def test_single_pair_to_three_model_outputs_matches_parent_functions(self):
        with mock.patch.object(preprocess_t1t2.subprocess, "run", side_effect=self.fake_preprocess):
            preprocessing.main(["--raw-root", str(self.raw), "--processed-root", str(self.processed)])
        for modality in ("T1", "T2"):
            path = self.processed / self.study / f"{modality}.nii.gz"
            np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), self.pixels[modality])
        images = extraction.load_study_images(self.study, self.processed)
        self.assertEqual(tuple(images.shape), (2, 1, 96, 96, 96))
        self.assertFalse(torch.equal(images[0], images[1]))
        expected_block, expected_final = extraction.extract_selected_embeddings(SyntheticEncoder(), images)
        (self.processed / "UNRELATED").mkdir()
        for kind in extraction.MODEL_KINDS:
            output_root = self.root / "embeddings" / kind
            with mock.patch.object(extraction, "load_encoder", return_value=SyntheticEncoder()) as loader:
                embedding.main([
                    "--processed-root", str(self.processed), "--output-root", str(output_root),
                    "--model-kind", kind, "--checkpoint", "model.ckpt",
                    "--brainiac-checkpoint", "base.ckpt", "--device", "cpu",
                ])
            loader.assert_called_once_with(
                kind, Path("model.ckpt"), brainiac_checkpoint=Path("base.ckpt"),
                device="cpu", brainiac_root=None,
            )
            self.assertEqual([p.name for p in output_root.iterdir()], [f"{self.study}.npz"])
            with np.load(output_root / f"{self.study}.npz", allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"block_embed", "final_embed"})
                for key, expected in (("block_embed", expected_block), ("final_embed", expected_final)):
                    self.assertEqual(data[key].dtype.names, ("token0", "mean_pooling"))
                    for pooling in extraction.POOLING_NAMES:
                        np.testing.assert_array_equal(data[key][pooling], expected[pooling])
                        self.assertEqual(data[key][pooling].dtype, np.float32)
                        self.assertTrue(np.isfinite(data[key][pooling]).all())
                np.testing.assert_array_equal(data["block_embed"]["token0"][0, :, 0], [3, 6, 9, 12])
                np.testing.assert_array_equal(data["final_embed"]["mean_pooling"][:, 0], [207, 1207])
        self.assertEqual((self.raw / "UNRELATED" / "keep.txt").read_text(), "untouched")

    def test_existing_partial_processed_is_preserved(self):
        target = self.processed / self.study
        target.mkdir(parents=True)
        (target / "T1.nii.gz").write_bytes(b"old result")
        with mock.patch.object(preprocess_t1t2, "preprocess_study") as process, self.assertRaises(FileExistsError):
            preprocessing.preprocess_study(self.study, self.raw, self.processed)
        process.assert_not_called()
        self.assertEqual((target / "T1.nii.gz").read_bytes(), b"old result")

    def test_existing_embedding_or_temporary_file_is_preserved(self):
        for suffix in (".npz", ".npz.tmp"):
            output_root = self.root / suffix[1:]
            output_root.mkdir()
            target = output_root / f"{self.study}{suffix}"
            target.write_bytes(b"old result")
            with mock.patch.object(extraction, "load_study_images") as load, self.assertRaises(FileExistsError):
                embedding.embed_study(self.study, self.processed, output_root,
                                      model_kind="backbone", checkpoint="unused")
            load.assert_not_called()
            self.assertEqual(target.read_bytes(), b"old result")

    def test_missing_raw_pair_does_not_start_preprocessing(self):
        (self.raw / self.study / "T2.nii.gz").unlink()
        with mock.patch.object(preprocess_t1t2, "preprocess_study") as process, \
                self.assertRaisesRegex(FileNotFoundError, "T2.nii.gz"):
            preprocessing.preprocess_study(self.study, self.raw, self.processed)
        process.assert_not_called()
        self.assertFalse(self.processed.exists())

    def test_overlapping_roots_and_unsafe_study_ids_are_rejected(self):
        for source, output in ((self.raw, self.raw), (self.raw, self.raw / "nested")):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_study(self.study, source, output)
            with self.assertRaises(ValueError):
                embedding.embed_study(self.study, source, output, model_kind="backbone", checkpoint="unused")
        with self.assertRaises(ValueError):
            preprocessing.preprocess_study("../other", self.raw, self.processed)
        with self.assertRaises(ValueError):
            embedding.embed_study("../other", self.processed, self.root / "out",
                                  model_kind="backbone", checkpoint="unused")

    def test_scripts_start_from_an_unrelated_working_directory(self):
        directory = Path(__file__).resolve().parent
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        for name in ("convert_single_study.py", "preprocess_single_study.py", "embed_single_study.py"):
            result = subprocess.run([sys.executable, str(directory / name), "--help"],
                                    cwd=self.root, env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
