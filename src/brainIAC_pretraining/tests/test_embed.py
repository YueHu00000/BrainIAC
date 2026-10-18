"""Standalone export: all-patch pooling, study grouping, NPZ and recovery."""

import argparse
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brainIAC_pretraining import embed


class TokenEncoder(torch.nn.Module):
    def forward(self, image):
        assert not self.training and not torch.is_grad_enabled()
        tokens = torch.zeros(1, 216, 768)
        tokens[:, 0, :] = 216
        return tokens + 100, [tokens + number for number in range(1, 13)]


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = ["volume.1_7", "volume.2_3", "volume.3_7"]
        self.args = argparse.Namespace(
            final_csv=self.root / "final.csv", input_dir=self.root / "images",
            selected_csv=self.root / "selected.csv", checkpoint=self.root / "model.ckpt",
            output_dir=self.root / "embeddings", device="cpu",
        )
        self.rows = [
            {"unique_id": self.ids[0], "study_id": "Study_A", "series_description": "轴位 T1"},
            {"unique_id": self.ids[1], "study_id": "Study_B", "series_description": ""},
            {"unique_id": self.ids[2], "study_id": "Study_A", "series_description": "AX T2"},
            {"unique_id": "excluded.9", "study_id": "Excluded", "series_description": "T2"},
        ]
        write_csv(self.args.final_csv, ["unique_id"], [{"unique_id": x} for x in self.ids])
        self.write_selected()

    def write_selected(self):
        write_csv(self.args.selected_csv,
                  ["unique_id", "study_id", "series_description"], self.rows)

    def make_images(self):
        self.args.input_dir.mkdir()
        image = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
        image[:2] = 0
        for unique_id in self.ids:
            nib.save(nib.Nifti1Image(image, np.eye(4)),
                     self.args.input_dir / f"{unique_id}.nii.gz")

    def test_metadata_grouping_uses_explicit_study_and_only_final_ids(self):
        groups = embed.group_series(self.args.final_csv, self.args.selected_csv)
        self.assertEqual(list(groups), ["Study_A", "Study_B"])
        self.assertEqual([r["unique_id"] for r in groups["Study_A"]],
                         [self.ids[0], self.ids[2]])
        self.assertEqual(groups["Study_B"][0]["series_description"], "")

    def test_missing_duplicate_and_unsafe_metadata_fail_before_output(self):
        original = list(self.rows)
        for rows, message in [
            (original[1:], "Missing series metadata"),
            (original + [original[0]], "Duplicate metadata"),
            ([dict(original[0], study_id="../escape"), *original[1:]], "plain file ID"),
        ]:
            with self.subTest(message=message):
                self.rows = rows
                self.write_selected()
                with self.assertRaisesRegex(ValueError, message):
                    embed.run(self.args)
                self.assertFalse(self.args.output_dir.exists())

    def test_duplicate_final_ids_and_missing_description_column_fail(self):
        write_csv(self.args.final_csv, ["unique_id"],
                  [{"unique_id": self.ids[0]}] * 2)
        with self.assertRaisesRegex(ValueError, "unique volume IDs"):
            embed.group_series(self.args.final_csv, self.args.selected_csv)
        write_csv(self.args.final_csv, ["unique_id"], [{"unique_id": self.ids[0]}])
        write_csv(self.args.selected_csv, ["unique_id", "study_id"],
                  [{"unique_id": self.ids[0], "study_id": "Study_A"}])
        with self.assertRaisesRegex(ValueError, "series_description"):
            embed.group_series(self.args.final_csv, self.args.selected_csv)

    def test_real_nifti_fixed_transform_is_repeatable_and_nonzero_normalized(self):
        self.make_images()
        transform = embed.build_transform()
        path = str(self.args.input_dir / f"{self.ids[0]}.nii.gz")
        first = transform({"image": path})["image"]
        second = transform({"image": path})["image"]
        self.assertEqual(tuple(first.shape), (1, 96, 96, 96))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        nonzero = first[first != 0]
        self.assertAlmostEqual(float(nonzero.mean()), 0, places=5)
        self.assertAlmostEqual(float(nonzero.std(unbiased=False)), 1, places=5)

    def test_all_216_pooling_final_separation_and_pickle_free_record(self):
        transform = lambda _: {"image": torch.zeros(1, 96, 96, 96)}
        blocks, final = embed.extract_volume(TokenEncoder(), "image.nii.gz", transform, "cpu")
        np.testing.assert_array_equal(blocks[:, 0], [4, 7, 10, 13])
        np.testing.assert_array_equal(final, np.full(768, 101))
        record = embed.pack_record("轴位 T1 / é", blocks, final)
        path = self.root / "study.npz"
        np.savez_compressed(path, **{self.ids[0]: record})
        with np.load(path, allow_pickle=False) as data:
            saved = data[self.ids[0]]
            self.assertEqual(saved.shape, ())
            self.assertEqual(saved.dtype.names,
                             ("series_description", "block_embed", "final_embed"))
            self.assertFalse(saved.dtype.hasobject)
            self.assertEqual(saved["series_description"].item(), "轴位 T1 / é")
            self.assertEqual(saved["block_embed"].shape, (4, 768))
            self.assertEqual(saved["final_embed"].shape, (768,))
            self.assertEqual(saved["block_embed"].dtype, np.float32)
            self.assertEqual(saved["final_embed"].dtype, np.float32)

    def test_incomplete_study_not_published_then_resume_and_skip_completed(self):
        self.make_images()
        missing = self.args.input_dir / f"{self.ids[1]}.nii.gz"
        missing.unlink()
        with patch.object(embed, "load_encoder", return_value=TokenEncoder()):
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                embed.run(self.args)
        completed = self.args.output_dir / "Study_A.npz"
        stamp = completed.stat().st_mtime_ns
        self.assertFalse((self.args.output_dir / "Study_B.npz").exists())
        image = np.ones((8, 8, 8), dtype=np.float32)
        nib.save(nib.Nifti1Image(image, np.eye(4)), missing)
        (self.args.output_dir / "Study_B.npz.tmp").write_bytes(b"interrupted")
        with patch.object(embed, "load_encoder", return_value=TokenEncoder()), patch.object(
                embed, "extract_volume", wraps=embed.extract_volume) as extract:
            embed.run(self.args)
            self.assertEqual(extract.call_count, 1)
        self.assertEqual(completed.stat().st_mtime_ns, stamp)
        self.assertFalse((self.args.output_dir / "Study_B.npz.tmp").exists())
        with np.load(completed, allow_pickle=False) as data:
            self.assertEqual(data.files, [self.ids[0], self.ids[2]])
        with patch.object(embed, "load_encoder") as load, patch.object(embed, "build_transform") as build:
            embed.run(self.args)
            load.assert_not_called()
            build.assert_not_called()
        self.args.checkpoint = self.root / "another_model.ckpt"
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            embed.run(self.args)

    def test_failure_in_second_series_does_not_publish_first_series(self):
        self.make_images()
        (self.args.input_dir / f"{self.ids[2]}.nii.gz").unlink()
        with patch.object(embed, "load_encoder", return_value=TokenEncoder()):
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                embed.run(self.args)
        self.assertFalse(list(self.args.output_dir.glob("*.npz")))

    def test_real_vit_checkpoint_strict_load_and_repeatable_export(self):
        self.make_images()
        encoder = embed.ViT(
            in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            classification=False, save_attn=True,
        ).eval()
        state = {"backbone." + key: value for key, value in encoder.state_dict().items()}
        state["projection_head.0.weight"] = torch.zeros(2, 2)
        torch.save({"state_dict": state, "epoch": 198}, self.args.checkpoint)
        loaded = embed.load_encoder(self.args.checkpoint, "cpu")
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        transform = embed.build_transform()
        path = self.args.input_dir / f"{self.ids[0]}.nii.gz"
        first = embed.extract_volume(loaded, path, transform, "cpu")
        second = embed.extract_volume(loaded, path, transform, "cpu")
        with torch.inference_mode():
            image = transform({"image": str(path)})["image"].unsqueeze(0)
            tokens, states = encoder(image)
        expected = torch.stack([states[n - 1].mean(1) for n in (3, 6, 9, 12)], 1)[0]
        np.testing.assert_allclose(first[0], expected.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(first[1], tokens.mean(1)[0].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(np.allclose(first[0][-1], first[1]))
        del loaded, encoder, state
        torch.save({"state_dict": {"backbone.not_a_parameter": torch.zeros(1)}}, self.args.checkpoint)
        with self.assertRaises(RuntimeError):
            embed.load_encoder(self.args.checkpoint, "cpu")


if __name__ == "__main__":
    unittest.main()
