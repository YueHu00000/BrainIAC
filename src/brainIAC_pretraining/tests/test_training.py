"""CSV contracts, all-patch pooling, finite loss updates and full-state resume."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brainIAC_pretraining.train import parse_args

HAS_TRAINING = all(importlib.util.find_spec(name) is not None
                   for name in ("torch", "lightly", "pytorch_lightning", "nibabel"))


class TrainingCliTests(unittest.TestCase):
    def test_required_inputs_and_default_scratch(self):
        args = parse_args(["--final-csv", "final.csv", "--input-dir", "images",
                           "--output-dir", "out", "--accelerator", "cpu"])
        self.assertIsNone(args.resume)
        self.assertIsNone(args.init_checkpoint)
        self.assertEqual(args.config.parent.name, "simclr")
        self.assertEqual(args.config, Path(__file__).resolve().parents[2] / "simclr/config.yml")
        self.assertTrue(args.config.is_file())

    def test_resume_and_initialization_are_exclusive(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--final-csv", "final.csv", "--input-dir", "images",
                        "--output-dir", "out", "--resume", "a.ckpt",
                        "--init-checkpoint", "b.ckpt"])


@unittest.skipUnless(HAS_TRAINING, "Requires pretraining dependencies")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def test_csv_only_and_independent_transform_calls(self):
        from simclr.dataset import NiftiDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "final.csv"
            ids = ["Study_1_1.2.826.0.1.131_7", "Study_1_1.2.826.0.1.134_7"]
            csv.write_text("unique_id\n" + "\n".join(ids) + "\n", encoding="utf-8")
            (root / "unlisted.nii.gz").write_bytes(b"not a nifti")
            calls = []

            def transform(image):
                calls.append(image["image"])
                return {"image": len(calls)}

            dataset = NiftiDataset(csv, root, transform)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0], ({"image": 1}, {"image": 2}))
            self.assertEqual(dataset[1], ({"image": 3}, {"image": 4}))
            self.assertEqual(calls, [str(root / f"{unique_id}.nii.gz")
                                     for unique_id in ids for _ in range(2)])

    def test_missing_input_raises_at_loading(self):
        from simclr.dataset import NiftiDataset
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "final.csv"
            csv.write_text("unique_id\nmissing_1.2.826.0.1.132_10\n", encoding="utf-8")
            dataset = NiftiDataset(csv, tmp, lambda image: Path(image["image"]).read_bytes())
            self.assertEqual(len(dataset), 1)
            with self.assertRaises(FileNotFoundError):
                dataset[0]

    def test_real_nifti_author_augmentation_produces_two_views(self):
        import nibabel as nib
        import numpy as np
        from simclr.dataset import NiftiDataset
        from simclr.train_multigpu import build_transform
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "final.csv"
            csv.write_text("unique_id\nsample_1.2.826.0.1.133_10\n", encoding="utf-8")
            image = np.random.default_rng(10).normal(size=(16, 16, 16)).astype("float32")
            nib.save(nib.Nifti1Image(image, np.eye(4)), root / "sample_1.2.826.0.1.133_10.nii.gz")
            transform = build_transform()
            transform.set_random_state(seed=0)
            a, b = NiftiDataset(csv, root, transform)[0]
            self.assertEqual(tuple(a["image"].shape), (1, 96, 96, 96))
            self.assertEqual(tuple(b["image"].shape), (1, 96, 96, 96))
            self.assertTrue(self.torch.isfinite(a["image"]).all())
            self.assertFalse(self.torch.equal(a["image"], b["image"]))

    def tiny_model(self):
        torch = self.torch
        from simclr.model import SimCLRModel

        class TinyBackbone(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 8)

            def forward(self, x):
                return self.encoder(x), []

        with patch("simclr.model.ViT", TinyBackbone), patch(
                "simclr.model.SimCLRProjectionHead",
                side_effect=lambda *_: torch.nn.Linear(8, 8)):
            return SimCLRModel({"lr": 0.0005, "max_epochs": 2})

    def test_pooling_keeps_first_patch(self):
        torch = self.torch
        from simclr.model import SimCLRModel

        class Tokens(torch.nn.Module):
            def forward(self, x):
                return x, []

        with patch("simclr.model.ViT", return_value=Tokens()) as constructor, patch(
                "simclr.model.SimCLRProjectionHead", return_value=torch.nn.Identity()):
            model = SimCLRModel({"lr": 0.0005})
        self.assertFalse(constructor.call_args.kwargs["classification"])
        tokens = torch.tensor([[[1., 0.], [0., 3.]]])
        expected = torch.nn.functional.normalize(torch.tensor([[0.5, 1.5]]), dim=1)
        torch.testing.assert_close(model(tokens), expected)

    def test_real_vit_forward_has_216_patches(self):
        from simclr.model import SimCLRModel
        torch = self.torch
        model = SimCLRModel({"lr": 0.0005}).eval()
        image = torch.randn(2, 1, 96, 96, 96)
        with torch.no_grad():
            tokens, _ = model.backbone(image)
            self.assertEqual(tuple(tokens.shape), (2, 216, 768))
            output = model(image)
        self.assertEqual(tuple(output.shape), (2, 2048))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.norm(dim=1), torch.ones(2))

    def test_loss_update_and_full_checkpoint_resume(self):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        torch = self.torch
        torch.manual_seed(0)
        loader = torch.utils.data.DataLoader(
            [({"image": torch.randn(3, 4)}, {"image": torch.randn(3, 4)})
             for _ in range(4)], batch_size=2)
        model = self.tiny_model()
        initial = model.backbone.encoder.weight.detach().clone()
        loss = model.criterion(model(torch.randn(2, 3, 4)), model(torch.randn(2, 3, 4)))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.configure_optimizers()["optimizer"].param_groups[0]["weight_decay"], 5e-4)
        with tempfile.TemporaryDirectory() as tmp:
            callback = ModelCheckpoint(dirpath=tmp, save_last=True, save_weights_only=False,
                                       monitor="train_loss_ssl", mode="min")
            trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                                 logger=False, callbacks=[callback],
                                 enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model, loader)
            self.assertFalse(torch.equal(initial, model.backbone.encoder.weight))
            checkpoint = torch.load(callback.last_model_path, map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["optimizer_states"])
            self.assertTrue(checkpoint["lr_schedulers"])
            step = checkpoint["global_step"]
            resumed = self.tiny_model()
            continuation = pl.Trainer(max_epochs=2, accelerator="cpu", devices=1,
                                      logger=False, enable_checkpointing=False,
                                      enable_progress_bar=False, enable_model_summary=False)
            continuation.fit(resumed, loader, ckpt_path=callback.last_model_path)
            self.assertEqual(continuation.global_step, step + len(loader))
            self.assertEqual(int(continuation.optimizers[0].state[
                resumed.backbone.encoder.weight]["step"]), continuation.global_step)


if __name__ == "__main__":
    unittest.main()
