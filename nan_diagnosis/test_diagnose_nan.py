"""Synthetic fault/replay tests. No patient data or existing programs are modified."""

import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import pytorch_lightning as pl

import diagnose_nan as diag

ORIGINAL_MODEL_FACTORY = diag.make_model


class TinyBackbone(torch.nn.Module):
    def __init__(self, **unused):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, image):
        if image.ndim == 5:
            flat = image.flatten(2)
            image = torch.stack([flat.mean(-1), flat.std(-1), flat.amin(-1), flat.amax(-1)], dim=-1)
        return self.dropout(self.linear(image)), []


def tiny_model(hparams=None, probe=None):
    from lightly.models.modules.heads import SimCLRProjectionHead
    constructor = SimCLRProjectionHead
    with patch("monai.networks.nets.ViT", TinyBackbone), patch(
            "lightly.models.modules.heads.SimCLRProjectionHead",
            side_effect=lambda *_: constructor(8, 8, 8)):
        return ORIGINAL_MODEL_FACTORY(hparams or {"lr": 5e-4, "max_epochs": 2}, probe)


def make_batch():
    return tuple({"image": torch.randn(2, 3, 4),
                  "unique_id": ["Study_1_1.2.3_7", "Study_2_1.2.4_7"]} for _ in range(2))


def begin(probe, model, batch, scaler=None):
    config = model.configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]
    probe.begin(model, batch, optimizer, scheduler, scaler,
                {"epoch": 150, "batch_idx": 0, "global_step": 100,
                 "lr": optimizer.param_groups[0]["lr"]}, "32-true" if scaler is None else "16-mixed")
    return optimizer


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(31)
        np.random.seed(31)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_cli_safety_and_precision_selection(self):
        args = diag.parse_args(["run", "--final-csv", "a.csv", "--input-dir", "images",
                                "--output-dir", "out"])
        self.assertIsNone(args.precision)
        self.assertFalse(args.capture_update_state)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            diag.parse_args(["inspect", "--checkpoint", "a", "--sample", "one.nii.gz",
                             "--output-dir", "out"])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            diag.parse_args(["replay", "--scene", "a", "--output-dir", "out",
                             "--forward-only", "--update"])
        diag.fresh_output(self.root / "new")
        with self.assertRaises(FileExistsError):
            diag.fresh_output(self.root / "new")

    def test_statistics_do_not_overflow_half_precision(self):
        stats = diag.tensor_stats(torch.full((1000,), 1000., dtype=torch.float16))
        self.assertTrue(stats["finite"])
        self.assertEqual(stats["mean"], 1000.)
        self.assertTrue(np.isfinite(stats["norm"]))
        self.assertEqual(stats["std"], 0.)

    def test_checkpoint_checks_optimizer_and_batchnorm(self):
        path = self.root / "test.ckpt"
        torch.save({"state_dict": {"projection.running_var": torch.tensor([-1.]),
                                   "backbone.weight": torch.tensor([1.])},
                    "optimizer_states": [{"state": {0: {"exp_avg": torch.tensor([float('inf')])}}}],
                    "epoch": 149, "global_step": 24600}, path)
        _, report = diag.inspect_checkpoint(path)
        self.assertFalse(report["static_finite"])
        self.assertEqual(report["negative_running_variance"], ["projection.running_var"])
        self.assertIn("exp_avg", report["optimizer_nonfinite"][0]["name"])
        self.assertEqual(report["epoch"], 149)

    def test_checkpoint_checks_nonfinite_scaler_and_learning_rate(self):
        path = self.root / "scaler.ckpt"
        torch.save({"state_dict": {"weight": torch.ones(1)},
                    "optimizer_states": [{"param_groups": [{"lr": float("nan")}]}],
                    "native_amp_scaling_state": {"scale": float("inf")}}, path)
        _, report = diag.inspect_checkpoint(path)
        self.assertFalse(report["static_finite"])
        self.assertTrue(report["optimizer_nonfinite"])
        self.assertTrue(report["amp_scaler_nonfinite"])

    def test_backbone_constructor_matches_original_source_ast(self):
        # Read source only, never import/execute or write its __pycache__.
        original = Path(__file__).resolve().parents[1] / "src/simclr/model.py"
        original_tree = ast.parse(original.read_text(encoding="utf-8"))
        new_tree = ast.parse(Path(diag.__file__).read_text(encoding="utf-8"))
        def find_call(tree, name):
            return next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == name)
        for name in ("ViT", "SimCLRProjectionHead", "NTXentLoss"):
            self.assertEqual(ast.dump(find_call(original_tree, name)),
                             ast.dump(find_call(new_tree, name)))

    def test_checked_forward_matches_reference_math_gradients_and_update(self):
        model = tiny_model()
        checked = tiny_model()
        checked.load_state_dict(model.state_dict())
        batch = make_batch()
        control_optim = model.configure_optimizers()["optimizer"]
        probe = diag.Probe(self.root / "diag")
        checked_optim = begin(probe, checked, batch)
        state = diag.rng_state()
        # Independent reference expression used by original training_step.
        control_loss = model.criterion(model(batch[0]["image"]), model(batch[1]["image"]))
        control_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        control_optim.step()
        diag.restore_rng(state)
        checked_loss = diag.forward_views(checked, batch, probe)
        checked_loss.backward()
        probe.check_gradients()
        torch.nn.utils.clip_grad_norm_(checked.parameters(), 2.)
        checked_optim.step()
        probe.updates += 1
        probe.finish()
        torch.testing.assert_close(control_loss, checked_loss, rtol=0, atol=0)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, checked.state_dict()[name], rtol=0, atol=0)
        rows = list(csv.DictReader((self.root / "diag/steps.csv").read_text().splitlines()))
        self.assertEqual(rows[0]["optimizer_updated"], "True")

    def test_forward_capture_restores_pre_forward_buffers_and_replays(self):
        def bad_model(hparams=None):
            model = tiny_model(hparams)
            model.projection_head.register_forward_hook(lambda module, args, output: output * float("nan"))
            return model
        model = bad_model()
        buffers = diag.cpu_copy(dict(model.named_buffers()))
        batch = make_batch()
        probe = diag.Probe(self.root / "diag")
        begin(probe, model, batch)
        with self.assertRaises(diag.NonFiniteError):
            diag.forward_views(model, batch, probe)
        folder = self.root / "diag/first_forward_nonfinite"
        summary = json.loads((folder / "summary.json").read_text())
        state = torch.load(folder / "state.pt", weights_only=False)
        self.assertEqual(summary["stage"], "view0.projection")
        self.assertEqual(summary["unique_ids"], batch[0]["unique_id"])
        for name, value in buffers.items():
            torch.testing.assert_close(value, state["buffers"][name])
        first = diag.replay_trial(state, batch, "32-true", "cpu", model_factory=bad_model)
        second = diag.replay_trial(state, batch, "32-true", "cpu", model_factory=bad_model)
        self.assertEqual(first, second)
        self.assertEqual(first["first_nonfinite"], "view0.projection")

    def test_forward_only_replay_does_not_construct_optimizer(self):
        torch.set_float32_matmul_precision("medium")
        model = tiny_model()
        probe = diag.Probe(self.root / "diag")
        batch = make_batch()
        begin(probe, model, batch)
        state = probe.snapshot()
        def factory(hparams):
            result = tiny_model(hparams)
            result.configure_optimizers = lambda: self.fail("Forward-only must not allocate optimizer states")
            return result
        result = diag.replay_trial(state, batch, "32-true", "cpu", model_factory=factory, forward_only=True)
        self.assertTrue(result["finite"])
        self.assertFalse(result["backward_ran"])
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")

    def test_buffer_fault_is_detected_even_when_output_is_finite(self):
        model = tiny_model()
        def corrupt_buffer(module, args, output):
            next(buffer for name, buffer in module.named_buffers() if name.endswith("running_mean")).fill_(float("nan"))
        model.projection_head.register_forward_hook(corrupt_buffer)
        probe = diag.Probe(self.root / "diag")
        begin(probe, model, make_batch())
        with self.assertRaises(diag.NonFiniteError):
            diag.forward_views(model, probe.batch, probe)
        summary = json.loads((self.root / "diag/first_buffer_nonfinite/summary.json").read_text())
        self.assertEqual(summary["stage"], "view0.buffers")
        self.assertTrue(summary["stages"]["view0.projection"]["finite"])

    def test_backward_inf_with_finite_loss_captures_and_stops_without_scaler(self):
        model = tiny_model()
        next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        probe = diag.Probe(self.root / "diag")
        begin(probe, model, make_batch())
        loss = diag.forward_views(model, probe.batch, probe)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        with self.assertRaises(diag.NonFiniteError):
            probe.check_gradients()
        self.assertTrue((self.root / "diag/first_gradient_nonfinite/state.pt").is_file())

    def test_amp_gradient_event_is_not_treated_as_forward_failure(self):
        class FakeScaler:
            def get_scale(self):
                return 65536.
            def state_dict(self):
                return {"scale": 65536.}
        model = tiny_model()
        probe = diag.Probe(self.root / "diag")
        begin(probe, model, make_batch(), FakeScaler())
        loss = diag.forward_views(model, probe.batch, probe)
        loss.backward()
        next(model.parameters()).grad.fill_(float("inf"))
        probe.check_gradients()  # No stop; actual GradScaler owns update skipping.
        self.assertFalse(probe.grad["finite"])
        probe.finish()  # Simulated skip: no optimizer.step called.
        row = next(csv.DictReader((self.root / "diag/steps.csv").read_text().splitlines()))
        self.assertEqual(row["optimizer_updated"], "False")
        self.assertEqual(row["state_finite"], "True")

    def test_update_fault_requires_pre_update_snapshot_for_replay(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                folder = self.root / str(capture)
                model = tiny_model()
                config = model.configure_optimizers()
                optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]
                # Finite but invalid second moment: update produces NaN at sqrt.
                for parameter in model.parameters():
                    optimizer.state[parameter] = {"step": torch.tensor(0.),
                        "exp_avg": torch.zeros_like(parameter),
                        "exp_avg_sq": -torch.ones_like(parameter)}
                batch = make_batch()
                probe = diag.Probe(folder, capture_update_state=capture)
                probe.begin(model, batch, optimizer, scheduler, None,
                            {"epoch": 0, "batch_idx": 0, "global_step": 0, "lr": 5e-4}, "32-true")
                diag.forward_views(model, batch, probe).backward()
                probe.check_gradients()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                probe.updates += 1
                with self.assertRaises(diag.NonFiniteError):
                    probe.finish()
                scene = folder / "first_update_nonfinite"
                state = torch.load(scene / "state.pt", weights_only=False)
                if capture:
                    result = diag.replay_trial(state, batch, "32-true", "cpu", True,
                                                model_factory=tiny_model)
                    self.assertEqual(result["first_nonfinite"], "post_update_state")
                    self.assertTrue((scene / "post_update_state.pt").is_file())
                else:
                    with self.assertRaisesRegex(ValueError, "Post-update-only"):
                        diag.replay_trial(state, batch, "32-true", "cpu", model_factory=tiny_model)

    def test_data_nan_records_stage_and_can_be_inspected(self):
        import nibabel as nib
        path = self.root / "nan.nii.gz"
        nib.save(nib.Nifti1Image(np.full((8, 8, 8), np.nan, dtype=np.float32), np.eye(4)), path)
        transform = diag.build_transform(self.root / "data")
        with self.assertRaises(RuntimeError):
            transform({"image": str(path), "unique_id": "Study_1_1.2.3_7", "path": str(path),
                       "view": 0, "trace": []})
        scene = next((self.root / "data").iterdir())
        report = json.loads((scene / "summary.json").read_text())
        self.assertEqual(report["stage"], "00.LoadImaged")
        args = diag.parse_args(["replay", "--scene", str(scene), "--output-dir", str(self.root / "replay")])
        diag.replay_command(args)
        self.assertFalse(json.loads((self.root / "replay/replay_report.json").read_text())["stats"]["finite"])

    def test_run_data_nan_wrapped_by_monai_returns_two(self):
        import nibabel as nib
        manifest = self.root / "final.csv"
        manifest.write_text("unique_id\nbad\nbad\n", encoding="utf-8")
        nib.save(nib.Nifti1Image(np.full((8, 8, 8), np.nan, dtype=np.float32), np.eye(4)),
                 self.root / "bad.nii.gz")
        with patch.object(diag, "make_model", tiny_model):
            result = diag.main(["run", "--final-csv", str(manifest), "--input-dir", str(self.root),
                                "--output-dir", str(self.root / "run"), "--accelerator", "cpu",
                                "--num-workers", "0", "--batch-size", "2", "--max-epochs", "1"])
        self.assertEqual(result, 2)
        scene = next((self.root / "run/diagnostics").glob("data_*"))
        self.assertTrue((scene / "summary.json").exists())

    def test_checks_preserve_augmentation_rng_and_output(self):
        import nibabel as nib
        from monai.transforms import Randomizable
        path = self.root / "finite.nii.gz"
        nib.save(nib.Nifti1Image(np.random.default_rng(1).normal(size=(12, 12, 12)).astype(np.float32),
                               np.eye(4)), path)
        plain = diag.build_transform()
        checked = diag.build_transform(self.root / "data")
        plain.set_random_state(seed=42)
        checked.set_random_state(seed=42)
        state = diag.rng_state()
        shared = copy.deepcopy(Randomizable.R.get_state())
        a = plain({"image": str(path)})["image"]
        diag.restore_rng(state)
        Randomizable.R.set_state(shared)
        b = checked({"image": str(path), "unique_id": "id", "path": str(path),
                     "view": 0, "trace": []})["image"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_augmentation_matches_read_only_original_reference(self):
        import nibabel as nib
        from monai.transforms import Randomizable
        source = Path(__file__).resolve().parents[1] / "src/simclr"
        # Execute source text in an isolated namespace, no project imports or writes.
        namespace = {"__name__": "_read_only_reference"}
        exec(compile((source / "random_resized_crop.py").read_text(encoding="utf-8"),
                     "<reference_crop>", "exec"), namespace)
        tree = ast.parse((source / "train_multigpu.py").read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or not node.level]
        exec(compile(tree, "<reference_augmentation>", "exec"), namespace)
        original = namespace["build_transform"]()
        checked = diag.build_transform(self.root / "data")
        original.set_random_state(seed=42)
        checked.set_random_state(seed=42)
        path = self.root / "input.nii.gz"
        nib.save(nib.Nifti1Image(np.random.default_rng(5).normal(size=(12, 12, 12)).astype(np.float32),
                               np.eye(4)), path)
        state, shared = diag.rng_state(), copy.deepcopy(Randomizable.R.get_state())
        expected = original({"image": str(path)})["image"]
        diag.restore_rng(state)
        Randomizable.R.set_state(shared)
        actual = checked({"image": str(path), "unique_id": "id", "path": str(path),
                          "view": 0, "trace": []})["image"]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_run_command_complete_data_and_resume_path(self):
        import nibabel as nib
        for index in range(2):
            nib.save(nib.Nifti1Image(np.random.default_rng(index).normal(size=(8, 8, 8)).astype(np.float32),
                                   np.eye(4)), self.root / f"Study_{index}_1.2.3_{index}.nii.gz")
        manifest = self.root / "final.csv"
        manifest.write_text("unique_id\nStudy_0_1.2.3_0\nStudy_1_1.2.3_1\n", encoding="utf-8")
        base = ["run", "--final-csv", str(manifest), "--input-dir", str(self.root),
                "--accelerator", "cpu", "--num-workers", "1", "--batch-size", "2"]
        with patch.object(diag, "make_model", tiny_model):
            result = diag.main(base + ["--output-dir", str(self.root / "run1"), "--max-epochs", "1"])
            self.assertEqual(result, 0)
            checkpoint = self.root / "run1/checkpoints/last.ckpt"
            self.assertTrue(checkpoint.is_file())
            result = diag.main(base + ["--output-dir", str(self.root / "run2"),
                                      "--resume", str(checkpoint), "--max-epochs", "2"])
            self.assertEqual(result, 0)
        row = next(csv.DictReader((self.root / "run2/diagnostics/steps.csv").read_text().splitlines()))
        self.assertEqual(row["epoch"], "1")
        self.assertEqual(row["global_step"], "1")
        event = json.loads((self.root / "run2/diagnostics/events.jsonl").read_text().splitlines()[0])
        self.assertIn("data.view0.00.LoadImaged", event["stages"])
        self.assertCountEqual(event["unique_ids"], ["Study_0_1.2.3_0", "Study_1_1.2.3_1"])

    def test_real_lightning_callback_update_count_and_resume(self):
        probe = diag.Probe(self.root / "first/diagnostics")
        model = tiny_model(probe=probe)
        batch = make_batch()
        loader = [batch, batch]
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint(dirpath=self.root / "first/checkpoints", save_last=True, save_top_k=0)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, precision="32-true",
                             gradient_clip_val=2., logger=False, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=[diag.make_callback(probe), checkpoint])
        # Dataset is already batched; avoids introducing an extra dimension.
        dataloader = torch.utils.data.DataLoader(loader, batch_size=None)
        trainer.fit(model, dataloader)
        self.assertEqual(probe.updates, 2)
        resumed_probe = diag.Probe(self.root / "second/diagnostics")
        resumed = tiny_model(probe=resumed_probe)
        continuation = pl.Trainer(max_epochs=2, accelerator="cpu", devices=1, precision="32-true",
                                  gradient_clip_val=2., logger=False, enable_checkpointing=False,
                                  enable_progress_bar=False, enable_model_summary=False,
                                  callbacks=[diag.make_callback(resumed_probe)])
        continuation.fit(resumed, dataloader, ckpt_path=checkpoint.last_model_path)
        self.assertEqual(continuation.global_step, 4)
        self.assertEqual(resumed_probe.updates, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires actual CUDA AMP")
    def test_cuda_amp_skips_injected_infinite_gradient(self):
        probe = diag.Probe(self.root / "cuda")
        model = tiny_model(probe=probe)
        calls = [0]
        def overflow_once(gradient):
            calls[0] += 1
            return torch.full_like(gradient, float("inf")) if calls[0] == 1 else gradient
        next(model.parameters()).register_hook(overflow_once)
        loader = torch.utils.data.DataLoader([make_batch(), make_batch()], batch_size=None)
        trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, precision="16-mixed",
                             gradient_clip_val=2., logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             callbacks=[diag.make_callback(probe)])
        trainer.fit(model, loader)
        rows = list(csv.DictReader((self.root / "cuda/steps.csv").read_text().splitlines()))
        self.assertEqual(rows[0]["optimizer_updated"], "False")
        self.assertLess(float(rows[0]["scale_after"]), float(rows[0]["scale_before"]))
        self.assertTrue((self.root / "cuda/first_gradient_nonfinite/state.pt").is_file())


if __name__ == "__main__":
    unittest.main()
