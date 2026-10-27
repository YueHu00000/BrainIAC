"""Focused fault and math checks for the additive replay-localization tool."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from monai.networks.nets import ViT

import diagnose_nan as diag
import replay_localize as localize


IDS = ["R01_Study_000001_1.2.3_7", "R01_Study_000002_1.2.4_8",
       "R01_Study_000003_1.2.5_9"]


class TinyViTModel(torch.nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams or {"lr": 5e-4}
        self.backbone = ViT(in_channels=1, img_size=(8, 8, 8), patch_size=(4, 4, 4),
                            hidden_size=12, mlp_dim=24, num_layers=2, num_heads=3,
                            classification=False, dropout_rate=0.2, save_attn=True)
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(12, 8),
                                                    torch.nn.Dropout(0.2))
        self.criterion = lambda a, b: ((a - b) ** 2).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}


def make_batch():
    return tuple({"image": torch.randn(3, 1, 8, 8, 8), "unique_id": list(IDS)}
                 for _ in range(2))


def make_state(model):
    config = model.configure_optimizers()
    return {"state_phase": "pre_forward", "hparams": dict(model.hparams),
            "model_state": copy.deepcopy(model.state_dict()),
            "buffers": diag.cpu_copy(dict(model.named_buffers())),
            "training": True, "optimizer_state": config["optimizer"].state_dict(),
            "scheduler_state": config["lr_scheduler"]["scheduler"].state_dict(),
            "scaler_state": None, "rng": diag.rng_state(), "precision": "32-true",
            "environment": diag.environment(), "model_device": "cpu"}


def bad_second_sample_factory(hparams=None):
    model = TinyViTModel(hparams)
    def inject(module, inputs, output):
        output = output.clone()
        output[1, 2, 4] = float("nan")
        return output
    model.backbone.blocks[1].register_forward_hook(inject)
    return model


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2026)
        np.random.seed(2026)

    def test_detailed_hooks_preserve_real_monai_forward_gradients_and_rng(self):
        plain = TinyViTModel().train()
        checked = copy.deepcopy(plain)
        batch = make_batch()
        before = diag.rng_state()
        expected_loss = diag.forward_views(plain, batch)
        expected_loss.backward()
        expected_rng = torch.get_rng_state().clone()
        expected_attention = plain.backbone.blocks[0].attn.att_mat.clone()
        diag.restore_rng(before)
        trace = localize.ForwardTraceV2(IDS, detailed_target="backbone.blocks.0")
        trace.install(checked)
        try:
            actual_loss = diag.forward_views(checked, batch)
            actual_loss.backward()
        finally:
            trace.close()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(checked.backbone.blocks[0].attn.att_mat,
                                   expected_attention, rtol=0, atol=0)
        for (name, expected), (actual_name, actual) in zip(plain.named_parameters(),
                                                        checked.named_parameters()):
            self.assertEqual(name, actual_name)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0,
                                       msg=lambda message: f"{name}: {message}")
        self.assertIsNone(trace.first_nonfinite)
        stage_names = [item["stage"] for item in trace.stages]
        self.assertIn("view0.backbone.blocks.0.attn.qk_raw", stage_names)
        self.assertIn("view1.backbone.blocks.0.attn.qk_raw", stage_names)
        self.assertFalse(checked.backbone._forward_pre_hooks)
        self.assertFalse(checked.backbone.blocks[0]._forward_hooks)

    def test_hook_failure_is_not_reported_as_finite_or_backward(self):
        state = make_state(TinyViTModel())
        batch = make_batch()
        result = localize.run_trial_v2(state, batch, "32-true", "cpu",
                                        model_factory=bad_second_sample_factory)
        self.assertFalse(result["finite"])
        self.assertEqual(result["first_nonfinite"], "view0.backbone.blocks.1.output")
        self.assertTrue(result["backward_requested"])
        self.assertFalse(result["backward_ran"])
        self.assertFalse(result["backward_completed"])
        self.assertFalse(result["optimizer_updated"])
        failed = next(item for item in result["trace_stages"] if not item["finite"])
        self.assertEqual(failed["sample_indices"], [1])
        self.assertEqual(failed["unique_ids"], [IDS[1]])
        self.assertEqual(batch[0]["unique_id"], IDS)

    def test_forward_only_and_backward_results_have_distinct_flags(self):
        state, batch = make_state(TinyViTModel()), make_batch()
        forward = localize.run_trial_v2(state, batch, "32-true", "cpu", forward_only=True,
                                         model_factory=TinyViTModel)
        backward = localize.run_trial_v2(state, batch, "32-true", "cpu",
                                          model_factory=TinyViTModel)
        self.assertTrue(forward["finite"])
        self.assertFalse(forward["backward_requested"])
        self.assertFalse(forward["backward_ran"])
        self.assertFalse(forward["backward_completed"])
        self.assertTrue(backward["finite"])
        self.assertTrue(backward["backward_requested"])
        self.assertTrue(backward["backward_ran"])
        self.assertTrue(backward["backward_completed"])
        self.assertTrue(backward["gradients"]["finite"])
        self.assertFalse(backward["optimizer_updated"])
        self.assertEqual(forward["stages"]["loss"], backward["stages"]["loss"])

    def test_coarse_to_detail_restarts_same_scene_at_the_failing_block(self):
        state, batch = make_state(TinyViTModel()), make_batch()
        result = localize.locate_trial_v2(state, batch, "32-true", "cpu", forward_only=True,
                                           model_factory=bad_second_sample_factory)
        self.assertEqual(result["detailed_target"], "backbone.blocks.1")
        self.assertFalse(result["coarse"]["finite"])
        self.assertFalse(result["detail"]["finite"])
        self.assertEqual(result["coarse"]["first_nonfinite"], result["detail"]["first_nonfinite"])
        stages = [item["stage"] for item in result["detail"]["trace_stages"]]
        self.assertIn("view0.backbone.blocks.1.attn.qk_raw", stages)
        self.assertNotIn("view0.backbone.blocks.0.attn.qk_raw", stages)
        coarse_input = result["coarse"]["trace_stages"][0]
        detail_input = result["detail"]["trace_stages"][0]
        self.assertEqual(coarse_input, detail_input)

    def test_qk_overflow_has_same_operand_float32_reference_and_bounded_sample_probe(self):
        model = TinyViTModel().half().train()
        # Supply finite Q/K/V large enough that only the FP16 dot product overflows.
        injection = model.backbone.blocks[0].attn.qkv.register_forward_hook(
            lambda module, inputs, output: torch.full_like(output, 256.))
        batch = make_batch()
        for view in batch:
            view["image"] = view["image"].half()
        trace = localize.ForwardTraceV2(IDS, detailed_target="backbone.blocks.0")
        trace.install(model)
        try:
            with self.assertRaises(diag.NonFiniteError):
                diag.forward_views(model, batch)
        finally:
            trace.close()
            injection.remove()
        self.assertEqual(trace.first_nonfinite, "view0.backbone.blocks.0.attn.qk_raw")
        failed = trace.stages[-1]
        self.assertEqual(failed["sample_indices"], [0, 1, 2])
        reference = failed["float32_reference"]
        self.assertTrue(reference["stats"]["finite"])
        self.assertEqual(reference["stats"]["max_abs"], 262144.)
        self.assertFalse(reference["cast_back_stats"]["finite"])
        self.assertEqual(reference["sample_indices"], [0])
        self.assertEqual(reference["unique_ids"], [IDS[0]])
        self.assertEqual(reference["omitted_bad_sample_count"], 2)

    def test_backward_nonfinite_is_not_attributed_to_a_forward_sample(self):
        def bad_gradient_factory(hparams=None):
            model = TinyViTModel(hparams)
            model.projection_head[0].weight.register_hook(
                lambda gradient: torch.full_like(gradient, float("nan")))
            return model
        state, batch = make_state(TinyViTModel()), make_batch()
        result = localize.run_trial_v2(state, batch, "32-true", "cpu",
                                        model_factory=bad_gradient_factory)
        self.assertFalse(result["finite"])
        self.assertEqual(result["first_nonfinite"], "unscaled_preclip_gradients")
        self.assertTrue(result["backward_ran"])
        self.assertTrue(result["backward_completed"])
        self.assertFalse(result["gradients"]["finite"])
        self.assertTrue(all(stage["finite"] for stage in result["trace_stages"]))
        self.assertFalse(result["optimizer_updated"])

    def test_mismatched_view_order_is_rejected_before_uid_attribution(self):
        batch = make_batch()
        batch[1]["unique_id"] = list(reversed(IDS))
        with self.assertRaisesRegex(ValueError, "matching"):
            localize.batch_ids_v2(batch)

    def test_cpu_mixed_trials_are_reported_as_not_run_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"
            scene.mkdir()
            diag.write_json(scene / "summary.json", {"kind": "forward_nonfinite"})
            torch.save(make_state(TinyViTModel()), scene / "state.pt")
            torch.save(make_batch(), scene / "batch.pt")
            args = ["--scene", str(scene), "--device", "cpu", "--precisions",
                    "16-mixed", "bf16-mixed", "--output-dir", str(root / "report")]
            self.assertEqual(localize.main_v2(args), 0)
            report_path = root / "report" / "localization_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual([trial["status"] for trial in report["trials"]],
                             ["not_run_requires_cuda", "not_run_requires_cuda"])
            self.assertFalse(report["optimizer_update_requested"])
            for trial in report["trials"]:
                self.assertNotIn("finite", trial)
            original = report_path.read_text(encoding="utf-8")
            with self.assertRaises(FileExistsError):
                localize.main_v2(args)
            self.assertEqual(report_path.read_text(encoding="utf-8"), original)

    def test_cli_saved_scene_serializes_backward_and_forward_fault_results(self):
        original_locate = localize.locate_trial_v2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"
            scene.mkdir()
            diag.write_json(scene / "summary.json",
                            {"kind": "forward_nonfinite", "unique_ids": IDS})
            torch.save(make_state(TinyViTModel()), scene / "state.pt")
            torch.save(make_batch(), scene / "batch.pt")
            for name, factory in (("healthy", TinyViTModel),
                                  ("fault", bad_second_sample_factory)):
                def locate(*args, **kwargs):
                    return original_locate(*args, **kwargs, model_factory=factory)
                with patch.object(localize, "locate_trial_v2", side_effect=locate):
                    result = localize.main_v2(["--scene", str(scene), "--device", "cpu",
                                              "--precisions", "32-true", "--output-dir",
                                              str(root / name)])
                self.assertEqual(result, 0)
                report = json.loads((root / name / "localization_report.json").read_text(
                    encoding="utf-8"))
                self.assertEqual(report["source_summary"]["unique_ids"], IDS)
                trial = report["trials"][0]
                self.assertEqual(trial["status"], "completed")
                self.assertFalse(trial["coarse"]["optimizer_updated"])
                if name == "healthy":
                    self.assertTrue(trial["coarse"]["finite"])
                    self.assertTrue(trial["coarse"]["backward_completed"])
                    self.assertIsNone(trial["detail"])
                else:
                    self.assertFalse(trial["coarse"]["finite"])
                    self.assertFalse(trial["coarse"]["backward_ran"])
                    failed = next(stage for stage in trial["detail"]["trace_stages"]
                                  if not stage["finite"])
                    self.assertEqual(failed["sample_indices"], [1])
                    self.assertEqual(failed["unique_ids"], [IDS[1]])

    def test_later_trial_error_preserves_completed_report_and_is_rethrown(self):
        original_locate = localize.locate_trial_v2
        calls = 0
        def locate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("CUDA out of memory")
            return original_locate(*args, **kwargs, model_factory=TinyViTModel)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"
            scene.mkdir()
            diag.write_json(scene / "summary.json", {"kind": "forward_nonfinite"})
            torch.save(make_state(TinyViTModel()), scene / "state.pt")
            torch.save(make_batch(), scene / "batch.pt")
            with patch.object(localize, "locate_trial_v2", side_effect=locate):
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                    localize.main_v2(["--scene", str(scene), "--device", "cpu",
                                      "--precisions", "32-true", "32-true", "--forward-only",
                                      "--output-dir", str(root / "report")])
            report = json.loads((root / "report" / "localization_report.json").read_text(
                encoding="utf-8"))
            first, failed = report["trials"]
            self.assertEqual(first["status"], "completed")
            self.assertTrue(first["coarse"]["finite"])
            self.assertFalse(first["coarse"]["backward_requested"])
            self.assertEqual(failed["status"], "error")
            self.assertEqual(failed["error_type"], "RuntimeError")
            self.assertEqual(failed["error"], "CUDA out of memory")
            self.assertNotIn("finite", failed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real mixed-precision smoke")
    def test_real_cuda_fp16_and_supported_bf16_forward_backward(self):
        state, batch = make_state(TinyViTModel()), make_batch()
        precisions = ["16-mixed"]
        if torch.cuda.is_bf16_supported():
            precisions.append("bf16-mixed")
        for precision in precisions:
            with self.subTest(precision=precision):
                result = localize.run_trial_v2(state, batch, precision, "cuda:0",
                                                model_factory=TinyViTModel)
                self.assertTrue(result["finite"])
                self.assertTrue(result["backward_completed"])
                self.assertTrue(result["gradients"]["finite"])
                self.assertFalse(result["optimizer_updated"])


if __name__ == "__main__":
    unittest.main()
