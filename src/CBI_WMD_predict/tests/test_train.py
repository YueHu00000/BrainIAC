"""Small CPU checks for Harvey selection and the linear objective."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CBI_WMD_predict.train import fit_trial, metric_values, penalty
from CBI_WMD_predict import train
from CBI_WMD_predict.common import FOLDS, write_csv


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_regularization_includes_classifier_bias(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight[:] = torch.tensor([[2.0, -3.0]])
            model.bias[:] = 4.0
        self.assertAlmostEqual(float(penalty(model, "L1", 0.3).detach()), 0.3 / 2 * 9, places=6)
        self.assertAlmostEqual(float(penalty(model, "L2", 0.3).detach()), 0.3 / 2 * 29, places=6)

    def test_metrics_are_stable_for_large_logits(self):
        values = metric_values([0, 1, 0, 1], [-1000, 1000, -2, 2])
        self.assertEqual(values["auroc"], 1.0)
        self.assertEqual(values["auprc"], 1.0)
        self.assertEqual(values["balanced_accuracy"], 1.0)
        self.assertTrue(np.isfinite(values["nll"]))

    def test_equal_train_and_validation_auroc_has_no_fallback(self):
        features = np.zeros((8, 2), dtype=np.float32)
        labels = np.array([0, 1] * 4, dtype=np.float32)
        history = []
        best, summary = fit_trial(
            features, labels, features, labels,
            lr=0.1, kind="L1", alpha=0, epochs=3,
            batch_size=8, seed=0, device=torch.device("cpu"),
            history_callback=history.append,
        )
        self.assertIsNone(best)
        self.assertEqual(summary["status"], "no_eligible_checkpoint")
        self.assertEqual(len(history), 3)
        self.assertTrue(all(row["eligible"] == 0 for row in history))

    def test_earliest_best_state_is_independent_of_later_updates(self):
        labels = np.array([0, 1] * 4, dtype=np.float32)
        features = (2 * labels - 1).reshape(-1, 1)
        options = dict(lr=0.2, kind="L2", alpha=0,
                       batch_size=8, seed=0, device=torch.device("cpu"))
        early, _ = fit_trial(features, labels, features, 1 - labels, epochs=3, **options)
        later, _ = fit_trial(features, labels, features, 1 - labels, epochs=10, **options)
        self.assertIsNotNone(early)
        self.assertIsNotNone(later)
        self.assertEqual(early["epoch"], later["epoch"])
        for name, value in early["state_dict"].items():
            torch.testing.assert_close(value, later["state_dict"][name], rtol=0, atol=0)


class TrainerBoundaryTests(unittest.TestCase):
    """Validate one manifest and verify that selection never accesses test features."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "manifests"
        self.inputs.mkdir()
        self.manifest = self.inputs / "only.csv"
        self.rows = []
        for site in (1, 12, 13):
            for label in (0, 1):
                study = f"S_{site}_{label}"
                self.rows.append(dict(patient_id=f"P_{site}_{label}", study_id=study,
                                      site_id=str(site), CBI=str(label), WMD=str(label),
                                      embedding_path=str(self.root / f"{study}.npz"),
                                      t1_unique_id=f"{study}_1.2.1_1",
                                      t2_unique_id=f"{study}_1.2.2_2"))
        self.write_inputs()

    def write_inputs(self):
        fields = [*self.rows[0], "modality"]
        write_csv(self.manifest, [dict(row, modality="T1") for row in self.rows], fields)

    def args(self):
        return train.parse_args([
            "--manifest", str(self.manifest), "--output-dir", str(self.root / "results"),
            "--folds", "1", "--targets", "CBI",
            "--epochs", "1", "--batch-size", "2", "--device", "cpu",
            "--learning-rates", "0.1", "0.01", "--penalties", "L1", "--alphas", "0",
        ])

    def test_one_manifest_derives_folds_without_sibling_files(self):
        for row in self.rows:
            del row["t2_unique_id"]
        self.write_inputs()
        indexed, assignments, modality = train.read_inputs(self.manifest)
        self.assertEqual(modality, "T1")
        self.assertEqual(len(indexed), 6)
        self.assertEqual(list(self.inputs.iterdir()), [self.manifest])
        for row in self.rows:
            for fold, groups in FOLDS.items():
                expected = next(role for role, sites in groups.items() if int(row["site_id"]) in sites)
                self.assertEqual(assignments[row["study_id"], fold], expected)

    def test_mixed_modalities_are_rejected(self):
        records = [dict(row, modality="T1" if index else "T2") for index, row in enumerate(self.rows)]
        write_csv(self.manifest, records, list(records[0]))
        with self.assertRaises(ValueError):
            train.read_inputs(self.manifest)

    def test_duplicate_patient_is_rejected(self):
        self.rows[1]["patient_id"] = self.rows[0]["patient_id"]
        self.write_inputs()
        with self.assertRaises(ValueError):
            train.read_inputs(self.manifest)

    def test_unknown_site_is_rejected(self):
        self.rows[0]["site_id"] = "99"
        self.write_inputs()
        with self.assertRaises(ValueError):
            train.read_inputs(self.manifest)

    def test_single_class_split_fails_before_output_or_feature_loading(self):
        for row in self.rows:
            if row["site_id"] == "13":
                row["CBI"] = "0"
        self.write_inputs()
        with patch.object(train, "load_features") as loader:
            with self.assertRaises(ValueError):
                train.run(self.args())
            loader.assert_not_called()
        self.assertFalse((self.root / "results").exists())

    def exercise_selection(self, eligible, embedding_field="final_embed", block=None):
        events = []

        def load(rows, modality, field, selected_block):
            self.assertEqual((field, selected_block), (embedding_field, block))
            site = int(rows[0]["site_id"])
            events.append(f"load_{site}")
            if site == 13:
                self.assertEqual(events.count("fit"), 2)
            return np.full((len(rows), 2), site, dtype=np.float32)

        def fit(*args, **kwargs):
            events.append("fit")
            self.assertNotIn("load_13", events)
            summary = dict(lr=kwargs["lr"], penalty=kwargs["kind"], alpha=kwargs["alpha"],
                           status="ok" if eligible else "no_eligible_checkpoint",
                           epoch=0 if eligible else "", val_auroc=0.7 if eligible else "")
            if not eligible:
                return None, summary
            model = torch.nn.Linear(2, 1)
            return dict(state_dict=model.state_dict(), epoch=0,
                        val_auroc=0.7, train_auroc=0.8), summary

        def evaluate(model, features, labels, device):
            events.append(f"evaluate_{int(features[0, 0])}")
            logits = 2 * np.asarray(labels) - 1
            return metric_values(labels, logits), logits

        with patch.object(train, "load_features", side_effect=load), \
                patch.object(train, "fit_trial", side_effect=fit), \
                patch.object(train, "evaluate", side_effect=evaluate):
            args = self.args()
            args.embedding_field, args.block = embedding_field, block
            results = train.run(args)
        return events, results

    def test_test_features_load_only_after_all_trial_selection(self):
        events, results = self.exercise_selection(eligible=True)
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(events, ["load_1", "load_12", "fit", "fit",
                                  "evaluate_1", "evaluate_12", "load_13", "evaluate_13"])
        self.assertTrue((self.root / "results" / "T1" / "CBI" / "fold_1" / "predictions.csv").exists())

    def test_no_eligible_checkpoint_never_loads_or_evaluates_test_features(self):
        events, results = self.exercise_selection(eligible=False)
        self.assertEqual(results[0]["status"], "no_eligible_checkpoint")
        self.assertEqual(events, ["load_1", "load_12", "fit", "fit"])
        self.assertFalse((self.root / "results" / "T1" / "CBI" / "fold_1" / "predictions.csv").exists())

    def test_block_selection_reaches_all_splits_and_saved_metadata(self):
        import json
        _, results = self.exercise_selection(eligible=True, embedding_field="block_embed", block=9)
        self.assertEqual((results[0]["embedding_field"], results[0]["block"]), ("block_embed", 9))
        path = self.root / "results" / "T1" / "CBI" / "fold_1" / "best_model.pt"
        metadata = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
        self.assertEqual((metadata["embedding_field"], metadata["block"]), ("block_embed", 9))
        config = json.loads((self.root / "results" / "run_config.json").read_text())
        self.assertEqual((config["embedding_field"], config["block"]), ("block_embed", 9))

    def test_cli_rejects_inconsistent_embedding_selection(self):
        from contextlib import redirect_stderr
        import io
        base = ["--manifest", str(self.manifest), "--output-dir", str(self.root / "results")]
        for options in (["--embedding-field", "block_embed"], ["--block", "6"],
                        ["--embedding-field", "block_embed", "--block", "1"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(base + options)


if __name__ == "__main__":
    unittest.main()
