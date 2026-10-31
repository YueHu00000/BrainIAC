"""Integration checks with synthetic legacy artifacts, never patient data."""

import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from vit_survival_embeding_convert.common import SITES


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ConversionFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / "legacy"
        self.embedding_dir = self.legacy / "embeddings" / "tufts_train_global_v1"
        self.embedding_dir.mkdir(parents=True)
        self.selected = []
        self.labels = []
        self.original = {}

    def add_series(self, study, number, description, **updates):
        uid = f"1.2.840.{len(self.selected) + 1}"
        row = dict(unique_id=f"{study}_{uid}_{number}", study_id=study,
                   series_instance_uid=uid, series_number=str(number),
                   series_description=description,
                   series_descriptions=json.dumps([description]),
                   acquisition_number="1", acquisition_numbers="[1]",
                   source_file_count="23", discarded_file_count="0",
                   missing_acquisition_file_count="0", file_count="23",
                   frame_count="23", coverage_mm="140", status="selected", flags="[]")
        row.update({key: str(value) for key, value in updates.items()})
        self.selected.append(row)
        return row

    def add_study(self, study="S_001", site=1, *, version=1, label=0, signal=None,
                  metadata_updates=None, norm_updates=None, source_updates=None):
        self.labels.append(dict(Study_ID=study, SiteID=str(site),
                                idCBI=str(label), idWMD=str(label)))
        first = self.add_series(study, 2, "Ax T1")
        second = self.add_series(study, 3, "Ax T2")
        layers = [3, 6, 9, 12] if version == 1 else list(range(1, 13))
        shape = (2, 3, len(layers), 768)
        blocks = np.broadcast_to(
            np.arange(2 * 3 * len(layers), dtype=np.float32).reshape(2, 3, len(layers), 1),
            shape).copy()
        final = np.broadcast_to(np.arange(6, dtype=np.float32).reshape(2, 3, 1) + 1000,
                                (2, 3, 768)).copy()
        if signal is not None:
            blocks.fill(0)
            final.fill(0)
            blocks[:, :, :, :2] = signal
            final[:, :, :2] = signal
        norm = dict(normalization_id="tufts_train_global_v1", fit_split="train",
                    mean=[0.0, 0.0], std=[1.0, 1.0], fit_study_ids=["FIT_REFERENCE"])
        norm.update(norm_updates or {})
        source = dict(prepared_path=f"/synthetic/prepared/{study}.npz",
                      prepared_sidecar={"study_id": study, "purpose": "synthetic test"})
        source.update(source_updates or {})
        metadata = dict(
            schema_version=f"vit_multilayer_embedding_v{version}", study_id=study,
            normalization_id="tufts_train_global_v1",
            normalization_provenance={"normalization_id": "tufts_train_global_v1",
                                      "parameters": norm},
            source_provenance=source, weights_id="synthetic_vit_weights",
            transform_id="synthetic_224", vit_implementation="synthetic:vit_b_16",
            single_channel_adapter="sum_rgb_weights", layers=layers, slice_count=3,
            shape={"block_cls": list(blocks.shape), "final_cls": list(final.shape)},
            dtype="float32", code_version="synthetic",
            layer_semantics={"block_cls": "CLS before final LayerNorm",
                             "final_cls": "CLS after final LayerNorm"},
        )
        metadata.update(metadata_updates or {})
        path = self.embedding_dir / f"{study}.npz"
        self.write_artifact(path, blocks, final, metadata)
        self.original[study] = (blocks, final, metadata, first, second)
        return self.original[study]

    @staticmethod
    def write_artifact(path, blocks, final, metadata):
        np.savez_compressed(path, block_cls=blocks, final_cls=final,
                            metadata_json=np.asarray(json.dumps(metadata)))

    def write_inputs(self):
        write_rows(self.root / "selected.csv", self.selected)
        write_rows(self.root / "labels.csv", self.labels)

    def args(self, name="converted", audit_only=False):
        self.write_inputs()
        converter = importlib.import_module("vit_survival_embeding_convert.convert")
        command = ["--legacy-results-dir", str(self.legacy),
                   "--selected-csv", str(self.root / "selected.csv"),
                   "--labels-csv", str(self.root / "labels.csv"),
                   "--output-dir", str(self.root / name)]
        if audit_only:
            command.append("--audit-only")
        return converter, converter.parse_args(command)

    def run_conversion(self, name="converted", audit_only=False):
        converter, args = self.args(name, audit_only)
        summary = converter.run(args)
        self.assertIsInstance(summary, dict)
        return self.root / name, summary

    def manifested_studies(self, output):
        all_rows = [read_rows(output / "manifests" / f"manifest_{mode}.csv")
                    for mode in ("T1", "T2", "T1T2")]
        identities = [[(row["patient_id"], row["study_id"], row["site_id"],
                        row["CBI"], row["WMD"]) for row in rows] for rows in all_rows]
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[1], identities[2])
        return {row["study_id"] for row in all_rows[0]}


class ConversionIntegrationTests(ConversionFixture):
    def test_v1_v2_conversion_preserves_exact_slice_means_and_pair_identity(self):
        self.add_study("S_V1", version=1)
        self.add_study("S_V2", site=13, version=2, label=1)
        output, _ = self.run_conversion()
        self.assertEqual(self.manifested_studies(output), {"S_V1", "S_V2"})
        for study, (blocks, final, metadata, first, second) in self.original.items():
            positions = [metadata["layers"].index(block) for block in (3, 6, 9, 12)]
            with np.load(output / "embeddings" / f"{study}.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {first["unique_id"], second["unique_id"]})
                for modality, row in enumerate((first, second)):
                    record = archive[row["unique_id"]]
                    self.assertEqual(record.shape, ())
                    np.testing.assert_array_equal(record["block_embed"],
                                                  blocks[modality].mean(axis=0)[positions])
                    np.testing.assert_array_equal(record["final_embed"], final[modality].mean(axis=0))
                    self.assertFalse(np.array_equal(record["block_embed"][-1], record["final_embed"]))
        self.assertTrue((output / "embeddings" / "embedding_config.json").exists())
        self.assertTrue((output / "source_metadata.jsonl").exists())
        self.assertTrue((output / "conversion_summary.json").exists())

    def test_generated_manifests_load_exact_features_for_every_mode_and_block(self):
        from CBI_WMD_predict.common import load_features

        self.add_study("S_V1", version=1)
        self.add_study("S_V2", site=13, version=2, label=1)
        output, _ = self.run_conversion()
        manifests = {mode: read_rows(output / "manifests" / f"manifest_{mode}.csv")
                     for mode in ("T1", "T2", "T1T2")}
        for field, block in [("final_embed", None),
                             *(("block_embed", number) for number in (3, 6, 9, 12))]:
            actual = {}
            for mode, rows in manifests.items():
                expected = []
                for row in rows:
                    blocks, final, metadata, _, _ = self.original[row["study_id"]]
                    if field == "final_embed":
                        modality_vectors = final.mean(axis=1)
                    else:
                        position = metadata["layers"].index(block)
                        modality_vectors = blocks.mean(axis=1)[:, position, :]
                    vector = (np.concatenate(modality_vectors) if mode == "T1T2"
                              else modality_vectors[0 if mode == "T1" else 1])
                    expected.append(vector)
                actual[mode] = load_features(rows, mode, field, block)
                np.testing.assert_array_equal(actual[mode], np.stack(expected),
                                              err_msg=f"{mode}/{field}/block={block}")
                self.assertEqual(actual[mode].dtype, np.float32)
            np.testing.assert_array_equal(actual["T1T2"][:, :768], actual["T1"])
            np.testing.assert_array_equal(actual["T1T2"][:, 768:], actual["T2"])

    def test_failed_maximum_quality_cannot_fall_back_to_lower_series(self):
        self.add_study("S_OK")
        self.add_study("S_BAD")
        self.add_series("S_BAD", 20, "T1 repeat", coverage_mm=99)
        output, _ = self.run_conversion()
        self.assertEqual(self.manifested_studies(output), {"S_OK"})
        self.assertFalse((output / "embeddings" / "S_BAD.npz").exists())
        audit = (output / "legacy_series_mapping.csv").read_text(encoding="utf-8-sig")
        self.assertIn("S_BAD", audit)
        self.assertIn("low_coverage", audit)

    def test_missing_t2_removes_study_from_all_three_modes(self):
        self.add_study("S_OK")
        self.add_study("S_MISSING")
        self.selected = [row for row in self.selected
                         if not (row["study_id"] == "S_MISSING" and "T2" in row["series_description"])]
        output, _ = self.run_conversion()
        self.assertEqual(self.manifested_studies(output), {"S_OK"})

    def test_duplicate_legacy_artifact_is_excluded_not_chosen_by_directory_order(self):
        self.add_study("S_OK")
        self.add_study("S_DUP")
        duplicate = self.embedding_dir / "duplicate"
        duplicate.mkdir()
        shutil.copyfile(self.embedding_dir / "S_DUP.npz", duplicate / "S_DUP.npz")
        output, _ = self.run_conversion()
        self.assertEqual(self.manifested_studies(output), {"S_OK"})
        self.assertIn("duplicate", (output / "legacy_series_mapping.csv").read_text(encoding="utf-8-sig"))

    def test_wrong_normalization_and_wrong_study_metadata_are_not_converted(self):
        self.add_study("S_OK")
        self.add_study("S_NORM", metadata_updates={"normalization_id": "per_study_per_modality_v1"})
        self.add_study("S_ID", metadata_updates={"study_id": "S_OTHER"})
        output, _ = self.run_conversion()
        self.assertEqual(self.manifested_studies(output), {"S_OK"})
        self.assertFalse((output / "embeddings" / "S_NORM.npz").exists())
        self.assertFalse((output / "embeddings" / "S_ID.npz").exists())

    def test_corrected_acquisition_two_uses_twenty_three_selected_frames(self):
        self.add_study("R01_Study_002904")
        t2 = next(row for row in self.selected if "T2" in row["series_description"])
        t2.update(acquisition_number="2", acquisition_numbers="[1, 2]",
                  source_file_count="46", file_count="23", discarded_file_count="23",
                  frame_count="23")
        output, _ = self.run_conversion()
        self.assertEqual(self.manifested_studies(output), {"R01_Study_002904"})
        manifest = read_rows(output / "manifests" / "manifest_T2.csv")[0]
        self.assertEqual(manifest["t2_acquisition_number"], "2")
        self.assertEqual(manifest["t2_frame_count"], "23")

    def test_mixed_normalization_parameters_fail_before_creating_output(self):
        self.add_study("S_ONE")
        self.add_study("S_TWO", norm_updates={"mean": [1.0, 0.0]})
        converter, args = self.args()
        with self.assertRaises(ValueError):
            converter.run(args)
        self.assertFalse((self.root / "converted").exists())

    def test_mixed_fitting_cohorts_fail_before_creating_output(self):
        self.add_study("S_ONE")
        self.add_study("S_TWO", norm_updates={"fit_study_ids": ["OTHER_REFERENCE"]})
        converter, args = self.args()
        with self.assertRaises(ValueError):
            converter.run(args)
        self.assertFalse((self.root / "converted").exists())

    def test_mixed_encoder_weights_fail_before_creating_output(self):
        self.add_study("S_ONE")
        self.add_study("S_TWO", metadata_updates={"weights_id": "different_encoder"})
        converter, args = self.args()
        with self.assertRaises(ValueError):
            converter.run(args)
        self.assertFalse((self.root / "converted").exists())

    def test_audit_only_never_publishes_embeddings_or_manifests(self):
        self.add_study()
        output, _ = self.run_conversion(audit_only=True)
        self.assertFalse(list(output.rglob("*.npz")))
        self.assertFalse(list(output.rglob("manifest_T*.csv")))
        self.assertTrue((output / "legacy_series_mapping.csv").exists())
        self.assertTrue((output / "conversion_summary.json").exists())

    def test_sources_stay_unchanged_and_output_overwrite_is_refused(self):
        self.add_study()
        converter, args = self.args()
        before = {path: path.read_bytes() for path in self.legacy.rglob("*") if path.is_file()}
        before.update({path: path.read_bytes() for path in
                       (self.root / "selected.csv", self.root / "labels.csv")})
        converter.run(args)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        with self.assertRaises((FileExistsError, ValueError)):
            converter.run(args)


class ConvertedTrainingSmokeTests(ConversionFixture):
    def test_single_manifest_final_and_block_six_train_from_external_cwd(self):
        for site in SITES:
            for index in range(16):
                signal = 1.0 if index % 2 else -1.0
                label = int(signal > 0)
                if site >= 6 and index < 8:
                    label = 1 - label
                self.add_study(f"S_{site}_{index:02d}", site, label=label, signal=signal)
        output, _ = self.run_conversion()
        trainer = SRC / "CBI_WMD_predict" / "train.py"
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for name, arguments in (("final", []),
                                ("block6", ["--embedding-field", "block_embed", "--block", "6"])):
            destination = self.root / f"training_{name}"
            command = [sys.executable, str(trainer), "--manifest",
                       str(output / "manifests" / "manifest_T1.csv"),
                       "--output-dir", str(destination), "--device", "cpu", "--folds", "1",
                       "--targets", "CBI", "--epochs", "6", "--batch-size", "8",
                       "--learning-rates", "0.1", "--alphas", "0", "--penalties", "L2",
                       *arguments]
            completed = subprocess.run(command, cwd=self.root, env=environment,
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            results = read_rows(destination / "results.csv")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "ok", completed.stdout)
            predictions = destination / "T1" / "CBI" / "fold_1" / "predictions.csv"
            self.assertTrue(predictions.exists(), str(list(destination.rglob("predictions.csv"))))


if __name__ == "__main__":
    unittest.main()
