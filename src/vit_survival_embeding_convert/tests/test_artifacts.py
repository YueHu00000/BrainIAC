"""Real legacy array layout, modality order, and provenance boundaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vit_survival_embeding_convert.artifacts import index_embeddings, load_embedding, read_metadata


def source_fixture(layers=4, study="S001", normalization="tufts_train_global_v1"):
    slices = 3
    numbers = [3, 6, 9, 12] if layers == 4 else list(range(1, 13))
    block = np.empty((2, slices, layers, 768), dtype=np.float32)
    final = np.empty((2, slices, 768), dtype=np.float32)
    for modality in range(2):
        for slice_index in range(slices):
            for layer_index, number in enumerate(numbers):
                block[modality, slice_index, layer_index] = 1000 * modality + 10 * number + slice_index
            final[modality, slice_index] = 10000 + 1000 * modality + slice_index
    metadata = {
        "schema_version": f"vit_multilayer_embedding_v{1 if layers == 4 else 2}",
        "study_id": study,
        "source_provenance": {"prepared_path": f"/prepared/{study}.npz", "prepared_sidecar": {}},
        "normalization_id": normalization,
        "normalization_provenance": {"normalization_id": normalization, "parameters": {
            "normalization_id": normalization, "fit_split": "train",
            "fit_study_ids": ["TRAIN001", "TRAIN002"], "fit_study_count": 2,
            "mean": [1.0, 2.0], "std": [3.0, 4.0],
            "aggregation": "equal_weight_study_sample_mean_and_std", "std_ddof": 1,
        }},
        "weights_id": "IMAGENET1K_V1",
        "single_channel_adapter": "rgb_conv_proj_weight_sum_v1",
        "vit_implementation": "torchvision.models.vision_transformer.VisionTransformer",
        "transform_id": "prepared_224_unnormalized_v1",
        "layers": numbers,
        "layer_semantics": {"block_cls": "pre-final-LN", "final_cls": "post-final-LN"},
        "slice_count": slices,
        "shape": {"block_cls": list(block.shape), "final_cls": list(final.shape)},
        "dtype": "float32",
        "code_version": "vit_survival_simple_v1",
    }
    return block, final, metadata


def write_fixture(path, block, final, metadata, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, block_cls=block, final_cls=final,
             metadata_json=np.asarray(json.dumps(metadata)), **extra)


class ArtifactTests(unittest.TestCase):
    def test_v1_and_v2_select_same_layers_and_only_average_slices(self):
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for layers in (4, 12):
                path = Path(directory) / str(layers) / "S001.npz"
                block, final, metadata = source_fixture(layers)
                write_fixture(path, block, final, metadata)
                block_mean, final_mean, provenance = load_embedding(path, "S001")
                self.assertEqual(block_mean.shape, (2, 4, 768))
                self.assertEqual(final_mean.shape, (2, 768))
                self.assertEqual(block_mean.dtype, np.float32)
                self.assertTrue(block_mean.flags.c_contiguous)
                np.testing.assert_array_equal(block_mean[0, :, 0], [31, 61, 91, 121])
                np.testing.assert_array_equal(block_mean[1, :, 0], [1031, 1061, 1091, 1121])
                np.testing.assert_array_equal(final_mean[:, 0], [10001, 11001])
                self.assertNotEqual(float(block_mean[0, -1, 0]), float(final_mean[0, 0]))
                self.assertEqual(provenance, metadata)
                outputs.append((block_mean, final_mean))
        for first, second in zip(*outputs):
            np.testing.assert_array_equal(first, second)

    def test_bad_arrays_and_metadata_are_rejected(self):
        mutations = {
            "bad_final_shape": lambda b, f, m: (b, f[:, :2], m),
            "empty_slices": lambda b, f, m: (b[:, :0], f[:, :0], m),
            "bad_block_width": lambda b, f, m: (b[..., :767], f, m),
            "nonfinite_block": lambda b, f, m: (b * np.float32(np.nan), f, m),
            "nonfinite_final": lambda b, f, m: (b, f * np.float32(np.inf), m),
            "float64": lambda b, f, m: (b.astype(np.float64), f, m),
            "wrong_layers": lambda b, f, m: (b, f, dict(m, layers=[1, 2, 3, 4])),
            "wrong_schema": lambda b, f, m: (b, f, dict(m, schema_version="unknown")),
            "norm_b": lambda b, f, m: (b, f, dict(m, normalization_id="per_study_per_modality_v1")),
            "wrong_study": lambda b, f, m: (b, f, dict(m, study_id="OTHER")),
            "wrong_declared_shape": lambda b, f, m: (b, f, dict(m, shape={})),
            "no_fit_cohort": lambda b, f, m: (b, f, dict(m, normalization_provenance={})),
            "no_weights": lambda b, f, m: (b, f, dict(m, weights_id="")),
            "no_transform": lambda b, f, m: (b, f, dict(m, transform_id="")),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001.npz"
            for name, mutation in mutations.items():
                with self.subTest(name=name):
                    write_fixture(path, *mutation(*source_fixture()))
                    with self.assertRaises((ValueError, TypeError)):
                        load_embedding(path, "S001")

    def test_filename_and_requested_identity_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RENAMED.npz"
            write_fixture(path, *source_fixture())
            with self.assertRaisesRegex(ValueError, "filename"):
                load_embedding(path, "S001")
            path = Path(directory) / "S001.npz"
            write_fixture(path, *source_fixture())
            with self.assertRaisesRegex(ValueError, "requested Study_ID"):
                load_embedding(path, "OTHER")

    def test_index_filters_other_branch_and_retains_duplicate_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "embeddings/norm_a/S001.npz", *source_fixture())
            write_fixture(root / "embeddings/corrected/S001.npz", *source_fixture())
            write_fixture(root / "embeddings/norm_b/S002.npz", *source_fixture(
                study="S002", normalization="per_study_per_modality_v1"))
            np.savez(root / "embeddings/pooled.npz", X=np.zeros((1, 768)))
            write_fixture(root / "outside/S003.npz", *source_fixture(study="S003"))
            index, issues = index_embeddings(root)
            self.assertEqual(set(index), {"S001"})
            self.assertEqual(len(index["S001"]), 2)
            self.assertEqual(sum(row["reason"] == "duplicate_study_embedding" for row in issues), 2)
            self.assertTrue(any(row["reason"] == "normalization_not_norm_a" for row in issues))
            self.assertTrue(any("not_raw_embedding" in row["reason"] for row in issues))
            chosen, _ = index_embeddings(root, root / "embeddings/norm_a")
            self.assertEqual(len(chosen["S001"]), 1)

    def test_index_only_reads_metadata_not_tensor_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            block, final, metadata = source_fixture()
            # Deliberately invalid tensors: discovery should still index it;
            # full tensor validation is deferred to loading selected studies.
            write_fixture(root / "embeddings/S001.npz", block[..., :2], final, metadata)
            index, issues = index_embeddings(root)
            self.assertEqual(set(index), {"S001"})
            self.assertEqual(issues, [])
            self.assertEqual(read_metadata(index["S001"][0]), metadata)
            with self.assertRaisesRegex(ValueError, "block_cls"):
                load_embedding(index["S001"][0], "S001")

    def test_index_reports_bad_metadata_without_choosing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "embeddings/S001.npz"
            block, final, metadata = source_fixture()
            path.parent.mkdir()
            np.savez(path, block_cls=block, final_cls=final, metadata_json=np.asarray("{"))
            index, issues = index_embeddings(root)
            self.assertEqual(index, {})
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0]["path"], str(path))

    def test_missing_keys_and_non_scalar_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001.npz"
            block, final, metadata = source_fixture()
            np.savez(path, final_cls=final, metadata_json=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "not_raw_embedding"):
                load_embedding(path, "S001")
            np.savez(path, block_cls=block, final_cls=final,
                     metadata_json=np.asarray([json.dumps(metadata)]))
            with self.assertRaisesRegex(ValueError, "text scalar"):
                load_embedding(path, "S001")

    def test_index_reports_corrupt_zip_and_keeps_valid_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "embeddings/S001.npz", *source_fixture())
            (root / "embeddings/broken.npz").write_bytes(b"PK\x03\x04broken")
            index, issues = index_embeddings(root)
            self.assertEqual(set(index), {"S001"})
            self.assertEqual(len(issues), 1)
            self.assertTrue(issues[0]["path"].endswith("broken.npz"))

    def test_norm_a_requires_fitting_identity_and_positive_std(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001.npz"
            for parameters in (
                    {"fit_study_ids": []},
                    {"fit_study_ids": ["TRAIN001", "TRAIN001"]},
                    {"std": [1, 0]},
                    {"mean": [1, float("nan")]},
                    {"fit_split": "all"}):
                with self.subTest(parameters=parameters):
                    block, final, metadata = source_fixture()
                    metadata["normalization_provenance"]["parameters"].update(parameters)
                    write_fixture(path, block, final, metadata)
                    with self.assertRaises(ValueError):
                        load_embedding(path, "S001")


if __name__ == "__main__":
    unittest.main()
