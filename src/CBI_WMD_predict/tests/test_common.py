"""Check feature identity, concatenation order, and malformed representation errors."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CBI_WMD_predict.common import FOLDS, load_features


def record(value, size=768, blocks=None):
    fields = [("series_description", "U8"), ("final_embed", "f4", (size,))]
    if blocks is not None:
        fields.append(("block_embed", "f4", np.asarray(blocks).shape))
    result = np.empty((), dtype=fields)
    result["series_description"] = "MRI"
    result["final_embed"] = value
    if blocks is not None:
        result["block_embed"] = blocks
    return result


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "S_001.npz"
        self.row = {"patient_id": "P_001", "study_id": "S_001",
                    "embedding_path": str(self.path),
                    "t1_unique_id": "S_001_1.2.1_9", "t2_unique_id": "S_001_1.2.2_10"}

    def write_records(self, first, second):
        np.savez(self.path, **{self.row["t1_unique_id"]: first,
                             self.row["t2_unique_id"]: second})

    def test_exact_feature_order_and_values(self):
        self.write_records(record(3), record(7))
        t1 = load_features([self.row], "T1")
        t2 = load_features([self.row], "T2")
        both = load_features([self.row], "T1T2")
        self.assertEqual(t1.shape, (1, 768))
        self.assertEqual(t2.shape, (1, 768))
        self.assertEqual(both.shape, (1, 1536))
        self.assertEqual(both.dtype, np.float32)
        np.testing.assert_array_equal(both[:, :768], t1)
        np.testing.assert_array_equal(both[:, 768:], t2)
        np.testing.assert_array_equal(t1, np.full((1, 768), 3))

    def test_wrong_dimension_is_rejected(self):
        self.write_records(record(1, size=756), record(2))
        with self.assertRaises((ValueError, TypeError)):
            load_features([self.row], "T1")

    def test_nonfinite_embedding_is_rejected(self):
        self.write_records(record(np.nan), record(2))
        with self.assertRaises(ValueError):
            load_features([self.row], "T1T2")

    def test_missing_selected_key_is_rejected(self):
        np.savez(self.path, unrelated=record(4))
        with self.assertRaises((ValueError, KeyError)):
            load_features([self.row], "T1")

    def test_block_numbers_select_the_exact_row_for_both_modalities(self):
        self.write_records(record(90, blocks=np.repeat([[3], [6], [9], [12]], 768, axis=1)),
                           record(91, blocks=np.repeat([[30], [60], [90], [120]], 768, axis=1)))
        for block in (3, 6, 9, 12):
            both = load_features([self.row], "T1T2", "block_embed", block)
            np.testing.assert_array_equal(both[0, :768], np.full(768, block))
            np.testing.assert_array_equal(both[0, 768:], np.full(768, block * 10))
        final = load_features([self.row], "T1T2", "final_embed")
        np.testing.assert_array_equal(final[0, :768], np.full(768, 90))
        np.testing.assert_array_equal(final[0, 768:], np.full(768, 91))

    def test_only_selected_block_must_be_finite(self):
        blocks = np.full((4, 768), np.nan, dtype=np.float32)
        blocks[1] = 6
        self.write_records(record(np.nan, blocks=blocks), record(2))
        np.testing.assert_array_equal(load_features([self.row], "T1", "block_embed", 6),
                                      np.full((1, 768), 6))
        with self.assertRaises(ValueError):
            load_features([self.row], "T1", "block_embed", 3)

    def test_block_shape_and_missing_field_are_rejected(self):
        self.write_records(record(1, blocks=np.zeros((3, 768))), record(2))
        with self.assertRaises(ValueError):
            load_features([self.row], "T1", "block_embed", 3)
        with self.assertRaises(ValueError):
            load_features([self.row], "T2", "block_embed", 3)

    def test_fold_groups_match_documented_ten_thousand_plan(self):
        expected = {
            1: ((1, 2, 3, 6, 7, 9, 10, 11), (12,), (13, 14)),
            2: ((1, 2, 3, 6, 7, 9, 13, 14), (10, 11), (12,)),
            3: ((1, 2, 3, 6, 7, 12, 13, 14), (9,), (10, 11)),
            4: ((1, 2, 3, 6, 10, 11, 12, 13, 14), (7,), (9,)),
            5: ((1, 2, 3, 9, 10, 11, 12, 13, 14), (6,), (7,)),
        }
        for fold, groups in expected.items():
            self.assertEqual(tuple(FOLDS[fold]["train"]), groups[0])
            self.assertEqual(tuple(FOLDS[fold]["val"]), groups[1])
            self.assertEqual(tuple(FOLDS[fold]["test"]), groups[2])
            self.assertFalse(set(groups[0]) & set(groups[1]))
            self.assertFalse(set(groups[0]) & set(groups[2]))


if __name__ == "__main__":
    unittest.main()
