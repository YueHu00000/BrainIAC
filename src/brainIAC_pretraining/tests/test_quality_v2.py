"""Descriptions in quality CSV retain UID alignment and original screening rules."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining import quality, quality_v2
from brainIAC_pretraining._common import read_rows, write_rows
from brainIAC_pretraining.manifest import FIELDS as MANIFEST_FIELDS, build_manifest
from test_inventory import dicom, rows


class QualityV2Tests(unittest.TestCase):
    def test_statistics_descriptions_match_uid_not_row_order_without_header_reads(self):
        first, first_stats = rows()
        second, second_stats = rows("Study_1_1.2.3.11_11")
        first_stats["series_description"] = "AX T1, POST"
        second_stats["series_description"] = "T2 FLAIR"
        with patch("brainIAC_pretraining._dicom.pydicom.dcmread", side_effect=AssertionError("Unexpected header read")):
            accepted, rejected = quality_v2.build_quality([first, second], [second_stats, first_stats])
        self.assertFalse(rejected)
        self.assertEqual([row["series_description"] for row in accepted], ["AX T1, POST", "T2 FLAIR"])
        self.assertEqual(list(accepted[0]), quality_v2.FIELDS)
        original, _ = quality.build_quality([first, second], [first_stats, second_stats])
        self.assertEqual([{key: row[key] for key in quality.FIELDS} for row in accepted], original)

    def test_header_mode_uses_only_selected_acquisition_and_joins_descriptions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dicom(root, 10, 1, 1, description="OLD T1")
            dicom(root, 10, 2, 1, description="T2 FLAIR")
            dicom(root, 10, 2, 2, description="AX T2")
            manifest, _ = build_manifest(["Study_1"], [root])
            accepted, rejected = quality_v2.build_quality(manifest)
        self.assertFalse(rejected)
        self.assertEqual(accepted, [dict(unique_id="Study_1_1.2.3.10_10",
                                         series_description="AX T2 | T2 FLAIR", frame_count=2, coverage_mm=7.0)])

    def test_rejection_and_multiframe_behavior_is_preserved(self):
        manifest, selected = rows(unique_slice_count="22")
        selected["series_description"] = "AX T2"
        accepted, rejected = quality_v2.build_quality([manifest], [selected])
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [dict(unique_id=manifest["unique_id"], reason="count_mismatch")])
        selected.update(file_count="1", frame_count="23")
        with self.assertRaisesRegex(ValueError, "unsupported multi-frame"):
            quality_v2.build_quality([manifest], [selected])

    def test_cli_column_order_description_escaping_and_exclude_compatibility(self):
        from brainIAC_pretraining.exclude import exclusion_reason
        manifest, selected = rows()
        selected["series_description"] = "AX T2, FLAIR"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_rows(root / "manifest.csv", [manifest], MANIFEST_FIELDS)
            write_rows(root / "selected.csv", [selected], list(selected))
            result = subprocess.run([sys.executable, str(Path(quality_v2.__file__)),
                                     "--manifest", str(root / "manifest.csv"),
                                     "--selected-csv", str(root / "selected.csv"),
                                     "--output-dir", str(root / "output")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            accepted = read_rows(root / "output/image_number_coverage.csv")
            self.assertEqual(list(accepted[0]), ["unique_id", "series_description", "frame_count", "coverage_mm"])
            self.assertEqual(accepted[0]["series_description"], "AX T2, FLAIR")
            self.assertEqual(exclusion_reason(accepted[0], 20, 100), "")
            self.assertEqual(read_rows(root / "output/rejected_quality.csv"), [])

    def test_empty_cohort_and_empty_description(self):
        self.assertEqual(quality_v2.build_quality([], []), ([], []))
        manifest, selected = rows()
        selected["series_description"] = ""
        accepted, _ = quality_v2.build_quality([manifest], [selected])
        self.assertEqual(accepted[0]["series_description"], "")


if __name__ == "__main__":
    unittest.main()
