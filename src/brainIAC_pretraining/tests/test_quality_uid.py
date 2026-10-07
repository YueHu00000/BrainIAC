"""Quality must match the UID-based selection, not legacy series-number IDs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining.quality import build_quality
from brainIAC_pretraining._common import read_rows, write_rows
from test_inventory import dicom


def selection(uid, acquisition):
    row = dict(unique_id=f"Study_1_{uid}_10", study_id="Study_1",
               series_instance_uid=uid, series_number="10",
               study_directory="source/Study_1", acquisition_number=str(acquisition),
               file_names=json.dumps([f"{uid}/a.dcm", f"{uid}/b.dcm"]))
    statistics = dict(row, file_count="2", frame_count="2",
                      unique_slice_count="2", coverage_mm="7")
    return row, statistics


class QualityUIDTests(unittest.TestCase):
    def test_new_statistics_manifest_and_quality_clis_agree(self):
        scripts = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dicom"
            for uid, acquisitions in (("1.2.3.10", (1, 2)), ("1.2.3.11", (9, 10))):
                for acquisition in acquisitions:
                    for instance in (1, 2):
                        dicom(source, 10, acquisition, instance, uid=uid)
            dicom(source, 893, 0, 1, uid="1.2.3.12", description="FL:A/PJN",
                  image_type=["DERIVED", "SECONDARY", "PROJECTION IMAGE", "IVI"])
            write_rows(root / "labels.csv", [{"Study_ID": "Study_1"}], ["Study_ID"])
            (root / "folders.txt").write_text("dicom\n", encoding="utf-8")
            for script, output in ((scripts / "manifest.py", "inventory"),
                                   (scripts / "stat/summary_dicom_SeriesInstanceUID.py", "statistics")):
                subprocess.run([sys.executable, str(script), "--csv", str(root / "labels.csv"),
                                "--folder-list", str(root / "folders.txt"),
                                "--output-dir", str(root / output)],
                               check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, str(scripts / "quality.py"),
                            "--manifest", str(root / "inventory/manifest.csv"),
                            "--selected-csv", str(root / "statistics/selected_series.csv"),
                            "--output-dir", str(root / "quality")],
                           check=True, capture_output=True, text=True)
            self.assertEqual(len(read_rows(root / "statistics/selected_series.csv")), 3)
            manifest = read_rows(root / "inventory/manifest.csv")
            self.assertEqual([row["acquisition_number"] for row in manifest], ["2", "10"])
            accepted = read_rows(root / "quality/image_number_coverage.csv")
            self.assertEqual([row["unique_id"] for row in accepted],
                             [row["unique_id"] for row in manifest])
            self.assertTrue(all(row["frame_count"] == "2" and float(row["coverage_mm"]) == 7
                                for row in accepted))
            self.assertEqual(read_rows(root / "quality/rejected_quality.csv"), [])

    def test_same_series_number_matches_each_uid_and_its_acquisition(self):
        first, first_stats = selection("1.2.3.10", 2)
        second, second_stats = selection("1.2.3.11", 9)
        _, projection_stats = selection("1.2.3.12", 0)
        accepted, rejected = build_quality(
            [first, second], [second_stats, projection_stats, first_stats])
        self.assertFalse(rejected)
        self.assertEqual([row["unique_id"] for row in accepted],
                         [first["unique_id"], second["unique_id"]])
        self.assertEqual(list(accepted[0]), ["unique_id", "frame_count", "coverage_mm"])
        second_stats["acquisition_number"] = "2"
        accepted, rejected = build_quality([first, second], [first_stats, second_stats])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [dict(unique_id=second["unique_id"],
                                        reason="selected_source_mismatch")])

    def test_legacy_statistics_fail_with_new_version_message(self):
        manifest, statistics = selection("1.2.3.10", 2)
        statistics.pop("series_instance_uid")
        statistics["unique_id"] = "Study_1_10"
        with self.assertRaisesRegex(ValueError, "SeriesInstanceUID.*statistics CSV"):
            build_quality([manifest], [statistics])

    def test_uid_column_cannot_disagree_with_id(self):
        manifest, statistics = selection("1.2.3.10", 2)
        statistics["series_instance_uid"] = "1.2.3.11"
        with self.assertRaisesRegex(ValueError, "unique_id must equal"):
            build_quality([manifest], [statistics])


if __name__ == "__main__":
    unittest.main()
