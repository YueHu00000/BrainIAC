"""Synthetic geometry and CLI tests; no patient data or pixel decoding."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset

import test_summarize_dicom_t1t2 as fixtures
from summarize_dicom_t1t2 import summarize_series
from summarize_dicom_geometry import measure_geometry, distribution_table


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CohortTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def source(self, positions, orientation=None):
        items = []
        for i, position in enumerate(positions):
            ds = self.fixture.dataset(instance=i + 1)
            ds.ImagePositionPatient = list(position)
            if orientation is not None:
                ds.ImageOrientationPatient = orientation
            items.append((Path(str(i)), ds))
        return dict(study_id="synthetic", modality="T1", status="selected",
                    **summarize_series(items, include_frame_positions=True))

    def test_oblique_aligned_stack(self):
        angle = np.pi / 6
        normal = np.array([0, -np.sin(angle), np.cos(angle)])
        orientation = [1, 0, 0, 0, np.cos(angle), np.sin(angle)]
        row = measure_geometry(self.source([np.zeros(3), normal * 73.5, normal * 147], orientation))
        self.assertAlmostEqual(row["coverage_mm"], 147)
        self.assertAlmostEqual(row["distance_mm"], 147)
        self.assertAlmostEqual(row["z_mm"], 147 * np.cos(angle))
        self.assertAlmostEqual(row["distance_minus_coverage_over_distance"], 0)
        self.assertAlmostEqual(row["z_minus_coverage_over_distance"], np.cos(angle) - 1)

    def test_in_plane_shift(self):
        row = measure_geometry(self.source([[0, 0, 0], [2, 0, 147]]))
        distance = np.hypot(147, 2)
        self.assertAlmostEqual(row["distance_mm"], distance)
        self.assertEqual(row["coverage_mm"], 147)
        self.assertAlmostEqual(row["distance_minus_coverage_over_distance"], (distance - 147) / distance)

    def test_farthest_pair_is_not_first_last_or_extreme_z(self):
        row = measure_geometry(self.source([[0, 0, 0], [100, 0, 5], [-100, 0, 5], [0, 0, 10]]))
        self.assertEqual((row["farthest_frame_index_a"], row["farthest_frame_index_b"]), (1, 2))
        self.assertEqual(row["distance_mm"], 200)
        self.assertEqual(row["coverage_mm"], 10)
        self.assertEqual(row["z_mm"], 0)
        self.assertAlmostEqual(row["z_minus_coverage_over_distance"], -0.05)

    def test_second_ratio_can_be_positive(self):
        row = measure_geometry(self.source([[0, 0, 0], [0, 0, 10]], [1, 0, 0, 0, 0, 1]))
        self.assertEqual(row["coverage_mm"], 0)
        self.assertEqual(row["z_minus_coverage_over_distance"], 1)

    def test_repeated_planes_and_zero_distance(self):
        row = measure_geometry(self.source([[0, 0, z] for z in [0, 7, 14, 0, 7, 14]]))
        self.assertEqual(row["distance_mm"], 14)
        self.assertEqual(row["duplicate_plane_count"], 3)
        zero = measure_geometry(self.source([[0, 0, 0], [0, 0, 0]]))
        self.assertEqual(zero["geometry_status"], "zero_distance")
        self.assertTrue(np.isnan(zero["distance_minus_coverage_over_distance"]))

    def test_unknown_is_not_zero_in_statistics(self):
        source = self.source([[0, 0, 0], [0, 0, 10]])
        good = measure_geometry(source)
        source.pop("frame_positions_mm")
        bad = measure_geometry(source)
        stats = distribution_table([good, bad])
        row = stats[(stats.modality == "T1") & (stats.metric == "distance_mm")].iloc[0]
        self.assertEqual((row.n_total, row.n_valid, row.n_unknown), (2, 1, 1))
        self.assertEqual(row["median"], 10)

    def test_enhanced_multiframe(self):
        ds = self.fixture.dataset()
        ds.NumberOfFrames = 3
        ds.PerFrameFunctionalGroupsSequence = []
        for z in [0, 10, 20]:
            group, position = Dataset(), Dataset()
            position.ImagePositionPatient = [0, 0, z]
            group.PlanePositionSequence = [position]
            ds.PerFrameFunctionalGroupsSequence.append(group)
        source = dict(study_id="enhanced", modality="T2", status="selected",
                      **summarize_series([(Path("multi"), ds)], include_frame_positions=True))
        row = measure_geometry(source)
        self.assertEqual(row["distance_mm"], 20)
        self.assertEqual(row["farthest_frame_index_b"], 2)
        del ds.PerFrameFunctionalGroupsSequence
        bad = summarize_series([(Path("multi"), ds)], include_frame_positions=True)
        self.assertNotIn("frame_positions_mm", bad)

    def test_cli_multiple_roots_and_original_selection(self):
        root = self.fixture.root
        for study in ("A", "B"):
            for number, description in [(1, "T1"), (2, "AX T1"), (3, "AX T2")]:
                for i, z in enumerate((0, 147)):
                    self.fixture.write(study, f"{number}_{i}.dcm", z=z, instance=i + 1,
                                       number=number, description=description)
        first, second = root / "root1", root / "root2"
        first.mkdir()
        second.mkdir()
        (root / "A").rename(first / "A")
        (root / "B").rename(second / "B")
        (root / "folders.txt").write_text("root1\nroot2\n", encoding="utf-8")
        (root / "labels.csv").write_text("Study_ID,unused\nA,x\nB,y\nmissing,z\n", encoding="utf-8")
        script = Path(__file__).with_name("summarize_dicom_geometry.py")
        process = subprocess.run([sys.executable, str(script), "--csv", str(root / "labels.csv"),
                                  "--folder-list", str(root / "folders.txt"),
                                  "--output-dir", str(root / "reports")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        series = pd.read_csv(root / "reports/series_geometry.csv")
        self.assertEqual(len(series), 6)
        self.assertEqual(series.loc[0, "series_number"], 2)
        self.assertEqual(float(series.loc[0, "coverage_mm"]), 147)
        self.assertEqual(series.loc[4, "status"], "study_not_found")
        self.assertEqual(series.loc[4, "distance_mm"], "unknown")
        self.assertEqual(len(pd.read_csv(root / "reports/study_geometry.csv")), 3)
        self.assertTrue((root / "reports/summary.txt").is_file())


if __name__ == "__main__":
    unittest.main()
