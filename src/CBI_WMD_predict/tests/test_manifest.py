"""Selection and cohort tests use synthetic metadata and numeric NPZ records."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CBI_WMD_predict import manifest


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def packed_vector(description, value=1.0, size=768):
    record = np.empty((), dtype=[
        ("series_description", "U64"),
        ("block_embed", "f4", (4, 768)),
        ("final_embed", "f4", (size,)),
    ])
    record["series_description"] = description
    record["block_embed"] = value
    record["final_embed"] = value
    return record


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.embeddings = self.root / "embeddings"
        self.embeddings.mkdir()
        self.series = []
        self.labels = []
        self.quality = []
        self.records = {}

    def add_study(self, study="S_001", patient="P_001", site=1,
                  cbi="0", wmd="1"):
        self.labels.append({"Study_ID": study, "Patient_ID": patient,
                            "SiteID": str(site), "idCBI": cbi, "idWMD": wmd})
        return study

    def add_series(self, study, number, description, *, frames=20,
                   coverage=100, include_embedding=True):
        uid = f"1.2.840.{len(self.series) + 1}"
        unique_id = f"{study}_{uid}_{number}"
        self.series.append({
            "unique_id": unique_id, "study_id": study,
            "series_instance_uid": uid, "series_number": str(number),
            "study_directory": str(self.root / study),
            "acquisition_number": "1", "file_names": "[]",
            "series_description": description,
        })
        self.quality.append({"unique_id": unique_id,
                             "series_description": description,
                             "frame_count": str(frames),
                             "coverage_mm": str(coverage)})
        if include_embedding:
            self.records.setdefault(study, {})[unique_id] = packed_vector(description)
        return unique_id

    def add_good_pair(self, study="S_001", patient="P_001", site=1):
        self.add_study(study, patient, site)
        return (self.add_series(study, 2, "ax t1"),
                self.add_series(study, 3, "Ax T2"))

    def build(self, output_name="manifests"):
        upstream_fields = ["unique_id", "study_id", "series_instance_uid",
                           "series_number", "study_directory",
                           "acquisition_number", "file_names"]
        write_csv(self.root / "upstream.csv",
                  [{k: row[k] for k in upstream_fields} for row in self.series],
                  upstream_fields)
        write_csv(self.root / "final.csv",
                  [{"unique_id": row["unique_id"]} for row in self.series],
                  ["unique_id"])
        write_csv(self.root / "selected.csv", self.series,
                  upstream_fields + ["series_description"])
        write_csv(self.root / "quality.csv", self.quality,
                  ["unique_id", "series_description", "frame_count", "coverage_mm"])
        write_csv(self.root / "labels.csv", self.labels,
                  ["Study_ID", "Patient_ID", "SiteID", "idCBI", "idWMD"])
        for study, records in self.records.items():
            np.savez_compressed(self.embeddings / f"{study}.npz", **records)
        output = self.root / output_name
        args = manifest.parse_args([
            "--manifest", str(self.root / "upstream.csv"),
            "--final-csv", str(self.root / "final.csv"),
            "--selected-csv", str(self.root / "selected.csv"),
            "--quality-csv", str(self.root / "quality.csv"),
            "--labels-csv", str(self.root / "labels.csv"),
            "--embedding-dir", str(self.embeddings),
            "--output-dir", str(output),
            "--patient-id-column", "Patient_ID",
        ])
        manifest.build(args)
        return output

    def test_quality_boundaries_and_numeric_max_before_embedding(self):
        study = self.add_study()
        self.add_series(study, 9, "t1", frames=20, coverage=100)
        winner = self.add_series(study, 10, "T1", frames=20, coverage=100)
        self.add_series(study, 11, "T1", frames=19)
        self.add_series(study, 12, "T1", frames=41)
        self.add_series(study, 13, "T1", coverage=99.99)
        t2 = self.add_series(study, 3, "t2", frames=40, coverage=100)
        output = self.build()
        rows = read_csv(output / "manifest_t1t2.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["t1_unique_id"], winner)
        self.assertEqual(rows[0]["t2_unique_id"], t2)
        self.assertEqual(rows[0]["t1_series_number"], "10")
        counts = read_csv(output / "cohort_counts.csv")[0]
        self.assertEqual((counts["candidate_series"], counts["eligible_series"],
                          counts["selected_series"]), ("6", "3", "2"))

    def test_unknown_upstream_acquisition_is_retained(self):
        self.add_good_pair()
        self.series[0]["acquisition_number"] = "unknown"
        output = self.build()
        rows = read_csv(output / "manifest_t1t2.csv")
        self.assertEqual(rows[0]["t1_acquisition_number"], "unknown")

    def test_three_manifests_share_paired_patients_and_exact_harvey_folds(self):
        self.add_good_pair()
        self.add_good_pair("S_002", "P_002", site=13)
        self.add_study("S_003", "P_003", site=9)
        self.add_series("S_003", 1, "T1")  # Missing T2 excludes all modes.
        output = self.build()
        modes = [read_csv(output / f"manifest_{mode}.csv")
                 for mode in ("t1", "t2", "t1t2")]
        identities = [[(r["patient_id"], r["study_id"], r["CBI"], r["WMD"])
                       for r in rows] for rows in modes]
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[1], identities[2])
        self.assertEqual({row["study_id"] for row in modes[0]}, {"S_001", "S_002"})
        folds = read_csv(output / "folds.csv")
        self.assertEqual(len(folds), 10)
        first = {(int(row["fold"]), row["study_id"]): row["split"] for row in folds}
        self.assertEqual(first[(1, "S_002")], "test")
        self.assertTrue(all(first[(fold, "S_001")] == "train" for fold in range(1, 6)))
        self.assertEqual({row["study_id"] for row in read_csv(output / "excluded_studies.csv")},
                         {"S_003"})

    def test_missing_winner_key_does_not_fall_back(self):
        self.add_good_pair("S_OK", "P_OK")
        study = self.add_study()
        self.add_series(study, 1, "T1")
        missing = self.add_series(study, 10, "T1", include_embedding=False)
        self.add_series(study, 2, "T2")
        output = self.build()
        rows = read_csv(output / "manifest_t1.csv")
        self.assertEqual([row["study_id"] for row in rows], ["S_OK"])
        audit = (output / "excluded_studies.csv").read_text(encoding="utf-8-sig")
        self.assertIn(study, audit)
        self.assertTrue("embedding" in audit.lower() or missing in audit)

    def test_equal_winning_numbers_are_not_resolved_by_uid_sort(self):
        self.add_good_pair("S_OK", "P_OK")
        study = self.add_study()
        self.add_series(study, 10, "T1")
        self.add_series(study, 10, "T1 repeat")
        self.add_series(study, 2, "T2")
        output = self.build()
        self.assertEqual([r["study_id"] for r in read_csv(output / "manifest_t1t2.csv")],
                         ["S_OK"])
        self.assertIn(study, (output / "excluded_studies.csv").read_text(encoding="utf-8-sig"))

    def test_ambiguous_description_is_not_both_modalities(self):
        self.add_good_pair("S_OK", "P_OK")
        study = self.add_study()
        ambiguous = self.add_series(study, 99, "T1 T2 combined")
        output = self.build()
        rows = read_csv(output / "manifest_t1t2.csv")
        self.assertEqual([r["study_id"] for r in rows], ["S_OK"])
        audit = (output / "series_selection.csv").read_text(encoding="utf-8-sig")
        self.assertIn(ambiguous, audit)
        self.assertIn("ambiguous", audit.lower())

    def test_duplicate_patient_requires_explicit_index_study_selection(self):
        self.add_good_pair("S_001", "P_SAME")
        self.add_good_pair("S_002", "P_SAME")
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.root / "manifests").exists())

    def test_missing_labels_are_retained_but_invalid_binary_labels_fail(self):
        self.add_good_pair()
        self.labels[0]["idCBI"] = ""
        output = self.build()
        self.assertEqual(read_csv(output / "manifest_t1.csv")[0]["CBI"], "")
        self.labels[0]["idCBI"] = "2"
        with self.assertRaises(ValueError):
            self.build("invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_output_overwrite_is_rejected(self):
        self.add_good_pair()
        output = self.build()
        saved = (output / "manifest_t1.csv").read_bytes()
        with self.assertRaises((FileExistsError, ValueError)):
            self.build()
        self.assertEqual((output / "manifest_t1.csv").read_bytes(), saved)

    def test_sites_outside_harvey_plan_are_reported(self):
        self.add_good_pair("S_OK", "P_OK")
        self.add_good_pair("S_SITE5", "P_SITE5", site=5)
        output = self.build()
        self.assertEqual([r["study_id"] for r in read_csv(output / "manifest_t1.csv")],
                         ["S_OK"])
        self.assertIn("S_SITE5", (output / "excluded_studies.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
