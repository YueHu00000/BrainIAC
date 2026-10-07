"""Match and reuse old paired outputs without decoding NIfTI images."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainIAC_pretraining import pre_process, reuse_preprocess as reuse
from brainIAC_pretraining._common import read_rows, write_rows


class ReusePreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old_dir = self.root / "old"
        self.study_dir = self.root / "dicom" / "R01_Study_1"
        self.old_dir.mkdir()

    def pair(self, modality="T1", uid="1.2.3.1", number="3", files=None):
        files = files or ["acq2/001.dcm", "acq2/002.dcm"]
        study = "R01_Study_1"
        old_file = self.old_dir / study / f"{modality}.nii.gz"
        old_file.parent.mkdir(exist_ok=True)
        old_file.write_bytes(f"not a NIfTI: {modality}".encode())
        old = {
            "study_id": study, "modality": modality, "status": "selected",
            "series_number": number, "series_instance_uids": json.dumps([uid]),
            "file_names": json.dumps(files), "study_directory": str(self.study_dir),
            "flags": "[]",
        }
        manifest = {
            "unique_id": f"{study}_{uid}_{number}", "study_id": study,
            "series_number": number, "series_instance_uid": uid,
            "acquisition_number": "2", "study_directory": str(self.study_dir),
            "file_names": json.dumps(files),
        }
        return old, manifest

    def audit(self, old, manifest):
        return reuse.audit_reuse(old, manifest, self.old_dir)

    def test_same_series_number_separate_uids_map_each_modality(self):
        t1, m1 = self.pair()
        t2, m2 = self.pair("T2", "1.2.3.2")
        rows = self.audit([t1, t2], [m1, m2])
        self.assertEqual([row["status"] for row in rows], ["reusable", "reusable"])
        self.assertEqual([row["unique_id"] for row in rows], [m1["unique_id"], m2["unique_id"]])
        self.assertEqual([row["acquisition_number"] for row in rows], ["2", "2"])
        for row in rows:
            self.assertTrue(Path(row["old_file"]).is_file())

    def test_selected_source_files_must_match_max_acquisition(self):
        old, manifest = self.pair()
        old["file_names"] = json.dumps(["acq1/001.dcm", "acq1/002.dcm",
                                         "acq2/001.dcm", "acq2/002.dcm"])
        row = self.audit([old], [manifest])[0]
        self.assertEqual((row["status"], row["reason"]),
                         ("abandoned", "selected_files_mismatch"))

    def test_ambiguous_missing_and_invalid_old_selection(self):
        cases = [
            ({"series_instance_uids": '["1.2.3.1", "1.2.3.2"]'}, "multiple_series_instance_uids"),
            ({"series_instance_uids": "[]"}, "missing_series_instance_uid"),
            ({"series_instance_uids": '["unknown"]'}, "missing_series_instance_uid"),
            ({"flags": '["missing_SeriesInstanceUID"]'}, "missing_series_instance_uid"),
            ({"series_instance_uids": "not JSON"}, "invalid_selection_metadata"),
            ({"series_instance_uids": '"1.2.3.1"'}, "invalid_selection_metadata"),
            ({"file_names": "[]"}, "invalid_selection_metadata"),
            ({"file_names": '"001.dcm"'}, "invalid_selection_metadata"),
            ({"file_names": "bad JSON"}, "invalid_selection_metadata"),
            ({"series_number": "unknown"}, "invalid_selection_metadata"),
            ({"status": "unknown"}, "old_selection_not_selected"),
        ]
        for changes, reason in cases:
            with self.subTest(changes=changes):
                old, manifest = self.pair()
                old.update(changes)
                row = self.audit([old], [manifest])[0]
                self.assertEqual((row["status"], row["reason"]), ("abandoned", reason))

    def test_missing_manifest_and_different_source_directory(self):
        old, manifest = self.pair()
        self.assertEqual(self.audit([old], [])[0]["reason"], "not_in_manifest")
        old["study_directory"] = str(self.root / "different_dicom" / old["study_id"])
        self.assertEqual(self.audit([old], [manifest])[0]["reason"], "study_directory_mismatch")

    def test_path_separators_normalize_but_file_order_must_match(self):
        old, manifest = self.pair()
        old["file_names"] = json.dumps(["acq2\\001.dcm", "acq2\\002.dcm"])
        old["study_directory"] = str(self.study_dir / ".." / self.study_dir.name)
        self.assertEqual(self.audit([old], [manifest])[0]["status"], "reusable")
        old["file_names"] = json.dumps(["acq2/002.dcm", "acq2/001.dcm"])
        self.assertEqual(self.audit([old], [manifest])[0]["reason"], "file_order_mismatch")

    def test_file_comparison_preserves_multiplicity(self):
        old, manifest = self.pair(files=["001.dcm", "001.dcm", "002.dcm"])
        old["file_names"] = json.dumps(["001.dcm", "002.dcm", "002.dcm"])
        self.assertEqual(self.audit([old], [manifest])[0]["reason"], "selected_files_mismatch")

    def test_missing_one_modality_does_not_reject_present_other_modality(self):
        t1, m1 = self.pair()
        t2, m2 = self.pair("T2", "1.2.3.2")
        (self.old_dir / t2["study_id"] / "T2.nii.gz").unlink()
        rows = self.audit([t1, t2], [m1, m2])
        self.assertEqual(rows[0]["status"], "reusable")
        self.assertEqual((rows[1]["status"], rows[1]["reason"]),
                         ("abandoned", "missing_preprocess_file"))

    def test_existing_old_output_without_statistics_is_reported_and_not_guessed(self):
        t1, m1 = self.pair()
        self.pair("T2", "1.2.3.2")
        scratch = self.old_dir / ".preprocess_tmp" / "T1.nii.gz"
        scratch.parent.mkdir()
        scratch.write_bytes(b"temporary scratch file")
        rows = self.audit([t1], [m1])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "reusable")
        orphan = next(row for row in rows if row["modality"] == "T2")
        self.assertEqual((orphan["status"], orphan["reason"]),
                         ("abandoned", "missing_old_statistics"))
        self.assertEqual(orphan["unique_id"], "")
        self.assertEqual(orphan["old_file"], str(self.old_dir / t1["study_id"] / "T2.nii.gz"))

    def test_both_old_modalities_mapping_one_target_are_rejected(self):
        t1, manifest = self.pair()
        t2, _ = self.pair("T2")
        rows = self.audit([t1, t2], [manifest])
        self.assertEqual([row["status"] for row in rows], ["abandoned", "abandoned"])
        self.assertEqual({row["reason"] for row in rows}, {"multiple_old_outputs_for_unique_id"})

    def test_duplicate_old_selection_or_new_id_is_input_error(self):
        old, manifest = self.pair()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.audit([old, dict(old)], [manifest])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.audit([old], [manifest, dict(manifest)])

    def test_manifest_must_have_new_uid_identity(self):
        old, manifest = self.pair()
        for changes in ({"series_instance_uid": ""},
                        {"unique_id": "R01_Study_1_3"},
                        {"unique_id": "R01_Study_1_1.2.3.9_3"}):
            with self.subTest(changes=changes):
                invalid = dict(manifest, **changes)
                with self.assertRaisesRegex(ValueError, "SeriesInstanceUID"):
                    self.audit([old], [invalid])
        del manifest["series_instance_uid"]
        with self.assertRaisesRegex(ValueError, "SeriesInstanceUID"):
            self.audit([old], [manifest])

    def cli(self, output_dir):
        return subprocess.run(
            [sys.executable, str(Path(reuse.__file__)),
             "--old-selected-csv", str(self.root / "old_selected.csv"),
             "--manifest", str(self.root / "manifest.csv"),
             "--old-preprocess-dir", str(self.old_dir), "--output-dir", str(output_dir)],
            capture_output=True, text=True,
        )

    def write_inputs(self, old, manifest):
        write_rows(self.root / "old_selected.csv", old, list(old[0]))
        write_rows(self.root / "manifest.csv", manifest, list(manifest[0]))

    def run_reuse(self, output):
        return reuse.reuse_preprocess(self.root / "old_selected.csv", self.root / "manifest.csv",
                                      self.old_dir, output)

    def test_cli_copies_matching_output_and_preserves_old_files(self):
        t1, m1 = self.pair()
        t2, m2 = self.pair("T2", "1.2.3.2")
        t2["series_instance_uids"] = '["1.2.3.2", "1.2.3.9"]'
        self.write_inputs([t1, t2], [m1, m2])
        before = {p.relative_to(self.old_dir): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.old_dir.rglob("*") if p.is_file()}
        output = self.root / "reports"
        result = self.cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        mapping = read_rows(output / "old_name_change.csv")
        self.assertEqual(len(mapping), 2)
        reusable = [row for row in mapping if row["status"] == "reusable"]
        abandoned = [row for row in mapping if row["status"] == "abandoned"]
        self.assertEqual(reusable[0]["unique_id"], m1["unique_id"])
        self.assertEqual(abandoned[0]["reason"], "multiple_series_instance_uids")
        self.assertEqual(reusable[0]["action"], "copied")
        self.assertEqual({path.name for path in output.glob("*.csv")}, {"old_name_change.csv"})
        self.assertFalse(list(output.glob("*.txt")))
        copied = output / f"{m1['unique_id']}.nii.gz"
        self.assertEqual(copied.read_bytes(), before[Path(t1["study_id"]) / "T1.nii.gz"][0])
        self.assertFalse((output / "pre_process.csv").exists())
        after = {p.relative_to(self.old_dir): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.old_dir.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        copied_time = copied.stat().st_mtime_ns
        completion_csv = output / "pre_process.csv"
        completion_csv.write_bytes(b"unique_id\nEarlier_Completed_1.2.3.4_1\n")
        completion_before = (completion_csv.read_bytes(), completion_csv.stat().st_mtime_ns)
        repeated = self.cli(output)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(read_rows(output / "old_name_change.csv")[0]["action"], "skipped_existing")
        self.assertEqual(copied.stat().st_mtime_ns, copied_time)
        self.assertEqual((completion_csv.read_bytes(), completion_csv.stat().st_mtime_ns), completion_before)

    def test_cli_rejects_output_directory_overlapping_old_preprocess_root(self):
        old, manifest = self.pair()
        write_rows(self.root / "old_selected.csv", [old], list(old))
        write_rows(self.root / "manifest.csv", [manifest], list(manifest))
        output = self.old_dir / "new_reports"
        result = self.cli(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        for output in (self.old_dir, self.root):
            with self.subTest(output=output):
                self.assertNotEqual(self.cli(output).returncode, 0)

    def test_existing_output_and_completion_csv_are_never_overwritten(self):
        old, manifest = self.pair()
        self.write_inputs([old], [manifest])
        output = self.root / "processed"
        output.mkdir()
        existing = output / f"{manifest['unique_id']}.nii.gz"
        existing.write_bytes(b"previously completed output")
        previous_id = "Other_Study_1.2.3.8_5"
        (output / f"{previous_id}.nii.gz").write_bytes(b"independently processed series")
        completion_csv = output / "pre_process.csv"
        write_rows(completion_csv, [{"unique_id": previous_id}], ["unique_id"])
        completion_before = (completion_csv.read_bytes(), completion_csv.stat().st_mtime_ns)
        reports = self.run_reuse(output)
        self.assertEqual(reports[0]["action"], "skipped_existing")
        self.assertEqual(existing.read_bytes(), b"previously completed output")
        self.assertEqual((completion_csv.read_bytes(), completion_csv.stat().st_mtime_ns), completion_before)

    def test_copy_failure_is_reported_and_other_modality_still_copies(self):
        t1, m1 = self.pair()
        t2, m2 = self.pair("T2", "1.2.3.2")
        self.write_inputs([t1, t2], [m1, m2])
        output = self.root / "processed"
        copy_file = reuse.shutil.copyfile

        def fail_t1(source, destination, *args, **kwargs):
            if Path(source).name == "T1.nii.gz":
                Path(destination).write_bytes(b"incomplete temporary data")
                raise OSError("synthetic copy failure")
            return copy_file(source, destination, *args, **kwargs)

        with patch.object(reuse.shutil, "copyfile", side_effect=fail_t1):
            rows = self.run_reuse(output)
        self.assertEqual((rows[0]["status"], rows[0]["reason"], rows[0]["action"]),
                         ("abandoned", "copy_failed", "copy_failed"))
        self.assertIn("synthetic copy failure", rows[0]["error"])
        self.assertEqual(rows[1]["action"], "copied")
        self.assertFalse((output / f"{m1['unique_id']}.nii.gz").exists())
        self.assertFalse((output / "pre_process.csv").exists())
        self.assertEqual(read_rows(output / "old_name_change.csv")[1]["action"], "copied")
        self.assertFalse(list((output / ".reuse_tmp").rglob("*.nii.gz")))

    def test_interruption_records_copies_and_next_run_resumes_without_completion_csv(self):
        t1, m1 = self.pair()
        t2, m2 = self.pair("T2", "1.2.3.2")
        self.write_inputs([t1, t2], [m1, m2])
        output = self.root / "processed"
        copy_file = reuse.shutil.copyfile

        def interrupt_t2(source, destination, *args, **kwargs):
            if Path(source).name == "T2.nii.gz":
                raise KeyboardInterrupt()
            return copy_file(source, destination, *args, **kwargs)

        with patch.object(reuse.shutil, "copyfile", side_effect=interrupt_t2):
            with self.assertRaises(KeyboardInterrupt):
                self.run_reuse(output)
        self.assertFalse((output / "pre_process.csv").exists())
        self.assertEqual(len(read_rows(output / "old_name_change.csv")), 2)
        self.assertFalse((output / f"{m2['unique_id']}.nii.gz").exists())
        t1_time = (output / f"{m1['unique_id']}.nii.gz").stat().st_mtime_ns
        rows = self.run_reuse(output)
        self.assertEqual([row["action"] for row in rows], ["skipped_existing", "copied"])
        self.assertEqual((output / f"{m1['unique_id']}.nii.gz").stat().st_mtime_ns, t1_time)
        self.assertFalse((output / "pre_process.csv").exists())
        self.assertTrue((output / f"{m2['unique_id']}.nii.gz").is_file())

    def test_preprocess_generates_completion_csv_separately_and_skips_copied_images(self):
        t1, m1 = self.pair()
        t2, m2 = self.pair("T2", "1.2.3.2")
        self.write_inputs([t1, t2], [m1, m2])
        output = self.root / "processed"
        self.run_reuse(output)
        self.assertFalse((output / "pre_process.csv").exists())
        copied_before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                         for path in output.glob("*.nii.gz")}
        converted_csv = self.root / "converted.csv"
        write_rows(converted_csv, [{"unique_id": m1["unique_id"]}, {"unique_id": m2["unique_id"]}],
                   ["unique_id"])
        with patch.object(pre_process.subprocess, "run") as imaging:
            completed = pre_process.preprocess_csv(converted_csv, self.root / "absent_raw", output)
        imaging.assert_not_called()
        self.assertEqual(set(completed), {m1["unique_id"], m2["unique_id"]})
        self.assertEqual({row["unique_id"] for row in read_rows(output / "pre_process.csv")},
                         {m1["unique_id"], m2["unique_id"]})
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in output.glob("*.nii.gz")}, copied_before)


if __name__ == "__main__":
    unittest.main()
