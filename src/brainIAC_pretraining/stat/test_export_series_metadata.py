"""Direct-DICOM metadata export regressions; synthetic headers need no pixels."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from export_series_metadata import META_COLUMNS, MULTIVALUE_COLUMNS, inspect_study


class MetadataExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "dicom"
        self.root.mkdir()
        self.study_id = "STUDY_1"
        self.uid, self.study_uid = generate_uid(), generate_uid()
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def write(self, *, uid=None, number=3, acquisition=1, instance=1, z=0, **values):
        folder = self.root / self.study_id / "nested"
        folder.mkdir(parents=True, exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        defaults = dict(SOPClassUID=meta.MediaStorageSOPClassUID,
                        SOPInstanceUID=meta.MediaStorageSOPInstanceUID,
                        StudyInstanceUID=self.study_uid, SeriesInstanceUID=uid or self.uid,
                        SeriesNumber=number, AcquisitionNumber=acquisition,
                        InstanceNumber=instance, Modality="MR", SeriesDescription="AX T2",
                        ImageType=["ORIGINAL", "PRIMARY"], Rows=8, Columns=8,
                        PixelSpacing=[1, 1], SliceThickness="5", SpacingBetweenSlices="6",
                        ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, z])
        defaults.update(values)
        for field, value in defaults.items():
            if value is not None:
                setattr(ds, field, value)
        self.counter += 1
        path = folder / f"{self.counter}.dcm"
        ds.save_as(path, enforce_file_format=True)
        return path

    def inspect(self):
        return inspect_study(self.study_id, [self.root])

    def cli(self, output, study_id=None):
        labels, folders = self.base / "labels.csv", self.base / "folders.txt"
        labels.write_text(f"Study_ID\n{study_id or self.study_id}\n", encoding="utf-8")
        folders.write_text("dicom\n", encoding="utf-8")
        return subprocess.run([sys.executable, str(Path(__file__).with_name("export_series_metadata.py")),
                               "--csv", str(labels), "--folder-list", str(folders),
                               "--output-dir", str(output)], text=True, capture_output=True)

    def test_numeric_maximum_is_per_uid_and_number_and_keeps_repeated_planes(self):
        old = self.write(acquisition=9, SeriesDescription="OLD", SliceThickness="99")
        first = self.write(acquisition=10, instance=1)
        second = self.write(acquisition=10, instance=2)
        another_uid = generate_uid()
        self.write(uid=another_uid, acquisition=2, z=1)
        self.write(number=4, acquisition=0, z=2)
        self.write(number=5, Modality="CT")
        rows, multi, _ = self.inspect()
        self.assertEqual(len(rows), 3)
        row = next(row for row in rows if row["unique_id"] == f"{self.study_id}_{self.uid}_3")
        self.assertEqual(row["AcquisitionNumber"], 10)
        self.assertEqual(row["SeriesDescription"], "AX T2")
        self.assertEqual(row["SliceThickness"], 5)
        self.assertEqual(row["coverage_mm"], 0)
        self.assertEqual(set(row["dicom_file_list"]), {first.stem, second.stem})
        self.assertNotIn(old.stem, row["dicom_file_list"])
        self.assertEqual(multi, [])

    def test_equivalent_numeric_values_and_complete_image_type_collapse(self):
        self.write(SliceThickness="5", SpacingBetweenSlices="6")
        self.write(instance=2, z=6, SliceThickness="5.000", SpacingBetweenSlices="6.0")
        rows, multi, _ = self.inspect()
        self.assertEqual(rows[0]["SliceThickness"], 5.0)
        self.assertEqual(rows[0]["SpacingBetweenSlices"], 6.0)
        self.assertEqual(rows[0]["ImageType"], "ORIGINAL\\PRIMARY")
        self.assertEqual(rows[0]["coverage_mm"], 6)
        self.assertEqual(multi, [])

    def test_all_four_multivalue_fields_are_listed_once(self):
        self.write(SeriesDescription=" AX T2 ")
        self.write(instance=2, z=6, SeriesDescription="AX T2 REPEAT",
                   ImageType=["DERIVED", "SECONDARY"], SliceThickness="4", SpacingBetweenSlices="7")
        rows, multi, _ = self.inspect()
        row = rows[0]
        self.assertEqual(set(row["SeriesDescription"]), {"AX T2", "AX T2 REPEAT"})
        self.assertEqual(set(row["ImageType"]), {"ORIGINAL\\PRIMARY", "DERIVED\\SECONDARY"})
        self.assertEqual(set(row["SliceThickness"]), {4.0, 5.0})
        self.assertEqual(set(row["SpacingBetweenSlices"]), {6.0, 7.0})
        self.assertEqual(len(multi), 1)
        self.assertEqual(multi[0]["unique_id"], row["unique_id"])
        self.assertEqual(set(multi[0]["multiple_fields"]),
                         {"SeriesDescription", "ImageType", "SliceThickness", "SpacingBetweenSlices"})

    def test_missing_acquisition_keeps_all_only_when_no_known_number_exists(self):
        self.write(acquisition=None)
        self.write(acquisition=None, instance=2, z=4)
        malformed = self.write(acquisition=8, instance=3, z=8)
        tag = b"\x20\x00\x12\x00IS\x02\x008 "
        content = malformed.read_bytes()
        self.assertIn(tag, content)
        malformed.write_bytes(content.replace(tag, tag[:-2] + b"X ", 1))
        rows, _, issues = self.inspect()
        self.assertEqual(rows[0]["AcquisitionNumber"], "unknown")
        self.assertEqual(len(rows[0]["dicom_file_list"]), 3)
        self.assertTrue(any("AcquisitionNumber" in issue["reason"] for issue in issues))
        chosen = self.write(acquisition=0, z=1)
        rows, _, _ = self.inspect()
        self.assertEqual(rows[0]["AcquisitionNumber"], 0)
        self.assertEqual(rows[0]["dicom_file_list"], [chosen.stem])

    def test_file_names_omit_paths_and_extensions_but_preserve_duplicates(self):
        first = self.write()
        second = self.write(instance=2, z=2)
        extensionless = self.root / self.study_id / first.stem
        second.rename(extensionless)
        rows, _, _ = self.inspect()
        self.assertEqual(rows[0]["dicom_file_list"], [first.stem, first.stem])

    def test_partial_missing_fields_and_unknown_geometry_still_export_row(self):
        self.write(SeriesDescription=None, ImageType=None, SliceThickness=None,
                   SpacingBetweenSlices=None, ImagePositionPatient=None)
        self.write(instance=2, z=8, SpacingBetweenSlices=None)
        rows, multi, issues = self.inspect()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["SeriesDescription"], "AX T2")
        self.assertEqual(row["ImageType"], "ORIGINAL\\PRIMARY")
        self.assertEqual(row["SliceThickness"], 5)
        self.assertEqual(row["SpacingBetweenSlices"], "unknown")
        self.assertTrue(math.isnan(row["coverage_mm"]))
        self.assertEqual(multi, [])
        for field in ("SeriesDescription", "ImageType", "SliceThickness", "SpacingBetweenSlices"):
            self.assertTrue(any(field in issue["reason"] for issue in issues), field)

    def test_enhanced_shared_and_per_frame_pixel_measures(self):
        shared = Dataset()
        shared.PixelMeasuresSequence = [Dataset()]
        shared.PixelMeasuresSequence[0].PixelSpacing = [1, 1]
        shared.PixelMeasuresSequence[0].SliceThickness = 4
        shared.PixelMeasuresSequence[0].SpacingBetweenSlices = 6
        frames = []
        for index, z in enumerate((0, 6)):
            frame = Dataset()
            frame.PlanePositionSequence = [Dataset()]
            frame.PlanePositionSequence[0].ImagePositionPatient = [0, 0, z]
            if index:
                frame.PixelMeasuresSequence = [Dataset()]
                frame.PixelMeasuresSequence[0].SliceThickness = 5
                frame.PixelMeasuresSequence[0].SpacingBetweenSlices = 7
            frames.append(frame)
        self.write(NumberOfFrames=2, SliceThickness=None, SpacingBetweenSlices=None,
                   SharedFunctionalGroupsSequence=[shared], PerFrameFunctionalGroupsSequence=frames)
        rows, multi, _ = self.inspect()
        self.assertEqual(set(rows[0]["SliceThickness"]), {4, 5})
        self.assertEqual(set(rows[0]["SpacingBetweenSlices"]), {6, 7})
        self.assertEqual(rows[0]["coverage_mm"], 6)
        self.assertEqual(set(multi[0]["multiple_fields"]), {"SliceThickness", "SpacingBetweenSlices"})

    def test_cli_serializes_json_and_rejects_existing_output(self):
        self.write()
        self.write(instance=2, z=2, SliceThickness=4)
        output = self.base / "export"
        result = self.cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        table = pd.read_csv(output / "series_meta.csv", keep_default_na=False)
        self.assertEqual(table.columns.tolist(), META_COLUMNS)
        self.assertEqual(set(json.loads(table.loc[0, "SliceThickness"])), {4, 5})
        self.assertEqual(len(json.loads(table.loc[0, "dicom_file_list"])), 2)
        multi = pd.read_csv(output / "series_multivalue.csv", keep_default_na=False)
        self.assertEqual(multi.columns.tolist(), MULTIVALUE_COLUMNS)
        self.assertEqual(json.loads(multi.loc[0, "multiple_fields"]), ["SliceThickness"])
        before = (output / "series_meta.csv").read_bytes()
        self.assertNotEqual(self.cli(output).returncode, 0)
        self.assertEqual((output / "series_meta.csv").read_bytes(), before)

    def test_empty_export_has_fixed_headers_and_reports_missing_study(self):
        output = self.base / "empty"
        result = self.cli(output, "ABSENT")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, columns in (("series_meta.csv", META_COLUMNS),
                              ("series_multivalue.csv", MULTIVALUE_COLUMNS)):
            table = pd.read_csv(output / name)
            self.assertEqual(table.columns.tolist(), columns)
            self.assertTrue(table.empty)
        issues = pd.read_csv(output / "scan_issues.csv")
        self.assertEqual(issues.columns.tolist(), ["study_id", "unique_id", "file", "reason"])
        self.assertEqual(issues.loc[0, "study_id"], "ABSENT")


if __name__ == "__main__":
    unittest.main()
