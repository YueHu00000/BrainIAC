"""Run the real CLIs and all thirty fitted tasks on synthetic study records."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CBI_WMD_predict.common import FOLDS, MODALITIES, read_csv, write_csv


class CLIPipelineTests(unittest.TestCase):
    def test_all_modalities_targets_and_site_schemes(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embeddings = root / "embeddings"
            embeddings.mkdir()
            series, quality, labels = [], [], []
            sites = sorted({site for fold in FOLDS.values() for group in fold.values() for site in group})
            for site in sites:
                for index in range(16):
                    study = f"S_{site}_{index:02d}"
                    cbi, wmd = index % 2, (index // 2) % 2
                    labels.append(dict(Study_ID=study, Patient_ID=f"P_{site}_{index}", SiteID=site,
                                       idCBI=cbi ^ int(site >= 6 and index < 8),
                                       idWMD=wmd ^ int(site >= 6 and index < 8)))
                    records = {}
                    for number, modality in enumerate(("T1", "T2"), 1):
                        uid = f"1.2.840.{site}.{index}.{number}"
                        full_id = f"{study}_{uid}_{number}"
                        series.append(dict(unique_id=full_id, study_id=study, series_instance_uid=uid,
                                           series_number=number, acquisition_number=1,
                                           series_description=modality))
                        quality.append(dict(unique_id=full_id, frame_count=23, coverage_mm=154))
                        # Perfect signal at train-only sites, balanced label noise at
                        # held-out sites: exercises the strict train > val gate.
                        vector = np.zeros(768, dtype=np.float32)
                        vector[:2] = (2 * cbi - 1, 2 * wmd - 1)
                        record = np.zeros((), dtype=[("series_description", "U8"),
                                                    ("final_embed", "f4", (768,)),
                                                    ("block_embed", "f4", (4, 768))])
                        record["series_description"], record["final_embed"] = modality, vector
                        record["block_embed"] = np.stack([vector * scale for scale in (1, 2, 3, 4)])
                        records[full_id] = record
                    np.savez_compressed(embeddings / f"{study}.npz", **records)
            write_csv(root / "series.csv", series, list(series[0]))
            write_csv(root / "quality.csv", quality, list(quality[0]))
            write_csv(root / "labels.csv", labels, list(labels[0]))
            write_csv(root / "final.csv", series, ["unique_id"])
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

            def execute(script, arguments):
                completed = subprocess.run([sys.executable, str(package / script), *arguments],
                                           cwd=root, env=environment, capture_output=True, text=True,
                                           encoding="utf-8", timeout=180)
                self.assertEqual(completed.returncode, 0, completed.stdout[-5000:] + completed.stderr)

            manifests = root / "manifests"
            execute("manifest.py", ["--manifest", str(root / "series.csv"),
                                    "--final-csv", str(root / "final.csv"),
                                    "--selected-csv", str(root / "series.csv"),
                                    "--quality-csv", str(root / "quality.csv"),
                                    "--labels-csv", str(root / "labels.csv"),
                                    "--patient-id-column", "Patient_ID",
                                    "--embedding-dir", str(embeddings), "--output-dir", str(manifests)])
            for modality, block in zip(MODALITIES, (None, 6, 12)):
                isolated = root / f"isolated_{modality}"
                isolated.mkdir()
                manifest = isolated / "input.csv"
                shutil.copyfile(manifests / f"manifest_{modality.lower()}.csv", manifest)
                output = root / f"results_{modality}"
                field = "block_embed" if block is not None else "final_embed"
                selection = ["--embedding-field", field] + (["--block", str(block)] if block else [])
                execute("train.py", ["--manifest", str(manifest), "--output-dir", str(output),
                                     "--device", "cpu", "--epochs", "6", "--batch-size", "8",
                                     "--learning-rates", "0.1", "--penalties", "L2", "--alphas", "0", "0.01",
                                     *selection])
                results = read_csv(output / "results.csv")
                self.assertEqual(len(results), 10)
                self.assertTrue(all(row["status"] == "ok" for row in results), results)
                self.assertEqual(len(read_csv(output / "summary.csv")), 2)
                self.assertFalse((output / "paired_differences.csv").exists())
                self.assertEqual(list(isolated.iterdir()), [manifest])
                for target in ("CBI", "WMD"):
                    for fold in FOLDS:
                        folder = output / modality / target / f"fold_{fold}"
                        predictions = read_csv(folder / "predictions.csv")
                        self.assertTrue(predictions)
                        self.assertEqual(len(read_csv(folder / "history.csv")), 12)
                        self.assertEqual(len(read_csv(folder / "trials.csv")), 2)
                        checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
                        width = 1536 if modality == "T1T2" else 768
                        self.assertEqual(checkpoint["metadata"]["in_features"], width)
                        self.assertEqual(checkpoint["metadata"]["embedding_field"], field)
                        self.assertEqual(checkpoint["metadata"]["block"], block)
                        self.assertEqual(predictions[0]["embedding_field"], field)
                        self.assertEqual(predictions[0]["block"], "" if block is None else str(block))
                        self.assertEqual(tuple(checkpoint["state_dict"]["weight"].shape), (1, width))


if __name__ == "__main__":
    unittest.main()
