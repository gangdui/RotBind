import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvalFreqAnchorSmokeTest(unittest.TestCase):
    def test_eval_script_writes_csv_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "out"
            subprocess.run(
                [
                    sys.executable,
                    "eval_freq_anchor.py",
                    "--outdir",
                    str(outdir),
                    "--size",
                    "64",
                    "--alphas",
                    "0.02",
                    "--angles",
                    "15",
                    "--coarse-step",
                    "5",
                    "--fine-step",
                    "1",
                    "--no-show-progress",
                ],
                check=True,
            )

            csv_path = outdir / "freq_anchor_results.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertLess(float(rows[0]["angle_error"]), 3.0)
            for detector_field in [
                "base_score",
                "oracle_score",
                "anchorsync_score",
                "anchorsync_remove_score",
            ]:
                self.assertTrue(math.isnan(float(rows[0][detector_field])))
            for anchor_field in [
                "anchor_base_score",
                "anchor_oracle_score",
                "anchor_sync_score",
                "anchor_remove_score",
            ]:
                self.assertIn(anchor_field, rows[0])
                self.assertFalse(math.isnan(float(rows[0][anchor_field])))

            for name in [
                "angle_error_vs_theta.png",
                "surrogate_anchor_score_vs_theta.png",
                "quality_vs_alpha.png",
                "vae_mse_vs_alpha.png",
                "zt_mse_vs_alpha.png",
                "example_grid.png",
            ]:
                self.assertTrue((outdir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
