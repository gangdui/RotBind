import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvalPolarSyncSmokeTest(unittest.TestCase):
    def test_eval_polar_sync_writes_v1_and_v0_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "out"
            subprocess.run(
                [
                    sys.executable,
                    "eval_polar_sync.py",
                    "--outdir",
                    str(outdir),
                    "--size",
                    "64",
                    "--alphas",
                    "0.02",
                    "--angles",
                    "15",
                    "--num-r",
                    "16",
                    "--num-angles",
                    "90",
                    "--compare-v0",
                    "--v0-coarse-step",
                    "15",
                    "--v0-fine-step",
                    "5",
                    "--no-show-progress",
                ],
                check=True,
            )

            csv_path = outdir / "polar_sync_results.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({row["method"] for row in rows}, {"v0", "v1"})
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertIn("runtime_ms", row)
                self.assertIn("angle_error", row)

            with (outdir / "summary.csv").open(newline="") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertEqual({row["method"] for row in summary_rows}, {"v0", "v1"})

            for name in [
                "angle_error_vs_theta.png",
                "angle_error_box_by_alpha.png",
                "runtime_by_method.png",
                "failure_rate_vs_alpha.png",
                "anchor_sync_vs_oracle_by_alpha.png",
                "example_grid.png",
            ]:
                self.assertTrue((outdir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
