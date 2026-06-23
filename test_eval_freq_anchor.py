import csv
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def save_test_image(path, size):
    y = np.linspace(0.0, 1.0, size[1], dtype=np.float32)
    x = np.linspace(0.0, 1.0, size[0], dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    img = np.stack([xx, yy, 0.5 * xx + 0.5 * yy], axis=-1)
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(path)


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
                "angle_error_box_by_alpha.png",
                "anchor_sync_vs_oracle_by_alpha.png",
                "failure_rate_vs_alpha.png",
                "example_grid.png",
            ]:
                self.assertTrue((outdir / name).exists(), name)

            summary_path = outdir / "summary.csv"
            self.assertTrue(summary_path.exists())
            with summary_path.open(newline="") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertEqual(len(summary_rows), 1)
            self.assertIn("mean_angle_error", summary_rows[0])
            self.assertIn("mean_anchor_sync_score", summary_rows[0])

    def test_image_dir_takes_priority_and_no_resize_keeps_input_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "single.png"
            image_dir = root / "images"
            outdir = root / "out"
            image_dir.mkdir()
            save_test_image(image_path, (32, 32))
            save_test_image(image_dir / "wide.png", (40, 24))
            save_test_image(image_dir / "square.jpg", (36, 36))

            subprocess.run(
                [
                    sys.executable,
                    "eval_freq_anchor.py",
                    "--image",
                    str(image_path),
                    "--image-dir",
                    str(image_dir),
                    "--outdir",
                    str(outdir),
                    "--no-resize",
                    "--alphas",
                    "0.003",
                    "--angles",
                    "5",
                    "--coarse-step",
                    "5",
                    "--fine-step",
                    "1",
                    "--example-alpha",
                    "0.003",
                    "--example-theta",
                    "5",
                    "--no-show-progress",
                ],
                check=True,
            )

            with (outdir / "freq_anchor_results.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({row["image_id"] for row in rows}, {"square", "wide"})
            self.assertEqual(len(rows), 2)
            self.assertTrue((outdir / "x_w_wide.png").exists())
            self.assertTrue((outdir / "x_w_square.png").exists())
            with Image.open(outdir / "x_w_wide.png") as img:
                self.assertEqual(img.size, (40, 24))
            with Image.open(outdir / "x_w_square.png") as img:
                self.assertEqual(img.size, (36, 36))

            with (outdir / "summary.csv").open(newline="") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["alpha"], "0.003")

    def test_enable_vae_metrics_uses_encoder_hook_without_zt_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hook_path = root / "vae_hook.py"
            hook_path.write_text(
                "import numpy as np\n"
                "def encode(img):\n"
                "    arr = np.asarray(img, dtype=np.float32)\n"
                "    return arr.mean(axis=(0, 1))\n"
            )
            outdir = root / "out"
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"

            subprocess.run(
                [
                    sys.executable,
                    "eval_freq_anchor.py",
                    "--outdir",
                    str(outdir),
                    "--size",
                    "64",
                    "--alphas",
                    "0.003",
                    "--angles",
                    "5",
                    "--coarse-step",
                    "5",
                    "--fine-step",
                    "1",
                    "--enable-vae-metrics",
                    "--vae-encoder",
                    "vae_hook:encode",
                    "--no-show-progress",
                ],
                check=True,
                env=env,
            )

            with (outdir / "freq_anchor_results.csv").open(newline="") as f:
                row = next(csv.DictReader(f))
            self.assertFalse(math.isnan(float(row["vae_mse_sync"])))
            self.assertFalse(math.isnan(float(row["vae_cos_sync"])))
            self.assertTrue(math.isnan(float(row["zt_mse_sync"])))
            self.assertTrue(math.isnan(float(row["zt_cos_sync"])))

            with (outdir / "summary.csv").open(newline="") as f:
                summary_row = next(csv.DictReader(f))
            self.assertFalse(math.isnan(float(summary_row["mean_vae_mse_sync"])))
            self.assertFalse(math.isnan(float(summary_row["mean_vae_cos_sync"])))


if __name__ == "__main__":
    unittest.main()
