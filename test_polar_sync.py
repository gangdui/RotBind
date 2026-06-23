import unittest

import numpy as np

from freq_anchor import embed_anchor_rgb, rotate_image_keep_size
from polar_sync import (
    circular_corr_angle,
    detect_rotation_angle_polar,
    fft_polar_magnitude,
    make_angular_code,
    make_polar_anchor_template,
    make_radial_window_freq,
    resolve_180_ambiguity,
)


class PolarSyncTest(unittest.TestCase):
    def test_angular_code_is_reproducible_and_normalized(self):
        a = make_angular_code(180, key=5)
        b = make_angular_code(180, key=5)

        self.assertEqual(a.shape, (180,))
        self.assertEqual(a.dtype, np.float32)
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(a.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(a.std()), 1.0, places=5)

    def test_radial_window_is_soft_mid_band(self):
        win = make_radial_window_freq(64, 64, rmin=0.12, rmax=0.28, soft=True)

        self.assertEqual(win.shape, (64, 64))
        self.assertEqual(win.dtype, np.float32)
        self.assertGreater(float(win.max()), 0.9)
        self.assertAlmostEqual(float(win[0, 0]), 0.0, places=5)

    def test_circular_corr_angle_finds_shift_without_rotation_loop(self):
        template = np.tile(make_angular_code(180, key=7), (8, 1))
        feature = np.roll(template, 23, axis=1)

        angle_bin, score, curve = circular_corr_angle(feature, template)

        self.assertEqual(angle_bin, 23)
        self.assertGreater(score, float(np.mean(curve)) + 1.0)

    def test_polar_anchor_detects_rotation_mod_180(self):
        img = np.full((96, 96, 3), 0.5, dtype=np.float32)
        delta, polar_template, _meta = make_polar_anchor_template(
            96, 96, key=3, num_angles=180, beta=0.9
        )
        sync = embed_anchor_rgb(img, delta, alpha=0.08)
        attacked = rotate_image_keep_size(sync, 27.0)

        theta_hat, score, _curve, extra = detect_rotation_angle_polar(
            attacked,
            polar_template,
            delta=delta,
            num_r=32,
            num_angles=180,
            resolve_ambiguity=True,
        )

        err = abs(((theta_hat - 27.0 + 180.0) % 360.0) - 180.0)
        self.assertLess(err, 5.0)
        self.assertGreater(score, 0.0)
        self.assertTrue(extra["ambiguity_resolved"])

    def test_fft_polar_magnitude_shape(self):
        img = np.full((48, 64, 3), 0.5, dtype=np.float32)
        polar = fft_polar_magnitude(img, num_r=16, num_angles=90)

        self.assertEqual(polar.shape, (16, 90))
        self.assertEqual(polar.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
