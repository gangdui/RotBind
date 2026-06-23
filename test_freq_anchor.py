import unittest

import numpy as np
from scipy import ndimage

from freq_anchor import (
    bandpass_y_channel,
    detect_rotation_angle,
    embed_anchor_rgb,
    make_anchor_template,
    ncc,
    remove_anchor_rgb,
    rotate_image_keep_size,
)


def make_textured_image(size, key=0):
    rng = np.random.default_rng(key)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    low = 0.45 + 0.20 * np.sin(2 * np.pi * xx) + 0.12 * np.cos(2 * np.pi * yy)
    texture = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32)
    texture = ndimage.gaussian_filter(texture, sigma=1.1)
    texture = texture / max(float(texture.std()), 1e-8)
    y_chan = np.clip(low + 0.045 * texture, 0.0, 1.0)
    return np.stack(
        [
            np.clip(y_chan * 1.05, 0.0, 1.0),
            y_chan,
            np.clip(0.85 * y_chan + 0.08 * xx, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)


class FreqAnchorTest(unittest.TestCase):
    def test_template_is_reproducible_and_band_limited(self):
        delta_a, mask_a = make_anchor_template(64, 80, key=123, circular_window=False)
        delta_b, mask_b = make_anchor_template(64, 80, key=123, circular_window=False)

        self.assertEqual(delta_a.shape, (64, 80))
        self.assertEqual(delta_a.dtype, np.float32)
        self.assertTrue(np.array_equal(mask_a, mask_b))
        np.testing.assert_allclose(delta_a, delta_b)
        self.assertAlmostEqual(float(delta_a.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(delta_a.std()), 1.0, places=5)
        self.assertGreater(mask_a.sum(), 0)
        self.assertLess(mask_a.sum(), mask_a.size)

    def test_circular_windowed_template_is_renormalized(self):
        delta, _mask = make_anchor_template(65, 80, key=321, circular_window=True)

        self.assertAlmostEqual(float(delta.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(delta.std()), 1.0, places=5)

    def test_embed_and_remove_anchor_round_trip_on_gray_image(self):
        img = np.full((48, 48, 3), 0.5, dtype=np.float32)
        delta, mask = make_anchor_template(48, 48, key=4, circular_window=False)

        sync = embed_anchor_rgb(img, delta, alpha=0.01)
        clean, alpha_hat = remove_anchor_rgb(sync, delta, mask)

        self.assertEqual(sync.shape, img.shape)
        self.assertTrue(np.all(sync >= 0.0))
        self.assertTrue(np.all(sync <= 1.0))
        self.assertAlmostEqual(float(alpha_hat), 0.01, places=3)
        self.assertLess(float(np.mean((clean - img) ** 2)), 1e-5)

    def test_remove_anchor_alpha_estimate_ignores_luminance_dc_offset(self):
        base = np.linspace(-0.4, 0.6, 40 * 40, dtype=np.float32).reshape(40, 40)
        delta = base + 0.75
        alpha = 0.02
        y = np.clip(0.42 + alpha * delta, 0.0, 1.0).astype(np.float32)
        img = np.repeat(y[..., None], 3, axis=-1)

        _clean, alpha_hat = remove_anchor_rgb(img, delta)

        self.assertAlmostEqual(float(alpha_hat), alpha, places=3)

    def test_bandpass_and_ncc_find_embedded_anchor(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        delta, mask = make_anchor_template(64, 64, key=7, circular_window=False)
        sync = embed_anchor_rgb(img, delta, alpha=0.01)

        band = bandpass_y_channel(sync, mask)

        self.assertGreater(ncc(band, delta), 0.95)

    def test_detect_rotation_angle_uses_attack_sign_convention(self):
        img = np.full((96, 96, 3), 0.5, dtype=np.float32)
        delta, mask = make_anchor_template(96, 96, key=9, circular_window=True)
        sync = embed_anchor_rgb(img, delta, alpha=0.04)
        attacked = rotate_image_keep_size(sync, 17.0)

        theta_hat, score, curve = detect_rotation_angle(
            attacked,
            delta,
            mask,
            coarse_step=3.0,
            fine_step=0.5,
            angle_range=(-30.0, 30.0),
        )

        self.assertLess(abs(theta_hat - 17.0), 1.5)
        self.assertGreater(score, 0.4)
        self.assertIn("coarse", curve)
        self.assertIn("fine", curve)

    def test_detect_rotation_angle_with_weak_anchor_on_textured_image(self):
        img = make_textured_image(128, key=11)
        delta, mask = make_anchor_template(128, 128, key=13, circular_window=True)
        sync = embed_anchor_rgb(img, delta, alpha=0.008)
        attacked = rotate_image_keep_size(sync, 23.0)

        theta_hat, score, _curve = detect_rotation_angle(
            attacked,
            delta,
            mask,
            coarse_step=2.0,
            fine_step=0.5,
            angle_range=(-40.0, 40.0),
        )

        self.assertLess(abs(theta_hat - 23.0), 2.0)
        self.assertGreater(score, 0.08)


if __name__ == "__main__":
    unittest.main()
