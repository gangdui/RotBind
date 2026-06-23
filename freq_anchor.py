"""Pixel-frequency rotation synchronization anchors for RGB images.

The module is intentionally independent of diffusion/VAE code. It operates on
float32 RGB images in [0, 1], estimates rotation before encoder/inversion
stages, and returns plain NumPy arrays for easy integration into experiments.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


EPS = 1e-8


def _normalize_zero_mean_unit_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)
    std = float(np.std(x))
    if std < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return (x / std).astype(np.float32)


def _rgb_to_ycbcr(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(img_rgb, dtype=np.float32)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb_shift = cb - 0.5
    cr_shift = cr - 0.5
    r = y + 1.402 * cr_shift
    g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
    b = y + 1.772 * cb_shift
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def _validate_rgb(img_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(img_rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")
    return img


def make_anchor_template(
    H: int,
    W: int,
    key: int = 0,
    rmin: float = 0.12,
    rmax: float = 0.28,
    circular_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a reproducible weak mid-frequency spatial residual.

    Frequencies are measured with ``np.fft.fftfreq`` in cycles/pixel. The radial
    mask keeps components whose normalized radius satisfies ``rmin <= r <= rmax``.
    """

    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")
    if not 0 <= rmin < rmax:
        raise ValueError("Expected 0 <= rmin < rmax")

    rng = np.random.default_rng(key)
    noise = rng.standard_normal((H, W)).astype(np.float32)

    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    frequency_mask = (radius >= rmin) & (radius <= rmax)

    filtered_fft = np.fft.fft2(noise) * frequency_mask
    delta = np.fft.ifft2(filtered_fft).real.astype(np.float32)
    delta = _normalize_zero_mean_unit_std(delta)

    if circular_window:
        y = np.arange(H, dtype=np.float32) - (H - 1) / 2.0
        x = np.arange(W, dtype=np.float32) - (W - 1) / 2.0
        yy_sp, xx_sp = np.meshgrid(y, x, indexing="ij")
        crop_radius = min(H, W) / 2.0
        window = ((xx_sp * xx_sp + yy_sp * yy_sp) <= crop_radius * crop_radius).astype(
            np.float32
        )
        delta = (delta * window).astype(np.float32)

    return delta.astype(np.float32), frequency_mask


def embed_anchor_rgb(img_rgb: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    """Embed ``alpha * delta`` into the RGB image luminance channel."""

    img = _validate_rgb(img_rgb)
    delta_arr = np.asarray(delta, dtype=np.float32)
    if delta_arr.shape != img.shape[:2]:
        raise ValueError(f"Delta shape {delta_arr.shape} does not match image {img.shape[:2]}")

    y, cb, cr = _rgb_to_ycbcr(img)
    y_sync = np.clip(y + float(alpha) * delta_arr, 0.0, 1.0).astype(np.float32)
    return _ycbcr_to_rgb(y_sync, cb, cr)


def bandpass_y_channel(img_rgb: np.ndarray, frequency_mask: np.ndarray) -> np.ndarray:
    """Return normalized mid-frequency luminance content."""

    img = _validate_rgb(img_rgb)
    mask = np.asarray(frequency_mask)
    if mask.shape != img.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image {img.shape[:2]}")

    y, _, _ = _rgb_to_ycbcr(img)
    bandpassed = np.fft.ifft2(np.fft.fft2(y) * mask).real.astype(np.float32)
    return _normalize_zero_mean_unit_std(bandpassed)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross correlation between same-shaped arrays."""

    a_norm = _normalize_zero_mean_unit_std(np.asarray(a, dtype=np.float32))
    b_norm = _normalize_zero_mean_unit_std(np.asarray(b, dtype=np.float32))
    if a_norm.shape != b_norm.shape:
        raise ValueError(f"Shape mismatch for NCC: {a_norm.shape} vs {b_norm.shape}")
    return float(np.mean(a_norm * b_norm))


def rotate_image_keep_size(
    img: np.ndarray,
    angle: float,
    mode: str = "reflect",
) -> np.ndarray:
    """Rotate a 2D array or HxWxC image while preserving its original size."""

    if mode not in {"reflect", "constant"}:
        raise ValueError('mode must be "reflect" or "constant"')

    arr = np.asarray(img, dtype=np.float32)
    rotated = ndimage.rotate(
        arr,
        float(angle),
        axes=(1, 0),
        reshape=False,
        order=1,
        mode=mode,
        cval=0.0,
        prefilter=False,
    )
    return rotated.astype(np.float32)


def _angle_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Search step must be positive")
    count = int(np.floor((stop - start) / step)) + 1
    grid = start + step * np.arange(max(count, 1), dtype=np.float32)
    if grid.size == 0 or grid[-1] < stop - 1e-6:
        grid = np.append(grid, np.float32(stop))
    return grid.astype(np.float32)


def detect_rotation_angle(
    img_rgb: np.ndarray,
    delta: np.ndarray,
    frequency_mask: np.ndarray,
    coarse_step: float = 2.0,
    fine_step: float = 0.25,
    angle_range: Tuple[float, float] = (-180, 180),
    mode: str = "reflect",
) -> Tuple[float, float, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Estimate attack rotation angle by correlating bandpassed Y with delta.

    Sign convention: if the attack is ``rotate_image_keep_size(x, theta)``, this
    function should return ``theta_hat ~= theta``. Correction then uses
    ``rotate_image_keep_size(x_att, -theta_hat)``.
    """

    delta_arr = np.asarray(delta, dtype=np.float32)
    if delta_arr.ndim != 2:
        raise ValueError(f"Expected 2D delta, got {delta_arr.shape}")

    bandpassed_y = bandpass_y_channel(img_rgb, frequency_mask)
    if delta_arr.shape != bandpassed_y.shape:
        raise ValueError(
            f"Delta shape {delta_arr.shape} does not match image {bandpassed_y.shape}"
        )

    angle_min, angle_max = float(angle_range[0]), float(angle_range[1])
    if angle_min > angle_max:
        raise ValueError("angle_range must be ordered as (min, max)")

    coarse_angles = _angle_grid(angle_min, angle_max, float(coarse_step))
    coarse_scores = np.array(
        [ncc(bandpassed_y, rotate_image_keep_size(delta_arr, theta, mode=mode)) for theta in coarse_angles],
        dtype=np.float32,
    )
    best_coarse = float(coarse_angles[int(np.argmax(coarse_scores))])

    fine_min = max(angle_min, best_coarse - float(coarse_step))
    fine_max = min(angle_max, best_coarse + float(coarse_step))
    fine_angles = _angle_grid(fine_min, fine_max, float(fine_step))
    fine_scores = np.array(
        [ncc(bandpassed_y, rotate_image_keep_size(delta_arr, theta, mode=mode)) for theta in fine_angles],
        dtype=np.float32,
    )
    best_idx = int(np.argmax(fine_scores))
    theta_hat = float(fine_angles[best_idx])
    best_score = float(fine_scores[best_idx])

    return theta_hat, best_score, {
        "coarse": (coarse_angles, coarse_scores),
        "fine": (fine_angles, fine_scores),
    }


def remove_anchor_rgb(
    img_rgb_corr: np.ndarray,
    delta: np.ndarray,
    frequency_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Estimate and subtract the anchor strength from a corrected RGB image."""

    img = _validate_rgb(img_rgb_corr)
    delta_arr = np.asarray(delta, dtype=np.float32)
    if delta_arr.shape != img.shape[:2]:
        raise ValueError(f"Delta shape {delta_arr.shape} does not match image {img.shape[:2]}")

    y, cb, cr = _rgb_to_ycbcr(img)
    if frequency_mask is None:
        y_for_fit = y.astype(np.float32)
        delta_for_fit = delta_arr
    else:
        mask = np.asarray(frequency_mask)
        if mask.shape != y.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image {y.shape}")
        y_for_fit = np.fft.ifft2(np.fft.fft2(y) * mask).real.astype(np.float32)
        delta_for_fit = np.fft.ifft2(np.fft.fft2(delta_arr) * mask).real.astype(np.float32)

    numerator = float(np.sum(y_for_fit * delta_for_fit))
    denominator = float(np.sum(delta_for_fit * delta_for_fit))
    alpha_hat = 0.0 if denominator < EPS else numerator / denominator

    y_clean = np.clip(y - alpha_hat * delta_arr, 0.0, 1.0).astype(np.float32)
    return _ycbcr_to_rgb(y_clean, cb, cr), float(alpha_hat)
