"""Fourier-polar angular synchronization anchor (V1).

V1 estimates rotation by turning angular rotation in Fourier magnitude into a
circular shift along the polar angle axis, then using FFT-based circular
correlation. It keeps V0 in ``freq_anchor.py`` untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from freq_anchor import (
    bandpass_2d,
    bandpass_y_channel,
    ncc,
    rotate_image_keep_size,
)


EPS = 1e-8


def _normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr - np.mean(arr)
    std = float(np.std(arr))
    if std < EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / std).astype(np.float32)


def _rgb_to_y(img_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(img_rgb, dtype=np.float32)
    if img.ndim == 2:
        return img
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image or 2D Y channel, got {img.shape}")
    return (
        0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    ).astype(np.float32)


def _validate_rgb(img_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(img_rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img.shape}")
    return img


def _rgb_to_ycbcr(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = _validate_rgb(img_rgb)
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


def make_angular_code(num_angles: int, key: int = 0, mode: str = "rademacher") -> np.ndarray:
    """Generate a normalized pseudo-random angular code."""

    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    rng = np.random.default_rng(key)
    if mode == "rademacher":
        code = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=num_angles)
    else:
        raise ValueError(f"Unsupported angular code mode: {mode}")
    return _normalize(code)


def make_radial_window_freq(
    H: int,
    W: int,
    rmin: float = 0.12,
    rmax: float = 0.28,
    soft: bool = True,
) -> np.ndarray:
    """Construct a Cartesian frequency-domain radial mid-band window."""

    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")
    if not 0 <= rmin < rmax:
        raise ValueError("Expected 0 <= rmin < rmax")

    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)

    if not soft:
        return ((r >= rmin) & (r <= rmax)).astype(np.float32)

    width = max((rmax - rmin) * 0.15, 1.0 / max(H, W))
    inner_hi = min(rmin + width, (rmin + rmax) * 0.5)
    outer_lo = max(rmax - width, (rmin + rmax) * 0.5)

    win = np.zeros_like(r, dtype=np.float32)
    plateau = (r >= inner_hi) & (r <= outer_lo)
    win[plateau] = 1.0

    rising = (r >= rmin) & (r < inner_hi)
    if np.any(rising):
        t = (r[rising] - rmin) / max(inner_hi - rmin, EPS)
        win[rising] = 0.5 - 0.5 * np.cos(np.pi * t)

    falling = (r > outer_lo) & (r <= rmax)
    if np.any(falling):
        t = (rmax - r[falling]) / max(rmax - outer_lo, EPS)
        win[falling] = 0.5 - 0.5 * np.cos(np.pi * t)

    return win.astype(np.float32)


def _angle_code_on_grid(H: int, W: int, code: np.ndarray) -> np.ndarray:
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    bins = np.floor(phi / (2 * np.pi) * len(code)).astype(np.int64) % len(code)
    return code[bins].astype(np.float32)


def _pi_periodic_angular_code(num_angles: int, key: int) -> np.ndarray:
    half_angles = max(1, num_angles // 2)
    half_code = make_angular_code(half_angles, key=key)
    angular_code = np.tile(half_code, 2)
    if angular_code.size < num_angles:
        angular_code = np.append(angular_code, angular_code[: num_angles - angular_code.size])
    return angular_code[:num_angles].astype(np.float32)


def make_polar_anchor_template(
    H: int,
    W: int,
    key: int = 0,
    rmin: float = 0.12,
    rmax: float = 0.28,
    num_angles: int = 360,
    beta: float = 0.5,
):
    """Create a pixel-frequency polar angular anchor residual."""

    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    angular_code = _pi_periodic_angular_code(num_angles, key)

    radial = make_radial_window_freq(H, W, rmin=rmin, rmax=rmax, soft=True)
    angular_grid = _angle_code_on_grid(H, W, angular_code)
    amp = radial * np.maximum(0.0, 1.0 + float(beta) * angular_grid)

    rng = np.random.default_rng(key + 1009)
    noise = rng.standard_normal((H, W)).astype(np.float32)
    phase_source = np.fft.fft2(noise)
    phase = np.exp(1j * np.angle(phase_source))
    spectrum = amp * phase
    delta = np.fft.ifft2(spectrum).real.astype(np.float32)
    delta = _normalize(delta)

    polar_template = fft_polar_magnitude(
        delta,
        rmin=rmin,
        rmax=rmax,
        num_r=64,
        num_angles=num_angles,
        log_magnitude=True,
        normalize="per_radius",
    )
    metadata = {
        "rmin": rmin,
        "rmax": rmax,
        "num_angles": num_angles,
        "beta": beta,
        "key": key,
        "ambiguity": "180deg",
    }
    return delta.astype(np.float32), polar_template.astype(np.float32), metadata


def make_polar_magnitude_template(
    H: int,
    W: int,
    key: int = 0,
    rmin: float = 0.12,
    rmax: float = 0.28,
    num_angles: int = 360,
    beta: float = 1.0,
):
    """Create a V1.1 frequency-magnitude modulation anchor template.

    ``modulation_grid`` is defined on the unshifted FFT grid so it can be
    multiplied directly with ``np.fft.fft2(Y)``. Its angular code is pi-periodic,
    which keeps the real-valued image Hermitian symmetry but leaves the expected
    180 degree ambiguity for the optional resolver.
    """

    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")

    angular_code = _pi_periodic_angular_code(num_angles, key)
    radial = make_radial_window_freq(H, W, rmin=rmin, rmax=rmax, soft=True)
    angular_grid = _angle_code_on_grid(H, W, angular_code)
    modulation_grid = (radial * float(beta) * angular_grid).astype(np.float32)
    modulation_grid = modulation_grid - float(modulation_grid.mean())
    max_abs = float(np.max(np.abs(modulation_grid)))
    if max_abs > EPS:
        modulation_grid = modulation_grid / max_abs
    modulation_grid[0, 0] = 0.0

    polar_template = np.tile(angular_code[None, :], (64, 1)).astype(np.float32)
    polar_template = polar_template - polar_template.mean(axis=1, keepdims=True)
    polar_template = polar_template / np.maximum(
        polar_template.std(axis=1, keepdims=True), EPS
    )
    metadata = {
        "method": "v1_magmod",
        "rmin": rmin,
        "rmax": rmax,
        "num_angles": num_angles,
        "beta": beta,
        "key": key,
        "ambiguity": "180deg",
    }
    return modulation_grid.astype(np.float32), polar_template.astype(np.float32), metadata


def embed_polar_magnitude_anchor_rgb(
    img_rgb: np.ndarray, modulation_grid: np.ndarray, alpha: float
) -> np.ndarray:
    """Embed an angular code by modulating luminance FFT magnitudes."""

    y, cb, cr = _rgb_to_ycbcr(img_rgb)
    grid = np.asarray(modulation_grid, dtype=np.float32)
    if grid.shape != y.shape:
        raise ValueError(f"Modulation grid shape {grid.shape} does not match image {y.shape}")

    F = np.fft.fft2(y)
    gain = np.clip(1.0 + float(alpha) * grid, 0.1, 10.0).astype(np.float32)
    y_sync = np.fft.ifft2(F * gain).real.astype(np.float32)
    y_sync = np.clip(y_sync, 0.0, 1.0).astype(np.float32)
    return _ycbcr_to_rgb(y_sync, cb, cr)


def fft_polar_magnitude(
    img_or_y,
    rmin: float = 0.12,
    rmax: float = 0.28,
    num_r: int = 64,
    num_angles: int = 360,
    log_magnitude: bool = True,
    normalize: str = "per_radius",
) -> np.ndarray:
    """Extract Fourier magnitude on a polar frequency grid."""

    y = _rgb_to_y(img_or_y)
    H, W = y.shape
    F = np.fft.fftshift(np.fft.fft2(y))
    mag = np.abs(F).astype(np.float32)
    if log_magnitude:
        mag = np.log1p(mag).astype(np.float32)

    radii = np.linspace(rmin, rmax, num_r, dtype=np.float32)
    angles = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False, dtype=np.float32)
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    fx = rr * np.cos(aa)
    fy = rr * np.sin(aa)
    coords_y = H // 2 + fy * H
    coords_x = W // 2 + fx * W
    polar = ndimage.map_coordinates(
        mag,
        [coords_y, coords_x],
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)

    if normalize == "per_radius":
        polar = polar - polar.mean(axis=1, keepdims=True)
        polar = polar / np.maximum(polar.std(axis=1, keepdims=True), EPS)
    elif normalize in {None, "none"}:
        pass
    else:
        raise ValueError(f"Unsupported normalize mode: {normalize}")
    return polar.astype(np.float32)


def circular_corr_angle(polar_feature: np.ndarray, polar_template: np.ndarray):
    """Estimate angular circular shift using FFT-based circular correlation."""

    feature = np.asarray(polar_feature, dtype=np.float32)
    template = np.asarray(polar_template, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError("polar_feature must have shape [num_r, num_angles]")
    if template.ndim == 1:
        template = np.tile(template[None, :], (feature.shape[0], 1))
    if template.shape[1] != feature.shape[1]:
        raise ValueError(
            f"Angle bins differ: feature {feature.shape[1]} vs template {template.shape[1]}"
        )
    if template.shape[0] != feature.shape[0]:
        x_old = np.linspace(0.0, 1.0, template.shape[0], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, feature.shape[0], dtype=np.float32)
        resized = [np.interp(x_new, x_old, template[:, i]) for i in range(template.shape[1])]
        template = np.stack(resized, axis=1).astype(np.float32)

    feature = feature - feature.mean(axis=1, keepdims=True)
    feature = feature / np.maximum(feature.std(axis=1, keepdims=True), EPS)
    template = template - template.mean(axis=1, keepdims=True)
    template = template / np.maximum(template.std(axis=1, keepdims=True), EPS)

    corr = np.fft.ifft(
        np.fft.fft(feature, axis=1) * np.conj(np.fft.fft(template, axis=1)),
        axis=1,
    ).real
    score_curve = corr.mean(axis=0).astype(np.float32)
    angle_bin = int(np.argmax(score_curve))
    best_score = float(score_curve[angle_bin])
    return angle_bin, best_score, score_curve


def _anchor_score(img_rgb: np.ndarray, delta: np.ndarray, rmin: float, rmax: float) -> float:
    H, W = delta.shape
    mask = make_radial_window_freq(H, W, rmin=rmin, rmax=rmax, soft=False)
    return ncc(bandpass_y_channel(img_rgb, mask), bandpass_2d(delta, mask))


def resolve_180_ambiguity(
    img_rgb,
    delta,
    theta_mod180: float,
    mode: str = "spatial_ncc",
    rmin: float = 0.12,
    rmax: float = 0.28,
):
    """Resolve magnitude-only 180 degree ambiguity using two candidates."""

    if mode != "spatial_ncc":
        raise ValueError(f"Unsupported ambiguity resolver: {mode}")
    theta1 = float(theta_mod180) % 360.0
    theta2 = (theta1 + 180.0) % 360.0
    corr1 = rotate_image_keep_size(img_rgb, -theta1)
    corr2 = rotate_image_keep_size(img_rgb, -theta2)
    score1 = _anchor_score(corr1, delta, rmin, rmax)
    score2 = _anchor_score(corr2, delta, rmin, rmax)
    return (theta1, {"candidate_score_0": score1, "candidate_score_180": score2}) if score1 >= score2 else (
        theta2,
        {"candidate_score_0": score1, "candidate_score_180": score2},
    )


def detect_rotation_angle_polar(
    img_rgb,
    polar_template,
    delta=None,
    rmin: float = 0.12,
    rmax: float = 0.28,
    num_r: int = 64,
    num_angles: int = 360,
    resolve_ambiguity: bool = True,
):
    """Detect rotation angle with Fourier-polar circular correlation."""

    polar_feature = fft_polar_magnitude(
        img_rgb,
        rmin=rmin,
        rmax=rmax,
        num_r=num_r,
        num_angles=num_angles,
    )
    angle_bin, best_score, score_curve = circular_corr_angle(polar_feature, polar_template)
    raw_shift = angle_bin * 360.0 / float(num_angles)
    theta_mod = (-raw_shift) % 180.0
    theta_hat = theta_mod
    if score_curve.size > 1:
        top2_score = float(np.partition(score_curve, -2)[-2])
    else:
        top2_score = np.nan
    corr_margin = (
        float(best_score - top2_score) if not np.isnan(top2_score) else np.nan
    )
    extra = {
        "angle_bin": angle_bin,
        "raw_shift": raw_shift,
        "theta_mod": theta_mod,
        "top2_score": top2_score,
        "corr_margin": corr_margin,
        "ambiguity_resolved": False,
        "candidate_score_0": np.nan,
        "candidate_score_180": np.nan,
    }

    if resolve_ambiguity and delta is not None:
        theta_hat, scores = resolve_180_ambiguity(
            img_rgb, delta, theta_mod, rmin=rmin, rmax=rmax
        )
        extra.update(scores)
        extra["ambiguity_resolved"] = True

    if theta_hat > 180.0:
        theta_hat -= 360.0
    return float(theta_hat), best_score, score_curve, extra
