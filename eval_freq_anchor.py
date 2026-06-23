"""Standalone baseline evaluation for pixel-frequency rotation anchors.

This script is deliberately minimal and independent of diffusion code. If an
image is provided, it treats that image as the already generated watermarked
image ``x_w``. Otherwise it creates a deterministic synthetic RGB image so the
frequency-anchor pipeline can be smoke-tested end to end.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import importlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from freq_anchor import (
    bandpass_2d,
    bandpass_y_channel,
    detect_rotation_angle,
    embed_anchor_rgb,
    make_anchor_template,
    ncc,
    remove_anchor_rgb,
    rotate_image_keep_size,
)


CSV_FIELDS = [
    "image_id",
    "alpha",
    "rmin",
    "rmax",
    "theta_gt",
    "theta_hat",
    "angle_error",
    "ncc_score",
    "psnr_sync",
    "ssim_sync",
    "vae_mse_sync",
    "vae_cos_sync",
    "zt_mse_sync",
    "zt_cos_sync",
    "base_score",
    "oracle_score",
    "anchorsync_score",
    "anchorsync_remove_score",
    "anchor_base_score",
    "anchor_oracle_score",
    "anchor_sync_score",
    "anchor_remove_score",
    "alpha_hat",
]

SUMMARY_FIELDS = [
    "alpha",
    "mean_angle_error",
    "max_angle_error",
    "failure_rate_error_gt_1deg",
    "failure_rate_error_gt_3deg",
    "mean_psnr_sync",
    "mean_ssim_sync",
    "mean_anchor_oracle_score",
    "mean_anchor_sync_score",
    "mean_anchor_remove_score",
    "mean_vae_mse_sync",
    "mean_vae_cos_sync",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

_SSIM_HAS_CHANNEL_AXIS = "channel_axis" in inspect.signature(structural_similarity).parameters


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def load_rgb_image(path: Path, size: Optional[int] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return (np.asarray(img).astype(np.float32) / 255.0).clip(0.0, 1.0)


def collect_inputs(args: argparse.Namespace) -> List[Tuple[str, np.ndarray]]:
    resize_size = None if args.no_resize else args.size
    if args.image_dir:
        image_dir = Path(args.image_dir)
        paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No png/jpg/jpeg images found in {image_dir}")
        return [(path.stem, load_rgb_image(path, size=resize_size)) for path in paths]
    if args.image:
        path = Path(args.image)
        return [(path.stem, load_rgb_image(path, size=resize_size))]
    return [("synthetic", make_synthetic_image(args.size, key=args.key))]


def make_synthetic_image(size: int, key: int = 0) -> np.ndarray:
    rng = np.random.default_rng(key)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    base = np.stack(
        [
            0.25 + 0.55 * xx,
            0.20 + 0.45 * yy,
            0.35 + 0.25 * np.sin(2 * np.pi * (xx + yy)),
        ],
        axis=-1,
    )
    texture = rng.normal(0.0, 0.015, size=(size, size, 3)).astype(np.float32)
    return np.clip(base + texture, 0.0, 1.0).astype(np.float32)


def save_rgb(path: Path, img: np.ndarray) -> None:
    arr = np.clip(np.asarray(img), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_delta_vis(path: Path, delta: np.ndarray) -> None:
    delta_arr = np.asarray(delta, dtype=np.float32)
    lo, hi = np.percentile(delta_arr, [1, 99])
    vis = (delta_arr - lo) / max(float(hi - lo), 1e-8)
    Image.fromarray((np.clip(vis, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if _SSIM_HAS_CHANNEL_AXIS:
        return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
    return float(structural_similarity(a, b, multichannel=True, data_range=1.0))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-8:
        return math.nan
    return float(np.dot(av, bv) / denom)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2))


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float32)


def resolve_vae_encoder(args: argparse.Namespace) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    if not args.enable_vae_metrics:
        return None

    dotted = args.vae_encoder or os.environ.get("FREQ_ANCHOR_VAE_ENCODER")
    if not dotted:
        raise RuntimeError(
            "--enable-vae-metrics requires a project VAE encode hook. "
            "Pass --vae-encoder module:function or set FREQ_ANCHOR_VAE_ENCODER."
        )
    if ":" not in dotted:
        raise ValueError("VAE encoder hook must be in module:function format")

    module_name, function_name = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    encoder = getattr(module, function_name)
    if not callable(encoder):
        raise TypeError(f"VAE encoder hook {dotted} is not callable")
    return encoder


def angular_error(theta_hat: float, theta_gt: float) -> float:
    return abs(((theta_hat - theta_gt + 180.0) % 360.0) - 180.0)


def surrogate_anchor_score(img_rgb: np.ndarray, delta: np.ndarray, mask: np.ndarray) -> float:
    return ncc(bandpass_y_channel(img_rgb, mask), bandpass_2d(delta, mask))


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    summary_rows = summarize_by_alpha(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def finite_values(rows: Sequence[Dict[str, float]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = float(row[key])
        if not math.isnan(value):
            values.append(value)
    return values


def mean_or_nan(values: Sequence[float]) -> float:
    return math.nan if not values else float(np.mean(values))


def summarize_by_alpha(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    summary_rows = []
    for alpha in sorted({float(row["alpha"]) for row in rows}):
        subset = [row for row in rows if float(row["alpha"]) == alpha]
        angle_errors = finite_values(subset, "angle_error")
        n = len(angle_errors)
        summary_rows.append(
            {
                "alpha": alpha,
                "mean_angle_error": mean_or_nan(angle_errors),
                "max_angle_error": math.nan if not angle_errors else float(np.max(angle_errors)),
                "failure_rate_error_gt_1deg": math.nan
                if not n
                else float(np.mean([err > 1.0 for err in angle_errors])),
                "failure_rate_error_gt_3deg": math.nan
                if not n
                else float(np.mean([err > 3.0 for err in angle_errors])),
                "mean_psnr_sync": mean_or_nan(finite_values(subset, "psnr_sync")),
                "mean_ssim_sync": mean_or_nan(finite_values(subset, "ssim_sync")),
                "mean_anchor_oracle_score": mean_or_nan(
                    finite_values(subset, "anchor_oracle_score")
                ),
                "mean_anchor_sync_score": mean_or_nan(finite_values(subset, "anchor_sync_score")),
                "mean_anchor_remove_score": mean_or_nan(
                    finite_values(subset, "anchor_remove_score")
                ),
                "mean_vae_mse_sync": mean_or_nan(finite_values(subset, "vae_mse_sync")),
                "mean_vae_cos_sync": mean_or_nan(finite_values(subset, "vae_cos_sync")),
            }
        )
    return summary_rows


def aggregate_line_rows(
    rows: Sequence[Dict[str, float]],
    x_key: str,
    y_keys: Sequence[str],
) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[float, float], Dict[str, List[float]]] = {}
    for row in rows:
        group_key = (float(row["alpha"]), float(row[x_key]))
        grouped.setdefault(group_key, {y_key: [] for y_key in y_keys})
        for y_key in y_keys:
            value = float(row[y_key])
            if not math.isnan(value):
                grouped[group_key][y_key].append(value)

    aggregated = []
    for (alpha, x_value), values_by_key in sorted(grouped.items()):
        row = {"alpha": alpha, x_key: x_value}
        for y_key in y_keys:
            row[y_key] = mean_or_nan(values_by_key[y_key])
        aggregated.append(row)
    return aggregated


def plot_line(
    path: Path,
    rows: Sequence[Dict[str, float]],
    x_key: str,
    y_keys: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7, 4))
    rows_for_plot = aggregate_line_rows(rows, x_key, y_keys)
    alphas = sorted({float(row["alpha"]) for row in rows_for_plot})
    for y_key in y_keys:
        for alpha in alphas:
            subset = [row for row in rows_for_plot if float(row["alpha"]) == alpha]
            xs = [float(row[x_key]) for row in subset]
            ys = [float(row[y_key]) for row in subset]
            label = f"{y_key}, alpha={alpha:g}" if len(alphas) > 1 else y_key
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=label)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_metric_vs_alpha(
    path: Path,
    rows: Sequence[Dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    grouped: Dict[float, List[float]] = {}
    for row in rows:
        value = float(row[metric])
        if math.isnan(value):
            continue
        grouped.setdefault(float(row["alpha"]), []).append(value)

    plt.figure(figsize=(6, 4))
    if grouped:
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[x])) for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.8)
    else:
        plt.text(0.5, 0.5, "nan (not connected)", ha="center", va="center")
    plt.title(title)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_angle_error_box_by_alpha(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    alphas = sorted({float(row["alpha"]) for row in rows})
    data = [[float(row["angle_error"]) for row in rows if float(row["alpha"]) == alpha] for alpha in alphas]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=[f"{alpha:g}" for alpha in alphas], showmeans=True)
    plt.title("Angle error by alpha")
    plt.xlabel("alpha")
    plt.ylabel("absolute angular error (deg)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_anchor_sync_vs_oracle_by_alpha(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    summary_rows = summarize_by_alpha(rows)
    xs = [float(row["alpha"]) for row in summary_rows]
    oracle = [float(row["mean_anchor_oracle_score"]) for row in summary_rows]
    sync = [float(row["mean_anchor_sync_score"]) for row in summary_rows]
    removed = [float(row["mean_anchor_remove_score"]) for row in summary_rows]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, oracle, marker="o", linewidth=1.8, label="anchor_oracle")
    plt.plot(xs, sync, marker="o", linewidth=1.8, label="anchor_sync")
    plt.plot(xs, removed, marker="o", linewidth=1.8, label="anchor_remove")
    plt.title("Anchor sync vs oracle by alpha")
    plt.xlabel("alpha")
    plt.ylabel("mean surrogate NCC score")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_failure_rate_vs_alpha(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    summary_rows = summarize_by_alpha(rows)
    xs = [float(row["alpha"]) for row in summary_rows]
    gt1 = [float(row["failure_rate_error_gt_1deg"]) for row in summary_rows]
    gt3 = [float(row["failure_rate_error_gt_3deg"]) for row in summary_rows]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, gt1, marker="o", linewidth=1.8, label="error > 1 deg")
    plt.plot(xs, gt3, marker="o", linewidth=1.8, label="error > 3 deg")
    plt.title("Failure rate vs alpha")
    plt.xlabel("alpha")
    plt.ylabel("failure rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_example_grid(path: Path, examples: Sequence[Tuple[str, np.ndarray]]) -> None:
    cols = len(examples)
    plt.figure(figsize=(3 * cols, 3))
    for idx, (title, img) in enumerate(examples, start=1):
        plt.subplot(1, cols, idx)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(np.clip(img, 0.0, 1.0))
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_eval(args: argparse.Namespace) -> List[Dict[str, float]]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = collect_inputs(args)
    alphas = parse_float_list(args.alphas)
    angles = parse_float_list(args.angles)
    vae_encoder = resolve_vae_encoder(args)

    rows: List[Dict[str, float]] = []
    fallback_example_payload = None
    selected_example_payload = None
    total = len(inputs) * len(alphas) * len(angles)
    done = 0

    for image_id, x_w in inputs:
        H, W = x_w.shape[:2]
        delta, frequency_mask = make_anchor_template(
            H,
            W,
            key=args.key,
            rmin=args.rmin,
            rmax=args.rmax,
            circular_window=not args.no_circular_window,
        )

        save_rgb(outdir / f"x_w_{image_id}.png", x_w)
        save_delta_vis(outdir / f"anchor_delta_{image_id}.png", delta)
        if len(inputs) == 1:
            save_rgb(outdir / "x_w.png", x_w)
            save_delta_vis(outdir / "anchor_delta.png", delta)

        z0_w = to_numpy(vae_encoder(x_w)) if vae_encoder is not None else None

        for alpha in alphas:
            x_sync = embed_anchor_rgb(x_w, delta, alpha=alpha)
            save_rgb(outdir / f"x_sync_{image_id}_alpha_{alpha:g}.png", x_sync)

            psnr_sync = psnr(x_w, x_sync)
            ssim_sync = ssim(x_w, x_sync)
            if vae_encoder is None:
                vae_mse_sync = math.nan
                vae_cos_sync = math.nan
            else:
                z0_sync = to_numpy(vae_encoder(x_sync))
                vae_mse_sync = mse(z0_w, z0_sync)
                vae_cos_sync = cosine_similarity(z0_w, z0_sync)
            zt_mse_sync = math.nan
            zt_cos_sync = math.nan

            for theta in angles:
                x_att = rotate_image_keep_size(x_sync, theta, mode=args.rotate_mode)
                theta_hat, best_score, _curve = detect_rotation_angle(
                    x_att,
                    delta,
                    frequency_mask,
                    coarse_step=args.coarse_step,
                    fine_step=args.fine_step,
                    angle_range=(args.angle_min, args.angle_max),
                    mode=args.rotate_mode,
                )
                x_corr = rotate_image_keep_size(x_att, -theta_hat, mode=args.rotate_mode)
                x_corr_clean, alpha_hat = remove_anchor_rgb(
                    x_corr,
                    delta,
                    frequency_mask if args.remove_with_bandpass else None,
                )

                # Reserved for the real watermark/noise-space detector once the
                # diffusion pipeline is connected.
                base_score = math.nan
                oracle_score = math.nan
                anchorsync_score = math.nan
                anchorsync_remove_score = math.nan

                anchor_base_score = surrogate_anchor_score(x_att, delta, frequency_mask)
                anchor_oracle_img = rotate_image_keep_size(x_att, -theta, mode=args.rotate_mode)
                anchor_oracle_score = surrogate_anchor_score(
                    anchor_oracle_img, delta, frequency_mask
                )
                anchor_sync_score = surrogate_anchor_score(x_corr, delta, frequency_mask)
                anchor_remove_score = surrogate_anchor_score(x_corr_clean, delta, frequency_mask)

                rows.append(
                    {
                        "image_id": image_id,
                        "alpha": alpha,
                        "rmin": args.rmin,
                        "rmax": args.rmax,
                        "theta_gt": theta,
                        "theta_hat": theta_hat,
                        "angle_error": angular_error(theta_hat, theta),
                        "ncc_score": best_score,
                        "psnr_sync": psnr_sync,
                        "ssim_sync": ssim_sync,
                        "vae_mse_sync": vae_mse_sync,
                        "vae_cos_sync": vae_cos_sync,
                        "zt_mse_sync": zt_mse_sync,
                        "zt_cos_sync": zt_cos_sync,
                        "base_score": base_score,
                        "oracle_score": oracle_score,
                        "anchorsync_score": anchorsync_score,
                        "anchorsync_remove_score": anchorsync_remove_score,
                        "anchor_base_score": anchor_base_score,
                        "anchor_oracle_score": anchor_oracle_score,
                        "anchor_sync_score": anchor_sync_score,
                        "anchor_remove_score": anchor_remove_score,
                        "alpha_hat": alpha_hat,
                    }
                )

                candidate_payload = (
                    x_w,
                    x_sync,
                    x_att,
                    x_corr,
                    x_corr_clean,
                    delta,
                )
                if fallback_example_payload is None:
                    fallback_example_payload = candidate_payload
                if (
                    selected_example_payload is None
                    and abs(alpha - args.example_alpha) < 1e-12
                    and abs(theta - args.example_theta) < 1e-12
                ):
                    selected_example_payload = candidate_payload

                done += 1
                if not args.no_show_progress:
                    print(
                        f"[{done}/{total}] image={image_id} alpha={alpha:g} "
                        f"theta={theta:g} theta_hat={theta_hat:.2f}"
                    )

    write_csv(outdir / "freq_anchor_results.csv", rows)
    write_summary_csv(outdir / "summary.csv", rows)

    plot_line(
        outdir / "angle_error_vs_theta.png",
        rows,
        "theta_gt",
        ["angle_error"],
        "Angle error vs theta",
        "absolute angular error (deg)",
    )
    plot_line(
        outdir / "surrogate_anchor_score_vs_theta.png",
        rows,
        "theta_gt",
        [
            "anchor_base_score",
            "anchor_oracle_score",
            "anchor_sync_score",
            "anchor_remove_score",
        ],
        "Surrogate anchor score vs theta",
        "surrogate NCC score",
    )
    plot_metric_vs_alpha(
        outdir / "quality_vs_alpha.png",
        rows,
        "psnr_sync",
        "Quality vs alpha",
        "PSNR(x_sync, x_w)",
    )
    plot_metric_vs_alpha(
        outdir / "vae_mse_vs_alpha.png",
        rows,
        "vae_mse_sync",
        "VAE MSE vs alpha",
        "VAE latent MSE",
    )
    plot_angle_error_box_by_alpha(outdir / "angle_error_box_by_alpha.png", rows)
    plot_anchor_sync_vs_oracle_by_alpha(outdir / "anchor_sync_vs_oracle_by_alpha.png", rows)
    plot_failure_rate_vs_alpha(outdir / "failure_rate_vs_alpha.png", rows)

    example_payload = selected_example_payload or fallback_example_payload
    if example_payload is not None:
        x_w_example, x_sync, x_att, x_corr, x_corr_clean, delta_example = example_payload
        delta_vis = np.asarray(delta_example, dtype=np.float32)
        delta_vis = (delta_vis - delta_vis.min()) / max(float(delta_vis.max() - delta_vis.min()), 1e-8)
        save_example_grid(
            outdir / "example_grid.png",
            [
                ("original watermarked", x_w_example),
                ("anchor image", x_sync),
                ("rotated attack", x_att),
                ("corrected", x_corr),
                ("corrected removed", x_corr_clean),
                ("anchor residual", delta_vis),
            ],
        )

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="Optional x_w RGB image path")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional directory of png/jpg/jpeg x_w images. Takes priority over --image.",
    )
    parser.add_argument("--outdir", default="freq_anchor_eval", help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Resize input or synthetic size")
    parser.add_argument("--no-resize", action="store_true", help="Keep input image dimensions")
    parser.add_argument("--key", type=int, default=0, help="Anchor random seed")
    parser.add_argument("--alphas", default="0.001,0.002,0.003,0.005")
    parser.add_argument("--rmin", type=float, default=0.12)
    parser.add_argument("--rmax", type=float, default=0.28)
    parser.add_argument("--angles", default="5,10,15,30,45,60,75,90,120,150,180")
    parser.add_argument("--coarse-step", type=float, default=2.0)
    parser.add_argument("--fine-step", type=float, default=0.25)
    parser.add_argument("--angle-min", type=float, default=-180.0)
    parser.add_argument("--angle-max", type=float, default=180.0)
    parser.add_argument("--rotate-mode", choices=["reflect", "constant"], default="reflect")
    parser.add_argument("--no-circular-window", action="store_true")
    parser.add_argument("--remove-with-bandpass", action="store_true")
    parser.add_argument("--example-alpha", type=float, default=0.003)
    parser.add_argument("--example-theta", type=float, default=45.0)
    parser.add_argument("--enable-vae-metrics", action="store_true")
    parser.add_argument(
        "--vae-encoder",
        default=None,
        help="Optional module:function hook that VAE-encodes an RGB float32 image.",
    )
    parser.add_argument("--no-show-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
