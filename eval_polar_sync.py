"""Evaluate V1 Fourier-polar rotation synchronization anchors.

V1 keeps V0 available for comparison but estimates angle with FFT-based
circular correlation in Fourier-polar magnitude, not by rotating templates over
all candidate angles.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_freq_anchor import (
    angular_error,
    collect_inputs,
    parse_float_list,
    psnr,
    save_delta_vis,
    save_example_grid,
    save_rgb,
    ssim,
)
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
from polar_sync import (
    detect_rotation_angle_polar,
    embed_polar_magnitude_anchor_rgb,
    make_polar_anchor_template,
    make_polar_magnitude_template,
    make_radial_window_freq,
)


CSV_FIELDS = [
    "image_id",
    "method",
    "alpha",
    "rmin",
    "rmax",
    "theta_gt",
    "theta_hat",
    "angle_error",
    "best_score",
    "runtime_ms",
    "psnr_sync",
    "ssim_sync",
    "anchor_oracle_score",
    "anchor_sync_score",
    "anchor_remove_score",
    "raw_shift",
    "theta_mod",
    "angle_bin",
    "corr_margin",
    "top2_score",
    "ambiguity_resolved",
    "candidate_score_0",
    "candidate_score_180",
]

SUMMARY_FIELDS = [
    "method",
    "alpha",
    "mean_angle_error",
    "max_angle_error",
    "failure_rate_error_gt_1deg",
    "failure_rate_error_gt_3deg",
    "mean_runtime_ms",
    "mean_psnr_sync",
    "mean_ssim_sync",
    "mean_anchor_oracle_score",
    "mean_anchor_sync_score",
    "mean_anchor_remove_score",
]


def anchor_score(img_rgb: np.ndarray, delta: np.ndarray, rmin: float, rmax: float) -> float:
    H, W = delta.shape
    mask = make_radial_window_freq(H, W, rmin=rmin, rmax=rmax, soft=False)
    return ncc(bandpass_y_channel(img_rgb, mask), bandpass_2d(delta, mask))


def normalize_proxy(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr - float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / std).astype(np.float32)


def spatial_proxy_from_modulation(modulation_grid: np.ndarray) -> np.ndarray:
    return normalize_proxy(np.fft.ifft2(modulation_grid).real.astype(np.float32))


def finite_values(rows: Sequence[Dict[str, float]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = float(row[key])
        if not math.isnan(value):
            values.append(value)
    return values


def mean_or_nan(values: Sequence[float]) -> float:
    return math.nan if not values else float(np.mean(values))


def summarize(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    out = []
    keys = sorted({(row["method"], float(row["alpha"])) for row in rows})
    for method, alpha in keys:
        subset = [
            row for row in rows if row["method"] == method and float(row["alpha"]) == alpha
        ]
        errors = finite_values(subset, "angle_error")
        n = len(errors)
        out.append(
            {
                "method": method,
                "alpha": alpha,
                "mean_angle_error": mean_or_nan(errors),
                "max_angle_error": math.nan if not errors else float(np.max(errors)),
                "failure_rate_error_gt_1deg": math.nan
                if not n
                else float(np.mean([err > 1.0 for err in errors])),
                "failure_rate_error_gt_3deg": math.nan
                if not n
                else float(np.mean([err > 3.0 for err in errors])),
                "mean_runtime_ms": mean_or_nan(finite_values(subset, "runtime_ms")),
                "mean_psnr_sync": mean_or_nan(finite_values(subset, "psnr_sync")),
                "mean_ssim_sync": mean_or_nan(finite_values(subset, "ssim_sync")),
                "mean_anchor_oracle_score": mean_or_nan(
                    finite_values(subset, "anchor_oracle_score")
                ),
                "mean_anchor_sync_score": mean_or_nan(
                    finite_values(subset, "anchor_sync_score")
                ),
                "mean_anchor_remove_score": mean_or_nan(
                    finite_values(subset, "anchor_remove_score")
                ),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, float]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_method_alpha_theta(
    rows: Sequence[Dict[str, float]], y_keys: Sequence[str]
) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, float, float], Dict[str, List[float]]] = {}
    for row in rows:
        key = (row["method"], float(row["alpha"]), float(row["theta_gt"]))
        grouped.setdefault(key, {y_key: [] for y_key in y_keys})
        for y_key in y_keys:
            value = float(row[y_key])
            if not math.isnan(value):
                grouped[key][y_key].append(value)
    out = []
    for (method, alpha, theta), values in sorted(grouped.items()):
        item = {"method": method, "alpha": alpha, "theta_gt": theta}
        for y_key in y_keys:
            item[y_key] = mean_or_nan(values[y_key])
        out.append(item)
    return out


def plot_angle_error_vs_theta(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    agg = aggregate_by_method_alpha_theta(rows, ["angle_error"])
    plt.figure(figsize=(7, 4))
    for method in sorted({row["method"] for row in agg}):
        for alpha in sorted({float(row["alpha"]) for row in agg if row["method"] == method}):
            subset = [
                row for row in agg if row["method"] == method and float(row["alpha"]) == alpha
            ]
            plt.plot(
                [float(row["theta_gt"]) for row in subset],
                [float(row["angle_error"]) for row in subset],
                marker="o",
                linewidth=1.5,
                label=f"{method}, alpha={alpha:g}",
            )
    plt.title("Angle error vs theta")
    plt.xlabel("theta_gt")
    plt.ylabel("absolute angular error (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_angle_error_box(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    labels = []
    data = []
    for method in sorted({row["method"] for row in rows}):
        for alpha in sorted({float(row["alpha"]) for row in rows if row["method"] == method}):
            labels.append(f"{method}\n{alpha:g}")
            data.append(
                [
                    float(row["angle_error"])
                    for row in rows
                    if row["method"] == method and float(row["alpha"]) == alpha
                ]
            )
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title("Angle error by alpha")
    plt.ylabel("absolute angular error (deg)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_runtime_by_method(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    summary = summarize(rows)
    labels = [f"{row['method']}\n{float(row['alpha']):g}" for row in summary]
    values = [float(row["mean_runtime_ms"]) for row in summary]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, values)
    plt.title("Runtime by method")
    plt.ylabel("mean detection runtime (ms)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_failure_rate(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    summary = summarize(rows)
    plt.figure(figsize=(6, 4))
    for method in sorted({row["method"] for row in summary}):
        subset = [row for row in summary if row["method"] == method]
        xs = [float(row["alpha"]) for row in subset]
        plt.plot(
            xs,
            [float(row["failure_rate_error_gt_1deg"]) for row in subset],
            marker="o",
            label=f"{method} >1deg",
        )
        plt.plot(
            xs,
            [float(row["failure_rate_error_gt_3deg"]) for row in subset],
            marker="s",
            linestyle="--",
            label=f"{method} >3deg",
        )
    plt.title("Failure rate vs alpha")
    plt.xlabel("alpha")
    plt.ylabel("failure rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_anchor_sync_vs_oracle(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    summary = summarize(rows)
    plt.figure(figsize=(6, 4))
    for method in sorted({row["method"] for row in summary}):
        subset = [row for row in summary if row["method"] == method]
        xs = [float(row["alpha"]) for row in subset]
        plt.plot(
            xs,
            [float(row["mean_anchor_oracle_score"]) for row in subset],
            marker="o",
            label=f"{method} oracle",
        )
        plt.plot(
            xs,
            [float(row["mean_anchor_sync_score"]) for row in subset],
            marker="s",
            linestyle="--",
            label=f"{method} sync",
        )
    plt.title("Anchor sync vs oracle by alpha")
    plt.xlabel("alpha")
    plt.ylabel("mean surrogate NCC")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_row(
    image_id,
    method,
    alpha,
    rmin,
    rmax,
    theta,
    theta_hat,
    best_score,
    runtime_ms,
    psnr_sync,
    ssim_sync,
    x_att,
    x_corr,
    x_corr_clean,
    delta,
    ambiguity_resolved=False,
    candidate_score_0=np.nan,
    candidate_score_180=np.nan,
    raw_shift=np.nan,
    theta_mod=np.nan,
    angle_bin=np.nan,
    corr_margin=np.nan,
    top2_score=np.nan,
):
    oracle_img = rotate_image_keep_size(x_att, -theta)
    return {
        "image_id": image_id,
        "method": method,
        "alpha": alpha,
        "rmin": rmin,
        "rmax": rmax,
        "theta_gt": theta,
        "theta_hat": theta_hat,
        "angle_error": angular_error(theta_hat, theta),
        "best_score": best_score,
        "runtime_ms": runtime_ms,
        "psnr_sync": psnr_sync,
        "ssim_sync": ssim_sync,
        "anchor_oracle_score": anchor_score(oracle_img, delta, rmin, rmax),
        "anchor_sync_score": anchor_score(x_corr, delta, rmin, rmax),
        "anchor_remove_score": anchor_score(x_corr_clean, delta, rmin, rmax),
        "raw_shift": raw_shift,
        "theta_mod": theta_mod,
        "angle_bin": angle_bin,
        "corr_margin": corr_margin,
        "top2_score": top2_score,
        "ambiguity_resolved": bool(ambiguity_resolved),
        "candidate_score_0": candidate_score_0,
        "candidate_score_180": candidate_score_180,
    }


def run_eval(args: argparse.Namespace) -> List[Dict[str, float]]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args)
    alphas = parse_float_list(args.alphas)
    angles = parse_float_list(args.angles)
    methods = [args.method]
    if args.compare_v0 and "v0" not in methods:
        methods.append("v0")

    rows: List[Dict[str, float]] = []
    selected_example = None
    fallback_example = None
    total = len(inputs) * len(alphas) * len(angles) * len(methods)
    done = 0

    for image_id, x_w in inputs:
        save_rgb(outdir / f"x_w_{image_id}.png", x_w)
        H, W = x_w.shape[:2]
        additive_delta, additive_polar_template, _additive_meta = make_polar_anchor_template(
            H,
            W,
            key=args.key,
            rmin=args.rmin,
            rmax=args.rmax,
            num_angles=args.num_angles,
            beta=args.beta,
        )
        magmod_grid, magmod_polar_template, _magmod_meta = make_polar_magnitude_template(
            H,
            W,
            key=args.key,
            rmin=args.rmin,
            rmax=args.rmax,
            num_angles=args.num_angles,
            beta=args.beta,
        )
        magmod_delta = spatial_proxy_from_modulation(magmod_grid)
        v0_delta, v0_mask = make_anchor_template(
            H, W, key=args.key, rmin=args.rmin, rmax=args.rmax
        )

        if len(inputs) == 1:
            default_delta = (
                magmod_delta
                if args.method == "v1_magmod"
                else additive_delta
                if args.method == "v1_additive"
                else v0_delta
            )
            save_delta_vis(outdir / "anchor_delta.png", default_delta)
        save_delta_vis(outdir / f"anchor_delta_{image_id}_v1_magmod.png", magmod_delta)
        save_delta_vis(outdir / f"anchor_delta_{image_id}_v1_additive.png", additive_delta)

        for alpha in alphas:
            per_method_sync = {
                "v1_magmod": embed_polar_magnitude_anchor_rgb(x_w, magmod_grid, alpha),
                "v1_additive": embed_anchor_rgb(x_w, additive_delta, alpha),
                "v0": embed_anchor_rgb(x_w, v0_delta, alpha),
            }
            for method in methods:
                x_sync = per_method_sync[method]
                if method == "v1_magmod":
                    delta = magmod_delta
                    method_polar_template = magmod_polar_template
                elif method == "v1_additive":
                    delta = additive_delta
                    method_polar_template = additive_polar_template
                else:
                    delta = v0_delta
                    method_polar_template = None
                psnr_sync = psnr(x_w, x_sync)
                ssim_sync = ssim(x_w, x_sync)
                save_rgb(outdir / f"x_sync_{image_id}_{method}_alpha_{alpha:g}.png", x_sync)

                for theta in angles:
                    x_att = rotate_image_keep_size(x_sync, theta)
                    start = time.perf_counter()
                    if method in {"v1_magmod", "v1_additive"}:
                        theta_hat, best_score, _curve, extra = detect_rotation_angle_polar(
                            x_att,
                            method_polar_template,
                            delta=delta,
                            rmin=args.rmin,
                            rmax=args.rmax,
                            num_r=args.num_r,
                            num_angles=args.num_angles,
                            resolve_ambiguity=not args.no_resolve_ambiguity,
                        )
                    else:
                        theta_hat, best_score, _curve = detect_rotation_angle(
                            x_att,
                            delta,
                            v0_mask,
                            coarse_step=args.v0_coarse_step,
                            fine_step=args.v0_fine_step,
                        )
                        extra = {
                            "raw_shift": np.nan,
                            "theta_mod": np.nan,
                            "angle_bin": np.nan,
                            "corr_margin": np.nan,
                            "top2_score": np.nan,
                            "ambiguity_resolved": False,
                            "candidate_score_0": np.nan,
                            "candidate_score_180": np.nan,
                        }
                    runtime_ms = (time.perf_counter() - start) * 1000.0

                    x_corr = rotate_image_keep_size(x_att, -theta_hat)
                    x_corr_clean, _alpha_hat = remove_anchor_rgb(x_corr, delta)
                    rows.append(
                        make_row(
                            image_id,
                            method,
                            alpha,
                            args.rmin,
                            args.rmax,
                            theta,
                            theta_hat,
                            best_score,
                            runtime_ms,
                            psnr_sync,
                            ssim_sync,
                            x_att,
                            x_corr,
                            x_corr_clean,
                            delta,
                            ambiguity_resolved=extra["ambiguity_resolved"],
                            candidate_score_0=extra["candidate_score_0"],
                            candidate_score_180=extra["candidate_score_180"],
                            raw_shift=extra["raw_shift"],
                            theta_mod=extra["theta_mod"],
                            angle_bin=extra["angle_bin"],
                            corr_margin=extra["corr_margin"],
                            top2_score=extra["top2_score"],
                        )
                    )

                    example_payload = (x_w, x_sync, x_att, x_corr, x_corr_clean, delta)
                    if fallback_example is None:
                        fallback_example = example_payload
                    if (
                        selected_example is None
                        and method == args.method
                        and abs(alpha - args.example_alpha) < 1e-12
                        and abs(theta - args.example_theta) < 1e-12
                    ):
                        selected_example = example_payload

                    done += 1
                    if not args.no_show_progress:
                        print(
                            f"[{done}/{total}] image={image_id} method={method} "
                            f"alpha={alpha:g} theta={theta:g} theta_hat={theta_hat:.2f} "
                            f"runtime_ms={runtime_ms:.1f}"
                        )

    write_csv(outdir / "polar_sync_results.csv", rows, CSV_FIELDS)
    write_csv(outdir / "summary.csv", summarize(rows), SUMMARY_FIELDS)
    plot_angle_error_vs_theta(outdir / "angle_error_vs_theta.png", rows)
    plot_angle_error_box(outdir / "angle_error_box_by_alpha.png", rows)
    plot_runtime_by_method(outdir / "runtime_by_method.png", rows)
    plot_failure_rate(outdir / "failure_rate_vs_alpha.png", rows)
    plot_anchor_sync_vs_oracle(outdir / "anchor_sync_vs_oracle_by_alpha.png", rows)

    example = selected_example or fallback_example
    if example is not None:
        x_w, x_sync, x_att, x_corr, x_corr_clean, delta = example
        delta_vis = (delta - delta.min()) / max(float(delta.max() - delta.min()), 1e-8)
        save_example_grid(
            outdir / "example_grid.png",
            [
                ("original watermarked", x_w),
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
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--outdir", default="polar_sync_eval")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--alphas", default="0.001,0.002,0.003,0.005")
    parser.add_argument("--angles", default="5,10,15,30,45,60,75,90,120,150,180")
    parser.add_argument("--rmin", type=float, default=0.12)
    parser.add_argument("--rmax", type=float, default=0.28)
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--example-alpha", type=float, default=0.003)
    parser.add_argument("--example-theta", type=float, default=45.0)
    parser.add_argument(
        "--method",
        choices=["v1_magmod", "v1_additive", "v0"],
        default="v1_magmod",
    )
    parser.add_argument("--compare-v0", action="store_true")
    parser.add_argument("--num-r", type=int, default=64)
    parser.add_argument("--num-angles", type=int, default=360)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--no-resolve-ambiguity", action="store_true")
    parser.add_argument("--v0-coarse-step", type=float, default=2.0)
    parser.add_argument("--v0-fine-step", type=float, default=0.25)
    parser.add_argument("--no-show-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
