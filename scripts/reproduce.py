#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_fixed_grid import compare_fixed_grid
from mlpenet.data import generate_dataset
from mlpenet.evaluate import evaluate
from mlpenet.utils import ensure_dir, load_config, save_json


DEFAULT_CONFIG = Path("configs/alpha_ou.yaml")
DEFAULT_CHECKPOINT = Path("checkpoints/alpha_ou.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/repro")
PARAMETER_LABELS = {"eta": r"\eta", "epsilon": r"\epsilon", "alpha": r"\alpha"}


def _label(name: str) -> str:
    return PARAMETER_LABELS.get(name, name)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _format_float(value: str, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except ValueError:
        return value


def _parameter_names(config: dict[str, Any]) -> list[str]:
    return list(config["experiment"]["parameters"].keys())


def _axis_limits(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([true, pred]).astype(float)
    lo = float(np.nanpercentile(values, 0.5))
    hi = float(np.nanpercentile(values, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    pad = max((hi - lo) * 0.08, 1e-6)
    return lo - pad, hi + pad


def _paper_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=3)


def _binned_curves(true: np.ndarray, pred: np.ndarray, bins: int = 70) -> dict[str, np.ndarray]:
    edges = np.linspace(float(np.min(true)), float(np.max(true)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for lo, hi, center in zip(edges[:-1], edges[1:], centers):
        mask = (true >= lo) & (true < hi)
        if hi == edges[-1]:
            mask = (true >= lo) & (true <= hi)
        if int(mask.sum()) < 5:
            continue
        values = pred[mask]
        rows.append(
            (
                center,
                float(np.mean(values)),
                float(np.median(values)),
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.25)),
                float(np.quantile(values, 0.75)),
                float(np.quantile(values, 0.95)),
            )
        )
    if not rows:
        return {key: np.asarray([], dtype=float) for key in ["x", "mean", "median", "q05", "q25", "q75", "q95"]}
    arr = np.asarray(rows, dtype=float)
    return {
        "x": arr[:, 0],
        "mean": arr[:, 1],
        "median": arr[:, 2],
        "q05": arr[:, 3],
        "q25": arr[:, 4],
        "q75": arr[:, 5],
        "q95": arr[:, 6],
    }


def make_prediction_figure(predictions_npz: Path, parameter_names: list[str], output_path: Path) -> None:
    data = np.load(predictions_npz)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    fig, axes = plt.subplots(1, len(parameter_names), figsize=(10.4, 3.6), dpi=180)
    if len(parameter_names) == 1:
        axes = [axes]
    for idx, (ax, name) in enumerate(zip(axes, parameter_names)):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        lo, hi = _axis_limits(true, pred)
        curves = _binned_curves(true, pred)
        ax.scatter(true, pred, s=3, alpha=0.75, linewidths=0, color="red", label="Samples")
        if curves["x"].size:
            ax.plot(curves["x"], curves["mean"], color="navy", linewidth=1.4, label="Mean")
            ax.plot(curves["x"], curves["median"], color="black", linewidth=1.1, linestyle="--", label="Median")
            ax.fill_between(
                curves["x"],
                curves["q25"],
                curves["q75"],
                color="#55a868",
                alpha=0.45,
                label="[0.25, 0.75]",
            )
            ax.fill_between(
                curves["x"],
                curves["q05"],
                curves["q95"],
                color="#ffb24d",
                alpha=0.42,
                label="[0.05, 0.95]",
            )
        ax.plot([lo, hi], [lo, hi], color="blue", linewidth=1.5, label="Ideal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"({chr(97 + idx)}) ${_label(name)}$", fontsize=14, pad=4)
        ax.set_xlabel("True values", fontsize=13)
        ax.set_ylabel("Estimated values", fontsize=13)
        _paper_axes(ax)
        ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.85)
    fig.tight_layout(w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _smooth_counts(counts: np.ndarray) -> np.ndarray:
    kernel_x = np.linspace(-2.5, 2.5, 13)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel = kernel / kernel.sum()
    return np.convolve(counts, kernel, mode="same")


def make_residual_figure(predictions_npz: Path, parameter_names: list[str], output_path: Path) -> None:
    data = np.load(predictions_npz)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    fig, axes = plt.subplots(1, len(parameter_names), figsize=(10.2, 3.7), dpi=180)
    if len(parameter_names) == 1:
        axes = [axes]
    for idx, (ax, name) in enumerate(zip(axes, parameter_names)):
        residual = y_pred[:, idx] - y_true[:, idx]
        residual_mean = float(np.mean(residual))
        residual_std = float(np.std(residual, ddof=1))
        standardized = residual / residual_std if residual_std > 0 else residual
        lo = float(np.nanpercentile(standardized, 0.2))
        hi = float(np.nanpercentile(standardized, 99.8))
        pad = max((hi - lo) * 0.08, 0.5)
        hist_range = (lo - pad, hi + pad)
        counts, edges, _ = ax.hist(
            standardized,
            bins=62,
            range=hist_range,
            color="#b76db1",
            alpha=0.72,
            edgecolor="#5a2a64",
            linewidth=0.35,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            _smooth_counts(counts),
            color="purple",
            linewidth=1.4,
            label=f"Mean: {residual_mean:.2e}, Std: {residual_std:.2e}",
        )
        ax.set_title(f"({chr(97 + idx)}) ${_label(name)}$", fontsize=13, pad=4)
        ax.set_xlabel("Standardized Error", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        _paper_axes(ax)
        ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9)
    fig.tight_layout(w_pad=2.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    sample_count: int,
    metrics_path: Path,
    prediction_figure_path: Path,
    residual_figure_path: Path,
    fixed_grid_comparison_path: Path | None,
) -> None:
    rows = _read_metrics(metrics_path)
    lines = [
        "# Alpha-Stable OU Checkpoint Reproduction",
        "",
        "This run regenerates evaluation data, loads the provided checkpoint, runs inference, and writes diagnostic figures.",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Random evaluation paths: `{sample_count}`",
        f"- Metrics CSV: `{metrics_path}`",
        f"- Prediction scatter: `{prediction_figure_path}`",
        f"- Standardized residuals: `{residual_figure_path}`",
    ]
    if fixed_grid_comparison_path is not None:
        lines.append(f"- Fixed-grid comparison CSV: `{fixed_grid_comparison_path}`")
    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            "| parameter | pred mean | pred sd | bias | MAE | RMSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {parameter} | {pred_mean} | {pred_sd} | {bias} | {mae} | {rmse} |".format(
                parameter=row["parameter"],
                pred_mean=_format_float(row["pred_mean"]),
                pred_sd=_format_float(row["pred_sd"]),
                bias=_format_float(row["bias"]),
                mae=_format_float(row["mae"]),
                rmse=_format_float(row["rmse"]),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_eval_config(config_path: Path, sample_count: int, seed: int | None, device: str | None) -> dict[str, Any]:
    config = deepcopy(load_config(config_path))
    config.setdefault("data", {}).setdefault("samples", {})["eval"] = int(sample_count)
    if seed is not None:
        config["seed"] = int(seed)
    if device is not None:
        config.setdefault("evaluation", {})["device"] = device
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate alpha-stable OU evaluation data, load a checkpoint, evaluate it, and create diagnostic figures."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Alpha-stable OU YAML config.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint to load.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of random evaluation paths.")
    parser.add_argument("--seed", type=int, help="Optional seed override for regenerated eval data.")
    parser.add_argument("--device", help="Optional evaluation device, for example cuda or cpu.")
    parser.add_argument("--force-data", action="store_true", help="Regenerate data even if output data already exists.")
    parser.add_argument(
        "--fixed-grid",
        action="store_true",
        help="Also evaluate the manuscript fixed test grid and compare with the reported values.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = ensure_dir(Path(args.output_dir))
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Place the released checkpoint there or pass --checkpoint /path/to/best.pt."
        )

    config = build_eval_config(config_path, sample_count=args.samples, seed=args.seed, device=args.device)
    data_dir = ensure_dir(output_dir / "data")
    eval_dir = ensure_dir(output_dir / "evaluation")
    figure_dir = ensure_dir(output_dir / "figures")

    save_json(
        output_dir / "reproduction_manifest.json",
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "random_eval_samples": int(args.samples),
            "fixed_grid": bool(args.fixed_grid),
        },
    )

    print(f"[1/4] generating {args.samples} alpha-stable OU eval paths")
    generate_dataset(config, split="eval", output_dir=data_dir, force=args.force_data)

    print("[2/4] loading checkpoint and running inference")
    metrics_path = evaluate(config, data_dir=data_dir, checkpoint=checkpoint_path, output_dir=eval_dir, split="eval")
    predictions_path = eval_dir / "eval_predictions.npz"

    print("[3/4] writing diagnostic figures")
    parameter_names = _parameter_names(config)
    prediction_figure_path = figure_dir / "predictions.png"
    residual_figure_path = figure_dir / "residuals.png"
    make_prediction_figure(predictions_path, parameter_names, prediction_figure_path)
    make_residual_figure(predictions_path, parameter_names, residual_figure_path)

    fixed_grid_comparison_path = None
    if args.fixed_grid:
        print("[fixed-grid] generating manuscript fixed test grid and comparing with reported values")
        generate_dataset(config, split="test", output_dir=data_dir, force=args.force_data)
        fixed_grid_dir = ensure_dir(output_dir / "fixed_grid")
        evaluate(config, data_dir=data_dir, checkpoint=checkpoint_path, output_dir=fixed_grid_dir, split="test")
        fixed_grid_comparison_path = fixed_grid_dir / "comparison.csv"
        compare_fixed_grid(
            fixed_grid_dir / "test_grouped_metrics.csv",
            csv_out=fixed_grid_comparison_path,
        )

    print("[4/4] writing summary")
    write_summary(
        output_dir / "summary.md",
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        sample_count=int(args.samples),
        metrics_path=metrics_path,
        prediction_figure_path=prediction_figure_path,
        residual_figure_path=residual_figure_path,
        fixed_grid_comparison_path=fixed_grid_comparison_path,
    )
    print(f"done: {output_dir}")


if __name__ == "__main__":
    main()
