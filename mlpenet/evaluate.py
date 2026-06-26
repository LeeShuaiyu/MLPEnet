from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .train import build_model
from .utils import device_from_config, ensure_dir, save_json


def _metrics(parameter_names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    for idx, name in enumerate(parameter_names):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        err = pred - true
        rows.append(
            {
                "parameter": name,
                "target_mean": float(np.mean(true)),
                "pred_mean": float(np.mean(pred)),
                "pred_sd": float(np.std(pred, ddof=1)) if len(pred) > 1 else 0.0,
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
        )
    return rows


def _grouped_metrics(parameter_names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str]]:
    rounded = np.round(y_true.astype(np.float64), decimals=6)
    unique_rows, inverse = np.unique(rounded, axis=0, return_inverse=True)
    rows = []
    for group_idx, values in enumerate(unique_rows):
        mask = inverse == group_idx
        if int(mask.sum()) < 2:
            continue
        condition = ";".join(f"{name}={values[idx]:.6g}" for idx, name in enumerate(parameter_names))
        for param_idx, name in enumerate(parameter_names):
            pred = y_pred[mask, param_idx]
            true = y_true[mask, param_idx]
            err = pred - true
            rows.append(
                {
                    "condition": condition,
                    "parameter": name,
                    "true_value": float(values[param_idx]),
                    "pred_mean": float(np.mean(pred)),
                    "pred_sd": float(np.std(pred, ddof=1)),
                    "bias": float(np.mean(err)),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "n": int(mask.sum()),
                }
            )
    return rows


def evaluate(
    config: Mapping[str, Any],
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "test",
) -> Path:
    output_dir = ensure_dir(output_dir)
    eval_cfg = config.get("evaluation", {})
    device = device_from_config(eval_cfg.get("device", config.get("training", {}).get("device", "auto")))
    dataset = TrajectoryDataset(Path(data_dir) / split)
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 2048)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt.get("config", config)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y, lengths, h in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            h = h.to(device)
            pred = model(x, lengths, h).cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.numpy())
    y_true_np = np.vstack(y_true)
    y_pred_np = np.vstack(y_pred)

    np.savez_compressed(output_dir / f"{split}_predictions.npz", y_true=y_true_np, y_pred=y_pred_np)
    parameter_names = list(ckpt.get("parameter_names", dataset.parameter_names))
    rows = _metrics(parameter_names, y_true_np, y_pred_np)
    metrics_path = output_dir / f"{split}_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    grouped_rows = _grouped_metrics(parameter_names, y_true_np, y_pred_np)
    if grouped_rows:
        grouped_path = output_dir / f"{split}_grouped_metrics.csv"
        with open(grouped_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(grouped_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grouped_rows)
    save_json(output_dir / f"{split}_metrics.json", {"rows": rows})
    for row in rows:
        print(
            f"{row['parameter']}: pred={row['pred_mean']:.6g}±{row['pred_sd']:.6g} "
            f"bias={row['bias']:.6g} mae={row['mae']:.6g}",
            flush=True,
        )
    return metrics_path
