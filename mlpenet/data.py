from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import ensure_dir, save_json


SPLIT_OFFSETS = {"train": 0, "eval": 100_000, "test": 200_000}


def _uniform(rng: np.random.Generator, bounds: list[float], size: int) -> np.ndarray:
    return rng.uniform(float(bounds[0]), float(bounds[1]), size=size).astype(np.float32)


def _integers(rng: np.random.Generator, bounds: list[int], size: int) -> np.ndarray:
    return rng.integers(int(bounds[0]), int(bounds[1]) + 1, size=size, endpoint=False).astype(np.int32)


def symmetric_alpha_stable(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    """Sample S_alpha(1) with characteristic function exp(-|u|^alpha)."""
    alpha = alpha.astype(np.float64)
    out = np.empty_like(alpha, dtype=np.float64)
    near_gaussian = np.isclose(alpha, 2.0, atol=1e-6)
    if np.any(near_gaussian):
        out[near_gaussian] = rng.normal(0.0, math.sqrt(2.0), size=int(near_gaussian.sum()))
    mask = ~near_gaussian
    if np.any(mask):
        a = alpha[mask]
        u = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=a.shape[0])
        w = rng.exponential(1.0, size=a.shape[0])
        numerator = np.sin(a * u)
        denominator = np.power(np.cos(u), 1.0 / a)
        factor = np.power(np.cos((1.0 - a) * u) / w, (1.0 - a) / a)
        out[mask] = numerator / denominator * factor
    return out.astype(np.float32)


def student_levy_increment(
    rng: np.random.Generator,
    h: np.ndarray,
    nu: np.ndarray,
    method: str = "cauchy_approx",
    clip: float | None = 1_000.0,
) -> np.ndarray:
    """Sample increments for the Student-Levy driver.

    The paper uses characteristic-function rejection sampling for the Student-Levy
    process. This implementation exposes a documented high-frequency approximation
    that is fast enough for neural-network data generation. It is intentionally named
    in metadata so Student-Levy results can be interpreted honestly if the original
    CF-RS generator/checkpoints are unavailable.
    """
    if method == "cauchy_approx":
        z = rng.standard_cauchy(size=h.shape[0]).astype(np.float32)
        inc = h.astype(np.float32) * z
    elif method == "scaled_student":
        z = rng.standard_t(df=nu.astype(np.float64)).astype(np.float32)
        inc = np.sqrt(h.astype(np.float32)) * z
    else:
        raise ValueError(f"Unknown Student-Levy increment method: {method}")
    if clip is not None:
        inc = np.clip(inc, -float(clip), float(clip))
    return inc.astype(np.float32)


def _sample_parameters(
    rng: np.random.Generator,
    parameter_ranges: Mapping[str, list[float]],
    count: int,
    fixed: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    values = {}
    fixed = fixed or {}
    for name, bounds in parameter_ranges.items():
        if name in fixed:
            values[name] = np.full(count, float(fixed[name]), dtype=np.float32)
        else:
            values[name] = _uniform(rng, bounds, count)
    return values


def _simulate_ou_batch(
    rng: np.random.Generator,
    parameters: Mapping[str, np.ndarray],
    n_steps: np.ndarray,
    span_t: np.ndarray,
    max_n: int,
    noise_type: str,
    burnin_time: float = 0.0,
    x0: float = 0.0,
    student_method: str = "cauchy_approx",
    student_clip: float | None = 1_000.0,
) -> np.ndarray:
    batch = int(n_steps.shape[0])
    eta = parameters["eta"].astype(np.float32)
    epsilon = parameters["epsilon"].astype(np.float32)
    dt = (span_t / n_steps).astype(np.float32)
    burn_steps = np.ceil(float(burnin_time) / dt).astype(np.int32) if burnin_time > 0 else np.zeros(batch, dtype=np.int32)
    total_steps = burn_steps + n_steps.astype(np.int32)
    max_total = int(total_steps.max())
    state = np.full(batch, float(x0), dtype=np.float32)
    out = np.zeros((batch, max_n, 1), dtype=np.float32)

    for step in range(max_total):
        active = step < total_steps
        if not np.any(active):
            continue
        idx = np.flatnonzero(active)
        local_dt = dt[idx]
        if noise_type == "gaussian":
            d_noise = np.sqrt(local_dt) * rng.normal(0.0, 1.0, size=idx.shape[0]).astype(np.float32)
        elif noise_type == "alpha_stable":
            alpha = parameters["alpha"][idx]
            d_noise = np.power(local_dt, 1.0 / alpha).astype(np.float32) * symmetric_alpha_stable(rng, alpha)
        elif noise_type == "student_levy":
            nu = parameters["nu"][idx]
            d_noise = student_levy_increment(rng, local_dt, nu, method=student_method, clip=student_clip)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        drift = -eta[idx] * state[idx] * local_dt
        state[idx] = state[idx] + drift + epsilon[idx] * d_noise

        observed = active & (step >= burn_steps)
        if np.any(observed):
            obs_idx = step - burn_steps[observed]
            rows = np.flatnonzero(observed)
            valid = obs_idx < max_n
            out[rows[valid], obs_idx[valid], 0] = state[rows[valid]]
    return out


def _test_grid_rows(config: Mapping[str, Any]) -> list[dict[str, float]]:
    grid = config.get("data", {}).get("test_grid")
    if not grid:
        return []
    rows = []
    for row in grid:
        rows.append({k: float(v) for k, v in row.items()})
    return rows


def generate_dataset(config: Mapping[str, Any], split: str, output_dir: str | Path, force: bool = False) -> Path:
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"Unknown split {split}")
    data_cfg = config["data"]
    exp_cfg = config["experiment"]
    output_dir = ensure_dir(Path(output_dir) / split)
    meta_path = output_dir / "meta.json"
    if meta_path.exists() and not force:
        print(f"[data] using existing {split} dataset at {output_dir}", flush=True)
        return output_dir

    seed = int(config.get("seed", 0)) + SPLIT_OFFSETS[split]
    rng = np.random.default_rng(seed)
    parameter_ranges = exp_cfg["parameters"]
    parameter_names = list(parameter_ranges.keys())
    max_n = int(data_cfg["N"][1])
    chunk_size = int(data_cfg.get("chunk_size", 512))

    fixed_rows = _test_grid_rows(config) if split == "test" else []
    if fixed_rows:
        samples_per_point = int(data_cfg.get("test_samples_per_point", 1000))
        sample_count = samples_per_point * len(fixed_rows)
    else:
        sample_count = int(data_cfg["samples"][split])

    auxiliary_input = str(data_cfg.get("auxiliary_input", "span_t"))
    if auxiliary_input not in {"span_t", "dt"}:
        raise ValueError("data.auxiliary_input must be 'span_t' or 'dt'")

    x_path = output_dir / "x.npy"
    y_path = output_dir / "y.npy"
    h_path = output_dir / "h.npy"
    lengths_path = output_dir / "lengths.npy"
    x_mm = np.lib.format.open_memmap(x_path, mode="w+", dtype="float32", shape=(sample_count, max_n, 1))
    y_mm = np.lib.format.open_memmap(y_path, mode="w+", dtype="float32", shape=(sample_count, len(parameter_names)))
    h_mm = np.lib.format.open_memmap(h_path, mode="w+", dtype="float32", shape=(sample_count, 1))
    lengths_mm = np.lib.format.open_memmap(lengths_path, mode="w+", dtype="int32", shape=(sample_count,))

    def write_block(start: int, count: int, fixed: Mapping[str, float] | None = None) -> None:
        n_steps = _integers(rng, data_cfg["N"], count)
        span_t = _uniform(rng, data_cfg["T"], count)
        params = _sample_parameters(rng, parameter_ranges, count, fixed=fixed)
        x = _simulate_ou_batch(
            rng=rng,
            parameters=params,
            n_steps=n_steps,
            span_t=span_t,
            max_n=max_n,
            noise_type=exp_cfg["noise"],
            burnin_time=float(data_cfg.get("burnin_time", 0.0)),
            student_method=str(data_cfg.get("student_levy_method", "cauchy_approx")),
            student_clip=data_cfg.get("student_levy_clip", 1_000.0),
        )
        y = np.column_stack([params[name] for name in parameter_names]).astype(np.float32)
        x_mm[start : start + count] = x
        y_mm[start : start + count] = y
        if auxiliary_input == "span_t":
            h_mm[start : start + count, 0] = span_t.astype(np.float32)
        else:
            h_mm[start : start + count, 0] = (span_t / n_steps).astype(np.float32)
        lengths_mm[start : start + count] = n_steps

    if fixed_rows:
        cursor = 0
        for fixed in fixed_rows:
            remaining = samples_per_point
            while remaining:
                count = min(chunk_size, remaining)
                write_block(cursor, count, fixed=fixed)
                cursor += count
                remaining -= count
                print(f"[data] {split}: {cursor}/{sample_count} samples", flush=True)
    else:
        cursor = 0
        while cursor < sample_count:
            count = min(chunk_size, sample_count - cursor)
            write_block(cursor, count)
            cursor += count
            print(f"[data] {split}: {cursor}/{sample_count} samples", flush=True)

    for mm in (x_mm, y_mm, h_mm, lengths_mm):
        mm.flush()

    meta = {
        "split": split,
        "seed": seed,
        "sample_count": sample_count,
        "max_n": max_n,
        "parameter_names": parameter_names,
        "parameter_ranges": parameter_ranges,
        "noise": exp_cfg["noise"],
        "data": data_cfg,
        "auxiliary_input": auxiliary_input,
    }
    save_json(meta_path, meta)
    print(f"[data] wrote {split} dataset to {output_dir}", flush=True)
    return output_dir


class TrajectoryDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.x = np.load(self.path / "x.npy", mmap_mode="r")
        self.y = np.load(self.path / "y.npy", mmap_mode="r")
        self.h = np.load(self.path / "h.npy", mmap_mode="r")
        self.lengths = np.load(self.path / "lengths.npy", mmap_mode="r")

    @property
    def parameter_names(self) -> list[str]:
        return list(self.meta["parameter_names"])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(np.asarray(self.x[index], dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(self.y[index], dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(self.lengths[index], dtype=np.float32).reshape(1).copy()),
            torch.from_numpy(np.asarray(self.h[index], dtype=np.float32).reshape(1).copy()),
        )
