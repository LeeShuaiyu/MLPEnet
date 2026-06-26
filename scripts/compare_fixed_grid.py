#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPORTED_VALUES = {
    ("eta=1.5;epsilon=0.03;alpha=1.7", "eta"): (1.588, 0.408, 0.282),
    ("eta=1.5;epsilon=0.03;alpha=1.7", "epsilon"): (0.03017, 0.002637, 0.00199),
    ("eta=1.5;epsilon=0.03;alpha=1.7", "alpha"): (1.692, 0.0838, 0.0666),
    ("eta=3.5;epsilon=0.03;alpha=1.7", "eta"): (3.518, 0.458, 0.351),
    ("eta=3.5;epsilon=0.03;alpha=1.7", "epsilon"): (0.03004, 0.00261, 0.00197),
    ("eta=3.5;epsilon=0.03;alpha=1.7", "alpha"): (1.706, 0.0742, 0.0592),
    ("eta=2.5;epsilon=0.02;alpha=1.1", "eta"): (2.569, 0.466, 0.341),
    ("eta=2.5;epsilon=0.02;alpha=1.1", "epsilon"): (0.02017, 0.00167, 0.00128),
    ("eta=2.5;epsilon=0.02;alpha=1.1", "alpha"): (1.276, 0.055, 0.177),
    ("eta=2;epsilon=0.04;alpha=1.7", "eta"): (2.085, 0.453, 0.329),
    ("eta=2;epsilon=0.04;alpha=1.7", "epsilon"): (0.04005, 0.00259, 0.00194),
    ("eta=2;epsilon=0.04;alpha=1.7", "alpha"): (1.703, 0.074, 0.060),
    ("eta=2.5;epsilon=0.03;alpha=1.5", "eta"): (2.572, 0.479, 0.351),
    ("eta=2.5;epsilon=0.03;alpha=1.5", "epsilon"): (0.03005, 0.00269, 0.00199),
    ("eta=2.5;epsilon=0.03;alpha=1.5", "alpha"): (1.503, 0.081, 0.064),
    ("eta=2.5;epsilon=0.01;alpha=1.8", "eta"): (2.581, 0.483, 0.358),
    ("eta=2.5;epsilon=0.01;alpha=1.8", "epsilon"): (0.01025, 0.00086, 0.00068),
    ("eta=2.5;epsilon=0.01;alpha=1.8", "alpha"): (1.784, 0.075, 0.060),
}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def compare_fixed_grid(grouped_metrics_csv: str | Path, csv_out: str | Path | None = None) -> list[dict[str, object]]:
    grouped_path = Path(grouped_metrics_csv)
    with open(grouped_path, newline="", encoding="utf-8") as f:
        observed = {(row["condition"], row["parameter"]): row for row in csv.DictReader(f)}

    rows = []
    for key, (reported_mean, reported_sd, reported_mae) in REPORTED_VALUES.items():
        condition, parameter = key
        obs = observed.get(key)
        if obs is None:
            rows.append(
                {
                    "condition": condition,
                    "parameter": parameter,
                    "reported_mean": reported_mean,
                    "reproduced_mean": "",
                    "mean_delta": "",
                    "reported_sd": reported_sd,
                    "reproduced_sd": "",
                    "reported_mae": reported_mae,
                    "reproduced_mae": "",
                    "mae_ratio": "",
                    "n": "",
                }
            )
            continue
        reproduced_mean = _float(obs, "pred_mean")
        reproduced_sd = _float(obs, "pred_sd")
        reproduced_mae = _float(obs, "mae")
        rows.append(
            {
                "condition": condition,
                "parameter": parameter,
                "reported_mean": reported_mean,
                "reproduced_mean": reproduced_mean,
                "mean_delta": reproduced_mean - reported_mean,
                "reported_sd": reported_sd,
                "reproduced_sd": reproduced_sd,
                "reported_mae": reported_mae,
                "reproduced_mae": reproduced_mae,
                "mae_ratio": reproduced_mae / reported_mae if reported_mae else "",
                "n": obs["n"],
            }
        )

    fieldnames = [
        "condition",
        "parameter",
        "reported_mean",
        "reproduced_mean",
        "mean_delta",
        "reported_sd",
        "reproduced_sd",
        "reported_mae",
        "reproduced_mae",
        "mae_ratio",
        "n",
    ]
    if csv_out:
        out_path = Path(csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare alpha-stable OU fixed-grid grouped metrics with the manuscript values."
    )
    parser.add_argument("grouped_metrics_csv", help="Path to test_grouped_metrics.csv")
    parser.add_argument("--csv-out", help="Optional CSV path for the comparison output.")
    args = parser.parse_args()

    rows = compare_fixed_grid(args.grouped_metrics_csv, csv_out=args.csv_out)

    print("condition,parameter,reported_mae,reproduced_mae,mae_ratio,reported_mean,reproduced_mean,mean_delta,n")
    for row in rows:
        print(
            f"{row['condition']},{row['parameter']},"
            f"{row['reported_mae']},{row['reproduced_mae']},{row['mae_ratio']},"
            f"{row['reported_mean']},{row['reproduced_mean']},{row['mean_delta']},{row['n']}"
        )


if __name__ == "__main__":
    main()
