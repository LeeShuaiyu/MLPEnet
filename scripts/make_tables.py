#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print grouped metrics in a compact table-like form.")
    parser.add_argument("grouped_metrics_csv", help="Path to test_grouped_metrics.csv.")
    args = parser.parse_args()

    path = Path(args.grouped_metrics_csv)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        print(
            f"{row['condition']} | {row['parameter']} true={float(row['true_value']):.6g} "
            f"pred={float(row['pred_mean']):.6g}±{float(row['pred_sd']):.6g} "
            f"MAE={float(row['mae']):.6g} n={row['n']}"
        )


if __name__ == "__main__":
    main()
