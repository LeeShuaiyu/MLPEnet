#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlpenet.data import generate_dataset
from mlpenet.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OU SDE datasets for MLPEnet.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Dataset output directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    for split in args.splits:
        path = generate_dataset(config, split=split, output_dir=Path(args.output_dir), force=args.force)
        print(f"{split} dataset: {path}")


if __name__ == "__main__":
    main()
