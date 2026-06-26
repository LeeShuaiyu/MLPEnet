#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlpenet.train import train
from mlpenet.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLPEnet.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-dir", required=True, help="Directory containing train/eval/test splits.")
    parser.add_argument("--run-dir", required=True, help="Run output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt = train(config, data_dir=Path(args.data_dir), run_dir=Path(args.run_dir))
    print(f"best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
