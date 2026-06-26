#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlpenet.evaluate import evaluate
from mlpenet.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MLPEnet.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-dir", required=True, help="Directory containing dataset splits.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = evaluate(
        config,
        data_dir=Path(args.data_dir),
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        split=args.split,
    )
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
