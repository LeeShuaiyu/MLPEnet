#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlpenet.data import generate_dataset
from mlpenet.evaluate import evaluate
from mlpenet.train import train
from mlpenet.utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data, train, and evaluate one MLPEnet config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--work-dir", required=True, help="Experiment output directory.")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate when --skip-train is used.")
    args = parser.parse_args()

    config = load_config(args.config)
    work_dir = ensure_dir(Path(args.work_dir))
    data_dir = ensure_dir(work_dir / "data")
    run_dir = ensure_dir(work_dir / "run")
    eval_dir = ensure_dir(work_dir / "evaluation")

    for split in ["train", "eval", "test"]:
        generate_dataset(config, split=split, output_dir=data_dir, force=args.force_data)

    if args.skip_train:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required with --skip-train")
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint = train(config, data_dir=data_dir, run_dir=run_dir)
    evaluate(config, data_dir=data_dir, checkpoint=checkpoint, output_dir=eval_dir, split="test")


if __name__ == "__main__":
    main()
