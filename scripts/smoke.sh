#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" scripts/run_experiment.py \
  --config configs/smoke_gaussian_ou.yaml \
  --work-dir runs/smoke_gaussian \
  --force-data

"${PYTHON_BIN}" scripts/run_experiment.py \
  --config configs/smoke_alpha_ou.yaml \
  --work-dir runs/smoke_alpha \
  --force-data
