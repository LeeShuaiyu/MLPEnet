#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
MODE="${1:-pilot}"
FORCE_DATA="${FORCE_DATA:-1}"

case "${MODE}" in
  smoke)
    CONFIG="configs/smoke_alpha_ou.yaml"
    WORK_DIR="runs/smoke_alpha"
    ;;
  pilot)
    CONFIG="configs/pilot_alpha_ou.yaml"
    WORK_DIR="runs/pilot_alpha"
    ;;
  full)
    CONFIG="configs/alpha_ou.yaml"
    WORK_DIR="runs/alpha"
    ;;
  *)
    echo "Usage: PYTHON=/path/to/python bash scripts/run_cloud.sh [smoke|pilot|full]" >&2
    exit 2
    ;;
esac

mkdir -p runs
ARGS=(
  scripts/run_experiment.py
  --config "${CONFIG}"
  --work-dir "${WORK_DIR}"
)

if [[ "${FORCE_DATA}" != "0" ]]; then
  ARGS+=(--force-data)
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${WORK_DIR}.log"
