# Experiment Notes for MLPEnet

This repository provides the public code package for:

**Multi-level PEnet: A Robust Three-Stage Model for Parameter Estimation in Non-Gaussian Noise-Driven Stochastic Differential Equations**

The package contains config-driven data generation, model training, checkpoint
evaluation, fixed-grid evaluation, and figure generation for the alpha-stable OU
experiment.

## Implementation Contents

- Multi-parameter output: the model now estimates all configured parameters jointly.
- Weighted L1 training loss for parameter-scale balancing.
- Config-driven train/eval/test data generation.
- Auxiliary input uses `dt = T / N`.
- Long-sequence support: datasets are stored as memory-mapped `.npy` arrays.
- Fixed test grids: evaluation supports the parameter combinations reported in the paper.
- Audit trail: each run saves config, environment, commit hash, history, checkpoints, predictions, and metrics.
- Smoke tests: small configs verify the Gaussian sanity path and the alpha-stable
  evaluation path end-to-end.

## Public Release Scope

The primary public experiment is the **alpha-stable OU** case. The alpha-stable
increments are generated directly with the Chambers-Mallows-Stuck method.

Gaussian OU is included as a sanity check. Student-Levy experiments are not part
of this public release.

## Data Generation Details

The alpha-stable OU reproduction config uses the manuscript ranges for `N`, `T`,
`eta`, `epsilon`, and `alpha`, and simulates alpha-stable increments with the
Chambers-Mallows-Stuck method. The auxiliary scalar passed to the network is
`h = dt = T / N`.

The released config also uses `burnin_time: 10.0` before collecting each simulated
trajectory, following the transient-discard convention used in the PENN-derived OU
data workflow.

## Quick Smoke Test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_experiment.py \
  --config configs/smoke_gaussian_ou.yaml \
  --work-dir runs/smoke_gaussian \
  --force-data

python scripts/run_experiment.py \
  --config configs/smoke_alpha_ou.yaml \
  --work-dir runs/smoke_alpha \
  --force-data
```

Smoke tests are not expected to match paper numbers. They only verify the Gaussian
sanity path and the alpha-stable reproduction path.

## Paper-Scale Runs

Primary paper-scale config:

- `configs/alpha_ou.yaml`

## Checkpoint-Only Reproduction

For checkpoint evaluation, run:

```bash
python scripts/reproduce.py \
  --checkpoint checkpoints/alpha_ou.pt \
  --output-dir outputs/repro \
  --fixed-grid
```

The script regenerates 5,000 random alpha-stable OU evaluation paths, loads the
released checkpoint, writes `eval_metrics.csv`, stores predictions, and generates:

- `figures/predictions.png`
- `figures/residuals.png`
- `summary.md`

To regenerate an existing output dataset, add:

```bash
--force-data
```

Supporting/sanity configs:

- `configs/gaussian_ou.yaml`

Example:

```bash
python scripts/run_experiment.py \
  --config configs/alpha_ou.yaml \
  --work-dir runs/alpha \
  --force-data
```

Outputs:

- `runs/<name>/data/{train,eval,test}/`
- `runs/<name>/run/history.csv`
- `runs/<name>/run/checkpoints/best.pt`
- `runs/<name>/evaluation/test_metrics.csv`
- `runs/<name>/evaluation/test_grouped_metrics.csv`

The grouped metrics file contains the fixed-grid means, standard deviations,
biases, MAEs, and RMSEs for the manuscript parameter settings.

## Cloud Run Checklist

1. Clone this branch on the GPU server.
2. Create a clean Python environment.
3. Install `requirements.txt`.
4. Run the smoke configs first.
5. Start paper-scale runs inside `tmux`.
6. Save run directories before publishing or pruning checkpoints.

See [docs/PENN_REFERENCE.md](docs/PENN_REFERENCE.md) for compatibility notes with
the upstream PENN codebase and [docs/CLOUD_RUN.md](docs/CLOUD_RUN.md) for a more
detailed GPU-server runbook.
