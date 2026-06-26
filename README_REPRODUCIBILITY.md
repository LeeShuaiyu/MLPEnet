# Reproducibility Notes for MLPEnet

This branch is a repaired reproducibility implementation for:

**Multi-level PEnet: A Robust Three-Stage Model for Parameter Estimation in Non-Gaussian Noise-Driven Stochastic Differential Equations**

The original repository was an early implementation snapshot. It did not archive the
exact training seeds, final checkpoints, generated datasets, or table-generation scripts
used for the manuscript. This branch therefore aims to reproduce the experimental
pipeline and reported trends as closely as possible with explicit configs and logs.

## What Was Repaired

- Multi-parameter output: the model now estimates all configured parameters jointly.
- Weighted L1 training loss: matches the manuscript description.
- Reproducible data generation: train/eval/test splits are generated from YAML configs.
- Auxiliary input uses `dt = T / N` in manuscript reproduction configs.
- Long-sequence support: datasets are stored as memory-mapped `.npy` arrays.
- Fixed test grids: evaluation can reproduce the table-style parameter combinations.
- Audit trail: each run saves config, environment, commit hash, history, checkpoints, predictions, and metrics.
- Smoke tests: small configs verify that all three cases run end-to-end.

## Current Reproduction Scope

The current reproduction target is the **alpha-stable OU experiment**. This is the
cleanest and most defensible target because the alpha-stable increments are generated
directly with the Chambers-Mallows-Stuck method and do not depend on the separate
characteristic-function rejection sampler.

Gaussian OU is kept as a sanity check. Student-Levy paper reproduction configs are
not included in this release.

The paper describes SAM as part of the optimization strategy, but the original
repository did not archive the exact SAM hyperparameters or final checkpoints. In this
repaired branch, the default alpha-stable reproduction config uses Adam because it was
the stable optimizer that matched the paper-scale alpha-stable diagnostics most closely.
SAM variants are retained only as diagnostics unless their final full-run metrics are
explicitly reported.

## Optimizer Interface

Adam is the default optimizer in the training code and in the alpha-stable
reproduction config. SAM remains available for diagnostics by setting:

```yaml
training:
  optimizer: sam
  sam_base_optimizer: adam
  sam_rho: 0.01
```

For optimizer diagnostics, `sam_base_optimizer: sgd` can also be used, but the
repaired alpha-stable release does not depend on that setting.

## Important Caveat for Student-Levy Experiments

The manuscript used characteristic-function rejection sampling for Student-Levy
increments. The exact original generator and generated datasets were not archived in
the old GitHub snapshot. This branch keeps the code path documented, but does not
ship a paper-reproduction YAML config for Student-Levy. Any future Student-Levy run
must either restore the original CF-RS generator or be clearly labeled as a surrogate,
not as a reproduction of the manuscript table.

Consequently:

- Gaussian and alpha-stable cases are the primary exact reproduction targets.
- In the current release plan, alpha-stable OU is the main reproduction target.
- Student-Levy results should be skipped unless the original CF-RS generator is restored.
- Any public release should state this caveat rather than claiming bitwise or exact numerical reproduction.

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

For a PENN-style quick reproduction from a released checkpoint, run:

```bash
python scripts/reproduce.py \
  --checkpoint checkpoints/alpha_ou.pt \
  --output-dir outputs/repro \
  --force-data
```

The script regenerates 5,000 random alpha-stable OU evaluation paths, loads the
provided checkpoint, writes `eval_metrics.csv`, stores predictions, and generates:

- `figures/predictions.png`
- `figures/residuals.png`
- `summary.md`

To additionally regenerate the manuscript fixed test grid and compare with the
reported alpha-stable OU values, add:

```bash
--fixed-grid
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

The grouped metrics file is the closest fixed-grid output for comparison with the manuscript.
For the alpha-stable OU case, compare it with the reported fixed-grid values using:

```bash
python scripts/compare_fixed_grid.py \
  runs/alpha/evaluation/test_grouped_metrics.csv \
  --csv-out runs/alpha/evaluation/comparison.csv
```

## Expected Differences

Exact numeric agreement is not guaranteed because the old snapshot did not preserve:

- original random seeds
- exact generated datasets
- original checkpoints
- exact Student-Levy random number generator implementation
- full table-generation scripts

Acceptable reproduction should focus on:

- correct parameter ranges and SDE settings
- the manuscript auxiliary input convention `h = dt = T / N`
- table-scale MLPEnet estimates matching the manuscript's reported alpha-stable OU range
- boundary behavior consistent with the paper narrative
- grouped MAE, bias, and SD in the same broad scale as the manuscript after full training

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

See [docs/REPRODUCTION_STATUS.md](docs/REPRODUCTION_STATUS.md) for the current
cloud-run metrics and release recommendation.
