# Multi-level PEnet

Implementation of **Multi-level PEnet (MLPEnet)** for parameter estimation in
stochastic differential equations driven by Gaussian and non-Gaussian Levy noise.

> Reproducibility status: the original `main` branch was an early implementation
> snapshot and was not sufficient to reproduce the manuscript experiments. This
> branch adds a repaired, config-driven reproduction pipeline. See
> [README_REPRODUCIBILITY.md](README_REPRODUCIBILITY.md) before using the code for
> paper comparisons.

## Reproducible Entry Points

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small end-to-end smoke test:

```bash
bash scripts/smoke.sh
```

Primary paper-scale config:

- `configs/alpha_ou.yaml`

Checkpoint-only reproduction:

```bash
python scripts/reproduce.py \
  --checkpoint checkpoints/alpha_ou.pt \
  --output-dir outputs/repro \
  --fixed-grid
```

This command regenerates 5,000 alpha-stable OU evaluation paths, loads the
checkpoint, runs inference, and writes prediction and residual diagnostic plots
plus metrics. Use `--force-data` to regenerate an existing output dataset.

Gaussian OU is retained as a sanity check. Student-Levy is not part of the current
public reproduction target because the original CF-RS increment generator was not
archived in the old repository snapshot.

Each run saves generated data, config, environment metadata, training history,
checkpoints, predictions, and table-style metrics under the selected `--work-dir`.

## Repository Structure

- `mlpenet/`: repaired package implementation.
- `configs/`: smoke and paper-scale YAML configs.
- `scripts/`: reproducible CLI entry points for data generation, training, evaluation, and table printing.
- `README_REPRODUCIBILITY.md`: detailed reproducibility notes and caveats.
- `docs/PENN_REFERENCE.md`: compatibility notes for the PENN-derived conventions.
- `docs/CLOUD_RUN.md`: GPU-server runbook for pilot and full reproduction jobs.
- `docs/REPRODUCTION_STATUS.md`: current cloud-run evidence and release decision.

## Citation

If this repository is useful, please cite the associated paper:

Shuaiyu Li, Hiroto Saigo, Yang Ruan, Yuzhong Cheng.  
**Multi-level PEnet: A Robust Three-Stage Model for Parameter Estimation in Non-Gaussian Noise-Driven Stochastic Differential Equations.**

## License

MIT License. See [LICENSE](LICENSE).
