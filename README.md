# Multi-level PEnet

Official implementation of **Multi-level PEnet (MLPEnet)** for parameter
estimation in stochastic differential equations driven by Gaussian and
non-Gaussian Levy noise.

This repository focuses on the alpha-stable OU experiment from the paper and
provides the model code, experiment configs, checkpoint-based evaluation, and
training/evaluation entry points.

## Usage

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

Checkpoint evaluation:

```bash
python scripts/reproduce.py \
  --checkpoint checkpoints/alpha_ou.pt \
  --output-dir outputs/repro \
  --fixed-grid
```

This command regenerates 5,000 alpha-stable OU evaluation paths, loads the
released checkpoint, runs inference, and writes prediction/residual figures plus
metrics. Use `--force-data` to regenerate an existing output dataset.

Gaussian OU is included as a sanity check. This public release focuses on the
alpha-stable OU experiment.

Checkpoint-only runs write generated data, predictions, figures, and metrics under
the selected `--output-dir`. Training runs write data, logs, checkpoints, and
metrics under the selected `--work-dir`.

## Repository Structure

- `mlpenet/`: model, data generation, training, and evaluation package.
- `configs/`: smoke and paper-scale YAML configs.
- `scripts/`: CLI entry points for data generation, training, evaluation, and figure generation.
- `README_REPRODUCIBILITY.md`: experiment and evaluation notes.
- `docs/IMPLEMENTATION_NOTES.md`: implementation notes for data and model conventions.
- `docs/CLOUD_RUN.md`: GPU-server runbook for pilot and full reproduction jobs.

## Citation

If this repository is useful, please cite the associated paper:

Shuaiyu Li, Hiroto Saigo, Yang Ruan, Yuzhong Cheng.  
**Multi-level PEnet: A Robust Three-Stage Model for Parameter Estimation in Non-Gaussian Noise-Driven Stochastic Differential Equations.**

## License

MIT License. See [LICENSE](LICENSE).
