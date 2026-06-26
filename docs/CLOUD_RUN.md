# Cloud Reproduction Runbook

Use this runbook after the smoke tests pass locally.

## 1. Server Setup

```bash
git clone -b reproducibility-2026 https://github.com/LeeShuaiyu/MLPEnet.git
cd MLPEnet

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

If `torch.cuda.is_available()` is `False`, install the PyTorch build matching the
server's CUDA version before running paper-scale jobs.

## 2. Alpha-Stable Smoke Test

```bash
PYTHON=.venv/bin/python bash scripts/run_cloud.sh smoke
```

Do not start long jobs until this smoke test completes.

## 3. Alpha-Stable Pilot Run

Before the full 200k-sample alpha-stable run, run:

```bash
PYTHON=.venv/bin/python bash scripts/run_cloud.sh pilot
```

The pilot checks GPU memory, data generation time, and loss curves.

## 4. Full Alpha-Stable Run

Use `tmux` so jobs survive SSH disconnects:

```bash
tmux new -s mlpenet-alpha
source .venv/bin/activate

PYTHON=.venv/bin/python bash scripts/run_cloud.sh full
```

Set `FORCE_DATA=0` when rerunning training against an already generated
`runs/alpha/data` split:

```bash
FORCE_DATA=0 PYTHON=.venv/bin/python bash scripts/run_cloud.sh full
```

Student-Levy is intentionally skipped for this release plan because its manuscript
data generation depended on a CF-RS generator that was not archived in the old
repository snapshot.

## 5. Result Files to Preserve

For each run, keep:

- `run/config.json`
- `run/environment.json`
- `run/history.csv`
- `run/checkpoints/best.pt`
- `evaluation/test_metrics.csv`
- `evaluation/test_grouped_metrics.csv`
- top-level `*.log`

The generated `data/` folders are large but should be kept until results are accepted.

## 6. Publication Decision

Do not publish this branch until:

- smoke tests pass,
- full alpha-stable run produces table-scale metrics,
- Student-Levy is clearly marked as out of scope for this release,
- the release notes state which results are exact pipeline reproductions and which are repaired approximations.
