# Checkpoints

Place the released alpha-stable OU checkpoint here:

```text
checkpoints/alpha_ou.pt
```

The checkpoint is distributed separately from the Git repository, for example as
a GitHub Release asset named `alpha_ou.pt`.

Expected file:

```text
name: alpha_ou.pt
size: 351K
sha256: 4e9ada33f8611f31b28940dce5320acd0b02a163412bfc3ba4dcdba0b6634ba7
```

The checkpoint-only reproduction entry point is:

```bash
python scripts/reproduce.py \
  --checkpoint checkpoints/alpha_ou.pt \
  --output-dir outputs/repro \
  --fixed-grid
```

Add `--force-data` to regenerate an existing output dataset.

Large checkpoint files are not tracked directly in the repository.
