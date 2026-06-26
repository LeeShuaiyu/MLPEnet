# Checkpoints

Place the released alpha-stable OU checkpoint here:

```text
checkpoints/alpha_ou.pt
```

The checkpoint-only reproduction entry point is:

```bash
python scripts/reproduce.py \
  --checkpoint checkpoints/alpha_ou.pt \
  --output-dir outputs/repro \
  --force-data
```

Add `--fixed-grid` to evaluate the manuscript fixed test grid in the same run.

Large checkpoint files are not tracked directly in the repository. They can be
attached as release assets or downloaded separately, then placed at the path above.
