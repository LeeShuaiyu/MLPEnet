# Checkpoints

Place the released alpha-stable OU checkpoint here:

```text
checkpoints/alpha_ou.pt
```

The checkpoint is distributed separately from the Git repository as a GitHub
Release asset:

<https://github.com/LeeShuaiyu/MLPEnet/releases/download/alpha-ou-v1/alpha_ou.pt>

Expected file:

```text
name: alpha_ou.pt
size: 113K
sha256: 94ccc35644f94fb985415b0af6dcb874f60081e723567aedff418810878b98bf
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
