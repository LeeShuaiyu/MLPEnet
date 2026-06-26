# Alpha-Stable OU Reproduction Status

This note records the cloud runs used to decide the first public reproducibility
release of this branch.

## Scope

- Target: alpha-stable Levy driven OU process, corresponding to the paper's Case 2
  and Table 6.
- Data: regenerated from the repaired YAML pipeline with `Ktrain=200000`,
  `Keval=5000`, and six fixed test parameter combinations with 1000 paths each.
- Auxiliary scalar: `h = dt = T / N`.
- Simulation uses `burnin_time=10.0` before collecting each trajectory; this is a
  repaired-pipeline convention rather than a separately archived manuscript artifact.
- Student-Levy: out of scope because the manuscript CF-RS generator and generated
  datasets were not archived.

## Main Reproduction Run

Validation configuration:

- `configs/alpha_ou.yaml`
- `architecture: mlpenet`
- `split_parts: 8`
- `epochs: 800`

Overall test metrics:

| parameter | MAE | bias |
| --- | ---: | ---: |
| eta | 0.298414 | 0.007044 |
| epsilon | 0.001690 | 0.000159 |
| alpha | 0.035477 | -0.021876 |

Compared with the paper's Table 6 MLPEnet values, 14 of 18 fixed-condition MAEs are
no worse than the manuscript table. The remaining four are within the same scale.

## Diagnostics

- Diagnostic `SAM + SGD` was unstable and produced exploding losses.
- `SAM + Adam` with `rho=0.01` and gradient clipping was numerically stable but
  underperformed Adam on the alpha-stable diagnostic run.
- `split_parts=2` and `split_parts=4` diagnostics were slower or comparable in speed
  but did not improve the 120-epoch test metrics enough to justify full reruns.

## Release Decision

Recommended release position:

- Publish this branch as a repaired alpha-stable OU reproduction pipeline.
- Claim successful recovery of the paper-scale alpha-stable MLPEnet result, with
  explicit regenerated-data and optimizer caveats.
- Keep the public repository focused on reproducing the MLPEnet alpha-stable result.
- Mark Student-Levy as out of scope until the original CF-RS generator is restored.
