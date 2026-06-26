# Implementation Notes

This repository keeps the following OU data and model conventions explicit:

- The data format stores `X`, `Y`, and `H`, where `H` is the time span `T`.
- The model concatenates an auxiliary scalar to the pooled LSTM feature.
- The OU data generator discards a transient segment (`discard_T = 10`) before saving observations.
- The loss was a manually weighted L1 loss, with weights chosen to balance parameter scales.
- The alpha-stable OU target estimates `{eta, epsilon, alpha}` jointly.
- Long contiguous trajectory segments are split and corresponding positions are
  concatenated along the feature axis. This is not adjacent-point grouping.

The package uses memory-mapped `.npy` datasets for large runs. The YAML field
`data.auxiliary_input` controls whether the scalar appended to the network is
`span_t` or `dt`.

For manuscript configs, `dt` is the default because the paper denotes the
network's auxiliary observation-frequency input by `h = T / N`.
