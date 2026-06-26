# PENN Compatibility Notes

This repository was originally derived from the public PENN implementation:

<https://github.com/xiaolong-snnu/PENN>

Relevant conventions preserved or made explicit in this release:

- The legacy data format stored `X`, `Y`, and `H`, where `H` is the time span `T`.
- The model concatenated this scalar to the pooled LSTM feature and called it `h` in code.
- The OU data generator discarded a transient segment (`discard_T = 10`) before saving observations.
- The loss was a manually weighted L1 loss, with weights chosen to balance parameter scales.
- The standard OU target for alpha-stable noise estimated `{eta, epsilon, alpha}` jointly.
- The legacy split operation split long contiguous trajectory segments and concatenated
  corresponding positions along the feature axis. It was not adjacent-point grouping.

This release uses memory-mapped `.npy` datasets instead of large pickle files, but keeps
the same modeling idea. The YAML field `data.auxiliary_input` controls whether the
scalar appended to the network is `span_t` (legacy PENN-compatible) or `dt`.

For manuscript reproduction configs, `dt` is the default because the paper denotes the
network's auxiliary observation-frequency input by `h = T / N`.
