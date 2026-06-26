from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class MultiLevelPENet(nn.Module):
    """Three-stage MLPEnet for simultaneous SDE parameter estimation.

    Stage 1 has two condensation paths:
    - a CNN/pooling path for dense fluctuation and jump information;
    - a split-concatenation path that preserves sparse long-horizon drift evidence.
    Stage 2 extracts sequence features with LSTM.
    Stage 3 maps features plus sampling step h to all target parameters.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 25,
        lstm_layers: int = 4,
        conv_layers: int = 2,
        conv_channels: int = 25,
        conv_kernel_size: int = 3,
        split_parts: int = 8,
        fc_layers: int = 3,
        fc_hidden: int = 20,
        activation: str = "elu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if split_parts < 1:
            raise ValueError("split_parts must be >= 1")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.split_parts = split_parts
        self.activation_name = activation

        conv = []
        in_channels = input_dim
        padding = conv_kernel_size // 2
        for _ in range(conv_layers):
            conv.append(nn.Conv1d(in_channels, conv_channels, conv_kernel_size, padding=padding))
            conv.append(_activation(activation))
            conv.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
            in_channels = conv_channels
        self.cnn = nn.Sequential(*conv)

        self.split_projection = nn.Sequential(
            nn.Linear(input_dim * split_parts, conv_channels),
            _activation(activation),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels * 2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        layers: list[nn.Module] = []
        in_features = hidden_dim + 1
        for layer_idx in range(fc_layers):
            layers.append(nn.Linear(in_features, fc_hidden))
            if layer_idx == 0:
                layers.append(nn.BatchNorm1d(fc_hidden))
            layers.append(_activation(activation))
            in_features = fc_hidden
        layers.append(nn.Linear(in_features, output_dim))
        self.head = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                for block in param.chunk(4, 0):
                    nn.init.orthogonal_(block)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _split_concat(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, features = x.shape
        reduced_len = math.ceil(seq_len / self.split_parts)
        target_len = reduced_len * self.split_parts
        if target_len > seq_len:
            x = F.pad(x, (0, 0, 0, target_len - seq_len))
        # Split-concatenation follows the legacy split operation: split the
        # full trajectory into long contiguous chunks, then concatenate
        # corresponding positions from each chunk along the feature axis.
        # This preserves long-horizon drift information. Grouping adjacent
        # points here would collapse the operation into local downsampling.
        x = x.reshape(batch, self.split_parts, reduced_len, features)
        x = x.transpose(1, 2).reshape(batch, reduced_len, self.split_parts * features)
        return self.split_projection(x)

    def _align_cnn(self, cnn_features: torch.Tensor, target_len: int) -> torch.Tensor:
        if cnn_features.shape[1] == target_len:
            return cnn_features
        x = cnn_features.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        split_features = self._split_concat(x)
        cnn_features = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        cnn_features = self._align_cnn(cnn_features, split_features.shape[1])
        features = torch.cat([split_features, cnn_features], dim=-1)
        seq, _ = self.lstm(features)

        condensed_lengths = torch.full(
            (x.shape[0], 1),
            float(seq.shape[1]),
            device=seq.device,
            dtype=lengths.dtype,
        )
        time_index = torch.arange(seq.shape[1], device=seq.device).view(1, -1, 1)
        mask = (time_index < condensed_lengths.view(-1, 1, 1)).to(seq.dtype)
        pooled = (seq * mask).sum(dim=1) / condensed_lengths.to(seq.dtype).view(-1, 1)
        z = torch.cat([pooled, h], dim=1)
        return self.head(z)
