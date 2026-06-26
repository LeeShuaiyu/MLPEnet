from __future__ import annotations

import torch
from torch import nn


class WeightedL1Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32).view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(device=pred.device, dtype=pred.dtype)
        return torch.mean(torch.abs(pred - target) * weights)
