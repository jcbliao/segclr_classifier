"""Reconstruction decoder d_phi: x_hat_i = d_phi(z_i) -- step 3 of the Graph
AutoEncoder Classifier spec. Applied per-node, independently -- it only ever
sees z[mask], the encoded masked nodes (see model.py), never the graph
structure, so a plain MLP is the whole story here.
"""

from __future__ import annotations

import torch
from torch import nn


class ReconstructionDecoder(nn.Module):
    def __init__(self, in_dim: int = 128, out_dim: int = 64, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
