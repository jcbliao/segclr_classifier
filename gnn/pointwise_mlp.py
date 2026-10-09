"""Pointwise MLP phi: a per-node MLP applied independently to every node, whose
output gnn/readout.py::MeanReadout then averages over the window.

    z = (1/N) * sum_i phi(x_i),   y_hat = rho(z)

with `rho` the ordinary downstream classifier (gnn/lcpn.py::LCPNHead, with or
without the shared ResNet trunk) -- so nothing about the head changes, and the
comparison against the other architectures stays a comparison of aggregation
alone.

This is the rung of the ladder between "mean" and "mpnn": the mean baseline
averages the raw embeddings with no learned transform at all, this learns a
per-node transform first and then averages, and the MPNN additionally lets
nodes see each other before the average. The gap between "mean" and this one
is therefore attributable to the learned per-node feature map; the gap between
this one and "mpnn" to message passing, since those are the only things that
differ.

Deliberately set-only: no attention, no positional features, no graph. The
window reaches this encoder as an unordered bag of embeddings and `edge_index`
is never consulted, which is what makes it permutation invariant by
construction rather than by training. gnn/model.py refuses to combine it with
`use_spatial_features`, so the distinction cannot be quietly eroded by a flag.
"""

from __future__ import annotations

import torch
from torch import nn


class PointwiseMLPEncoder(nn.Module):
    """phi: (N, in_dim) -> (N, out_dim), applied node-wise.

    `num_layers=2` with the default widths is the 64 -> 128 -> 128 MLP this
    architecture was specified as. Every layer is Linear + GELU, the last one
    included: without a nonlinearity after the final Linear, that Linear would
    commute with the mean that follows and collapse into the head's first
    layer, leaving a shallower model than the depth suggests.
    """

    def __init__(
        self,
        in_dim: int = 64,  # raw SegCLR embedding dim (segclr_db's resnet_860b_reshuffled)
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        self.phi = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) -- every node in the batch, flat. Returns (N, out_dim).

        No `edge_index` argument, unlike gnn/encoder.py::MPNNEncoder: nodes are
        transformed independently and the graph plays no part. Which nodes
        belong to which window is settled afterwards, by the readout's
        `batch_index`.
        """
        return self.phi(x)
