"""Linear phi: one per-node `nn.Linear` and nothing else, averaged by
gnn/readout.py::MeanReadout over the window.

    z = (1/N) * sum_i W x_i + b,   y_hat = rho(z)

The bottom rung of the learned-aggregation ladder, sitting directly under
gnn/pointwise_mlp.py: identical wiring, with the hidden layers and every
nonlinearity taken out. What separates the two is therefore exactly the
nonlinearity, which is the point of having it.

**A node-wise linear map commutes with the mean**, so this is algebraically
`mean` followed by one linear projection: `(1/N) sum_i W x_i = W ((1/N) sum_i
x_i)`. Against the ResNet trunk that means a rank-<=64 factorization of the
trunk's first 64 -> 128 layer rather than any new expressivity, and against a
bare linear probe head it is a reparameterization of the same function class.
That is what makes it a control and not a contender: any margin the pointwise
MLP holds over `mean` that this run also holds is width and optimization, not
the learned per-node feature map.

Embeddings only, like the pointwise MLP: no attention, no positional features,
no graph, `edge_index` never consulted. gnn/model.py refuses
`use_spatial_features` and `use_embeddings=False` for it at construction rather
than honouring them quietly, since either would make the run something other
than what this module's name claims.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearEncoder(nn.Module):
    """phi: (N, in_dim) -> (N, out_dim), applied node-wise. One Linear, no activation.

    `out_dim` defaults to the pointwise MLP's output width so the two rungs
    hand the classification head the same number of channels and differ only
    in how they got there.
    """

    def __init__(
        self,
        in_dim: int = 64,  # raw SegCLR embedding dim (segclr_db's resnet_860b_reshuffled)
        out_dim: int = 128,
    ):
        super().__init__()
        self.phi = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) -- every node in the batch, flat. Returns (N, out_dim).

        No `edge_index` argument, as in gnn/pointwise_mlp.py: nodes are
        transformed independently and the graph plays no part. Which nodes
        belong to which window is settled afterwards, by the readout's
        `batch_index`.
        """
        return self.phi(x)
