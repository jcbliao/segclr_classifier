"""Plain message-passing encoder: Z = f_theta(X, A), no attention anywhere.

GraphSAGE-style mean aggregation (`SAGEConv`), stacked `num_layers` deep with
LayerNorm + GELU + dropout and a residual connection once the widths line up.
Paired with gnn/readout.py::MeanReadout in gnn/model.py's `architecture="mpnn"`
configuration, this is the middle point between the two other options: the
mean baseline aggregates raw embeddings with no learned mixing at all, the
GraphTransformer mixes them with adjacency-biased global attention, and this
mixes them with fixed local neighbor averaging over `num_layers` hops.

Deliberately SAGE-only. The attention-capable convolutions (GATv2,
TransformerConv) are not offered here -- attention is what
gnn/graph_transformer.py is for, and keeping the two cleanly separated is the
point of having this as a distinct architecture rather than a conv_type flag.

Depth interacts with window size. Windows average ~10.7 nodes at 10um, and
`window_nm` is a RADIUS, so a window's own geodesic diameter can reach 2x
that. 2 layers (the default) lets a node see 2 hops; going much deeper risks
over-smoothing a graph this small toward a constant vector, at which point the
readout has nothing left to distinguish.
"""

from __future__ import annotations

import torch
from torch import nn


class MPNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,  # raw SegCLR embedding dim (segclr_db's resnet_860b_reshuffled)
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: (2, E) -- already symmetrized (segclr_db's skeleton_edges are
            directed and readers symmetrize; data/geodesic_window.py does that
            before the window ever reaches here).

        Returns Z: (N, out_dim).
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            h_new = self.dropout(self.act(norm(conv(h, edge_index))))
            # Residual only from layer 1 on -- layer 0 changes the width from
            # in_dim to hidden_dim, so there is nothing to add onto yet.
            h = h_new if i == 0 else h + h_new
        return h
