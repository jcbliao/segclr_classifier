"""Graph encoder f_theta: Z = f_theta(X_tilde, A), Z in R^(N,d) -- step 2 of the
Graph AutoEncoder Classifier spec.

Skeleton graphs here are near-trees (long chains with occasional branch
points, up to ~20k nodes per the p99 in CLAUDE.md), so the encoder defaults to
GraphSAGE-style mean aggregation, which scales to that node count without the
O(N^2)-ish attention cost a dense transformer would pay. `conv_type="gat"`
switches to attention-weighted aggregation (GATv2, which supports edge
features) for when expressiveness matters more than raw throughput; try both
rather than assuming.

Edge length (nm) from aggregate.build_csr's edge weights is used as a scalar
edge feature when the conv type supports one -- physical distance along the
skeleton is exactly the kind of signal geodesic_mean is already implicitly
using, so the encoder gets access to it too.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def _make_conv(conv_type: str, in_dim: int, out_dim: int, edge_dim: int | None, heads: int):
    from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv

    if conv_type == "sage":
        return SAGEConv(in_dim, out_dim)
    if conv_type == "gat":
        # concat=False averages heads -> keeps the layer's output width == out_dim
        return GATv2Conv(in_dim, out_dim, heads=heads, concat=False, edge_dim=edge_dim)
    if conv_type == "transformer":
        return TransformerConv(in_dim, out_dim, heads=heads, concat=False, edge_dim=edge_dim)
    raise ValueError(f"unknown conv_type {conv_type!r}; expected 'sage', 'gat', or 'transformer'")


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,  # raw SegCLR embedding dim (D) -- confirmed 64 for the public release
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 4,
        conv_type: Literal["sage", "gat", "transformer"] = "sage",
        heads: int = 4,
        use_edge_length: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_edge_length = use_edge_length and conv_type in ("gat", "transformer")
        edge_dim = 1 if self.use_edge_length else None

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [
                _make_conv(conv_type, dims[i], dims[i + 1], edge_dim, heads)
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: (2, E) -- already symmetrized (segclr_db's skeleton_edges
            are directed and readers symmetrize; do that before this call).
        edge_attr: (E, 1) edge length in nm, or None. Ignored by SAGEConv.
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            kwargs = {"edge_attr": edge_attr} if self.use_edge_length else {}
            h_new = conv(h, edge_index, **kwargs)
            h_new = norm(h_new)
            h_new = self.act(h_new)
            h_new = self.dropout(h_new)
            if i == 0:
                h = h_new
            else:
                h = h + h_new  # residual once shapes line up (all hidden_dim after layer 0)
        return h
