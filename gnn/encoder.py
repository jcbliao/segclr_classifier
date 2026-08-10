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

One opt-in switch, `use_lpe` (gnn/model.py's `mpnn_use_lpe`,
scripts/train_gnn.py's --mpnn-lpe): concatenate the same per-window Laplacian
positional encoding the GraphTransformer consumes onto the raw node features.
Off by default, so the existing `mpnn_L{layers}` runs stay exactly the model
they were -- with it off not a single parameter changes shape.
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
        use_lpe: bool = False,
        pos_dim: int = 8,  # must match what the dataset attached (DEFAULT_POS_DIM)
    ):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        # Concatenated onto the raw features, not added to them the way
        # gnn/graph_transformer.py adds `to_pos_embedding(pos_enc)`. That
        # difference is structural, not a preference: the GraphTransformer has
        # an explicit `to_node_embedding` MLP lifting features to a common
        # width, which gives an additive term somewhere to land; here the raw
        # 64-dim embeddings feed SAGEConv directly, so there is no such space,
        # and adding an 8-dim PE onto 64 SegCLR channels would corrupt the
        # features rather than annotate them. Concatenation is the same family
        # of function anyway -- layer 0's weight matrix simply gains `pos_dim`
        # extra columns, i.e. a learned linear projection of the PE summed into
        # the layer-0 pre-activation.
        #
        # Injected once, at the input, rather than re-added at every layer:
        # that is what GraphDINO does and what the GraphTransformer here does,
        # and with 2 layers there is little for a second injection to recover.
        self.use_lpe = use_lpe
        self.pos_dim = pos_dim if use_lpe else 0

        dims = [in_dim + self.pos_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, pos_enc: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (N, in_dim)
        edge_index: (2, E) -- already symmetrized (segclr_db's skeleton_edges are
            directed and readers symmetrize; data/geodesic_window.py does that
            before the window ever reaches here).
        pos_enc: (N, pos_dim) -- the per-window Laplacian PE
            (data/geodesic_window.py::_window_laplacian_pos_enc), batched the
            same way x is. Required iff `use_lpe`; ignored otherwise.

        Returns Z: (N, out_dim).
        """
        if self.use_lpe:
            if pos_enc is None:
                raise ValueError("use_lpe=True but pos_enc was not provided")
            h = torch.cat([x, pos_enc], dim=-1)
        else:
            h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            h_new = self.dropout(self.act(norm(conv(h, edge_index))))
            # Residual only from layer 1 on -- layer 0 changes the width from
            # in_dim to hidden_dim, so there is nothing to add onto yet.
            h = h_new if i == 0 else h + h_new
        return h
