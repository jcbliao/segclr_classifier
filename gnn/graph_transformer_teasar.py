"""Graph Transformer for mixed SegCLR/TEASAR skeleton windows.

Every node is a real graph token. ``has_segclr`` controls only the SegCLR
feature contribution: observed nodes receive a projected 64-D embedding and
unobserved routing nodes receive a learned missing-modality token. Both kinds
still receive geometry, LPE, adjacency, and attention. Padding remains solely
the responsibility of ``to_dense_batch`` in the shared GraphTransformer.
"""

from __future__ import annotations

import torch
from torch import nn

from gnn.graph_transformer import GraphTransformer


class GraphTransformerTEASAR(GraphTransformer):
    """AC-attention transformer for K embedded plus M structural nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.cls_token.shape[-1]
        feat_dim = self.feat_dim

        # Replace the ordinary concatenate-then-project input with independent
        # modality projections. This prevents an absent SegCLR vector from
        # being mistaken for a legitimate all-zero observation.
        self.to_node_embedding = None
        self.segclr_projection = (
            nn.Sequential(nn.Linear(feat_dim, dim * 2), nn.ReLU(True), nn.Linear(dim * 2, dim))
            if self.use_features else None
        )
        self.missing_segclr_token = (
            nn.Parameter(torch.randn(1, 1, dim)) if self.use_features else None
        )
        geometry_dim = (
            (self.REL_POS_DIM if self.use_rel_pos else 0)
            + (self.THICKNESS_DIM if self.use_thickness else 0)
        )
        self.geometry_projection = (
            nn.Sequential(
                nn.Linear(geometry_dim, dim * 2), nn.ReLU(True), nn.Linear(dim * 2, dim)
            )
            if geometry_dim else None
        )
        if self.segclr_projection is None and self.geometry_projection is None:
            raise ValueError("graph_transformer_teasar has no enabled node modality")

    def _embed_nodes(
        self,
        x_dense: torch.Tensor,
        rel_pos_dense: torch.Tensor | None,
        thickness_dense: torch.Tensor | None,
        has_segclr_dense: torch.Tensor | None,
    ) -> torch.Tensor:
        node_emb = None
        if self.use_features:
            if has_segclr_dense is None:
                raise ValueError(
                    "graph_transformer_teasar requires has_segclr for every real node"
                )
            projected = self.segclr_projection(x_dense)
            node_emb = torch.where(
                has_segclr_dense.unsqueeze(-1), projected,
                self.missing_segclr_token.expand_as(projected),
            )

        geometry = []
        if self.use_rel_pos:
            geometry.extend([
                rel_pos_dense,
                torch.linalg.norm(rel_pos_dense, dim=-1, keepdim=True),
            ])
        if self.use_thickness:
            geometry.append(thickness_dense)
        if geometry:
            geom_input = torch.cat(geometry, dim=-1) if len(geometry) > 1 else geometry[0]
            geom_emb = self.geometry_projection(geom_input)
            node_emb = geom_emb if node_emb is None else node_emb + geom_emb
        return node_emb
