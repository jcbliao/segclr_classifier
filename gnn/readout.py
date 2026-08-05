"""CLS-query attention readout -- step 5 of the Graph AutoEncoder Classifier
spec. A learned query q_CLS reads the encoded node embeddings Z but does not
write back to the skeleton:

    alpha_i = softmax_i( (W_q q_CLS)^T W_k z_i / sqrt(d_k) )
    g = sum_i alpha_i W_v z_i

Deliberately not hierarchical pooling -- the spec's stated skepticism is that
stacking layers and then mean-pooling is unlikely to beat just using the
embeddings, which is the very mean-pool baseline this project exists to beat.
A later variant may send g back into the graph for global communication; not
implemented here.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax


class CLSAttentionReadout(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # q_CLS is a single learned vector, shared across all graphs -- it is
        # not part of any skeleton, so gnn/masking.py never sees or touches it.
        self.q_cls = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.q_cls, std=0.02)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model) if num_heads > 1 else nn.Identity()

    def forward(self, z: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """
        z: (N_total, d_model) -- node embeddings for a PyG-batched set of graphs
        batch_index: (N_total,) int64 -- which graph (0..B-1) each node belongs to

        Returns g: (B, d_model), one summary vector per graph.
        """
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        q = self.W_q(self.q_cls).unsqueeze(0).expand(num_graphs, -1)  # (B, d_model)
        k = self.W_k(z)  # (N_total, d_model)
        v = self.W_v(z)  # (N_total, d_model)

        H, d_k = self.num_heads, self.d_k
        q = q.view(num_graphs, H, d_k)  # (B, H, d_k)
        k = k.view(-1, H, d_k)  # (N_total, H, d_k)
        v = v.view(-1, H, d_k)  # (N_total, H, d_k)

        # Broadcast each node's graph-level query onto it, then a per-graph
        # scatter-softmax over the node axis (torch_geometric.utils.softmax is
        # exactly a segment-softmax keyed by `index`, which is what turns a
        # single global attention formula into "one attention per graph in the
        # batch" without a for-loop over graphs).
        q_per_node = q[batch_index]  # (N_total, H, d_k)
        logits = (q_per_node * k).sum(dim=-1) / (d_k**0.5)  # (N_total, H)
        alpha = scatter_softmax(logits, batch_index, dim=0)  # (N_total, H)

        weighted = alpha.unsqueeze(-1) * v  # (N_total, H, d_k)
        g = torch.zeros(num_graphs, H, d_k, device=z.device, dtype=z.dtype)
        g.index_add_(0, batch_index, weighted)
        g = g.reshape(num_graphs, H * d_k)  # (B, d_model)
        return self.out_proj(g)
