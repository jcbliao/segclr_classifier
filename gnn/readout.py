"""Graph-level readout: node embeddings -> one vector per graph (window).

Only a plain unweighted mean lives here. The other aggregation method,
gnn/graph_transformer.py's AC-attention GraphTransformer, carries its own CLS
token internally rather than bolting a separate readout module on top of an
encoder, so it needs nothing from this file.
"""

from __future__ import annotations

import torch
from torch import nn


class MeanReadout(nn.Module):
    """Plain, unweighted mean over each graph's nodes -- zero parameters.

    Applied directly to the raw per-node SegCLR embeddings (there is no
    encoder in front of it), this IS the mean-pooling baseline, expressed as
    a configuration of gnn/model.py::WindowClassifier rather than a
    separately coded pipeline: same window construction, same LCPNHead, same
    eval as the GraphTransformer, with the aggregation method (mean vs.
    learned attention) the only thing that varies.
    """

    def forward(self, z: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        d = z.shape[-1]
        g = torch.zeros(num_graphs, d, device=z.device, dtype=z.dtype)
        g.index_add_(0, batch_index, z)
        counts = torch.zeros(num_graphs, device=z.device, dtype=z.dtype)
        counts.index_add_(0, batch_index, torch.ones(z.shape[0], device=z.device, dtype=z.dtype))
        return g / counts.clamp(min=1).unsqueeze(-1)
