"""Pre-activation residual MLP trunk, ported from the lab's own
`segCLR_cell_classification` (`src/models/resnet.py::DeepResNet`, the
`@register_model("resnet")` one; `scripts/resnet.py` is an older standalone
copy of the same math with the input width hardcoded to 64).

Purpose here: an alternative to the linear probe in gnn/lcpn.py::LCPNHead.
Their `local_classifier_resnet_sngp` -- the model their production LCPN config
actually trains -- is exactly this trunk shared across all hierarchy nodes,
with one head per node reading its output. `LCPNHead(trunk=...)` reproduces
that arrangement: the trunk is shared, only the per-node heads are per-node.

Faithful to the original except for SNGP. Theirs subclasses to
`DeepResNetSNGP`, which wraps every dense layer in `spectral_norm` and swaps
the output layer for a `RandomFeatureGP`. This project has consistently
stripped SNGP (see gnn/lcpn.py's module docstring -- nothing else in gnn/ uses
it, and it exists for uncertainty estimation this project does not consume),
so the layers here are plain `nn.Linear`.

Two structural details worth not "fixing" on sight, both matching the source:

- **Pre-activation ordering.** Each block is `norm -> relu -> linear -> norm ->
  relu -> linear`, then `hidden = residual + hidden`. The activation precedes
  the weight, not the other way round, so the skip path stays linear all the
  way through the stack.
- **No normalization anywhere by default.** `use_bn=False` is the original's
  default and their LCPN config does not override it. Note that if it is
  enabled it is BatchNorm, which mixes statistics across every window in a
  batch -- see gnn/graph_transformer.py's AttentionBlock for why that is
  avoided elsewhere in this codebase.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Defaults from the lab's own configs/local_classifier_sngp.yaml -- the config
# behind their production LCPN run -- rather than from the class signature,
# which predates it (num_hidden=32 there).
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_HIDDEN_LAYERS = 4


class DeepResNetTrunk(nn.Module):
    """(B, in_dim) -> (B, hidden_size). Backbone only: no classifier, since
    LCPNHead supplies one head per hierarchy node on top."""

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn

        self.input_layer = nn.Linear(in_dim, hidden_size)
        self.dense_layers_1 = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        self.dense_layers_2 = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )
        if use_bn:
            self.bns_1 = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(hidden_layers)])
            self.bns_2 = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        hidden = self.input_layer(x)

        for i in range(self.hidden_layers):
            residual = hidden
            if self.dropout_rate > 0:
                residual = F.dropout(residual, p=self.dropout_rate, training=self.training)
            if self.use_bn:
                residual = self.bns_1[i](residual)
            residual = torch.relu(residual)
            residual = self.dense_layers_1[i](residual)

            if self.dropout_rate > 0:
                residual = F.dropout(residual, p=self.dropout_rate, training=self.training)
            if self.use_bn:
                residual = self.bns_2[i](residual)
            residual = torch.relu(residual)
            residual = self.dense_layers_2[i](residual)

            hidden = residual + hidden

        return hidden
