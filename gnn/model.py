"""Composes masking + encoder + decoder + readout + classification head into
one module, per the Graph AutoEncoder Classifier spec. Three modes:

  - "pretrain": masked reconstruction only (steps 1-4)
  - "classify": readout + classification head only, no masking (steps 5-6)
  - "joint":    both, on the same masked forward pass (L_joint)

Frozen-vs-fine-tuned-encoder is a training-loop concern, not baked in here:
freeze the encoder by setting `model.encoder.requires_grad_(False)` (or only
adding head/readout params to the optimizer) before calling `forward(...,
mode="classify")`. Keeping that out of forward() means the same module works
for all three frozen/fine-tuned/joint comparisons the spec asks for without a
flag threading through every call site.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from gnn.decoder import ReconstructionDecoder
from gnn.encoder import GNNEncoder
from gnn.masking import NodeMasker
from gnn.readout import CLSAttentionReadout


@dataclass
class ModelConfig:
    in_dim: int = 64  # raw SegCLR embedding dim
    hidden_dim: int = 128
    encoder_out_dim: int = 128
    num_encoder_layers: int = 4
    conv_type: Literal["sage", "gat", "transformer"] = "sage"
    encoder_heads: int = 4
    use_edge_length: bool = True
    encoder_dropout: float = 0.1
    readout_heads: int = 1
    num_classes: int = 2
    mask_prob: float = 0.3
    mask_token_prob: float = 0.8
    replace_prob: float = 0.1
    keep_prob: float = 0.1
    decoder_hidden_dim: int | None = None


class GraphAutoEncoderClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GNNEncoder(
            in_dim=config.in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.encoder_out_dim,
            num_layers=config.num_encoder_layers,
            conv_type=config.conv_type,
            heads=config.encoder_heads,
            use_edge_length=config.use_edge_length,
            dropout=config.encoder_dropout,
        )
        self.decoder = ReconstructionDecoder(
            in_dim=config.encoder_out_dim,
            out_dim=config.in_dim,
            hidden_dim=config.decoder_hidden_dim,
        )
        self.readout = CLSAttentionReadout(
            d_model=config.encoder_out_dim, num_heads=config.readout_heads
        )
        self.cls_head = nn.Linear(config.encoder_out_dim, config.num_classes)
        self.masker = NodeMasker(
            dim=config.in_dim,
            mask_prob=config.mask_prob,
            mask_token_prob=config.mask_token_prob,
            replace_prob=config.replace_prob,
            keep_prob=config.keep_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        mode: Literal["pretrain", "classify", "joint"] = "classify",
        replacement_source: Callable[[int], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        if mode in ("pretrain", "joint"):
            x_in, mask = self.masker(x, replacement_source)
            out["mask"] = mask
        else:
            x_in, mask = x, None

        z = self.encoder(x_in, edge_index, edge_attr)

        if mode in ("pretrain", "joint"):
            x_hat = self.decoder(z[mask])
            out["x_hat"] = x_hat
            out["target"] = x[mask]

        if mode in ("classify", "joint"):
            g = self.readout(z, batch_index)
            out["logits"] = self.cls_head(g)

        return out
