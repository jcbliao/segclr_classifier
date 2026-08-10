"""Window classifier: one local-neighborhood subgraph in, one hierarchical
cell-type prediction out. Trained end to end on the classification objective
alone -- there is no pretraining stage and no reconstruction path.

Three aggregation methods, selected by `ModelConfig.architecture`, and that
choice is the ONLY thing that differs between them. Same windows, same raw
SegCLR node embeddings, same LCPNHead, same evaluation:

  - "graph_transformer": gnn/graph_transformer.py's AC-attention
    GraphTransformer, a fused encoder + CLS readout producing the graph-level
    embedding directly. Carries five independent feature/ablation switches
    (`gt_use_lpe`, `gt_use_rel_pos`, `gt_use_adj_bias`,
    `gt_attention_scope`, `gt_use_thickness`); the first four default to the
    full model, `gt_use_thickness` defaults off since it needs an extra
    ingested cache.
  - "mpnn": gnn/encoder.py::MPNNEncoder (plain GraphSAGE message passing, no
    attention) followed by MeanReadout.
  - "mean": gnn/readout.py::MeanReadout over the raw node embeddings, with no
    encoder at all -- the mean-pooling baseline this project exists to beat,
    expressed as a configuration of this same class rather than a separately
    coded pipeline.

Read as a ladder of how much learned mixing happens before the readout: none
("mean"), fixed local neighbor averaging over a few hops ("mpnn"), or
adjacency-biased global attention ("graph_transformer").

Classification head is gnn/lcpn.py::LCPNHead (local-classifier-per-node),
not a flat softmax -- see gnn/hierarchy.py for the tree. By default its
per-node heads sit directly on the readout embedding (a linear probe);
`cls_head_resnet=True` inserts the lab's shared ResNet backbone
(gnn/resnet.py) first, matching their own `local_classifier_resnet_sngp`.
That choice is orthogonal to `architecture` -- it applies to all three. Because a single
(B, num_classes) logits tensor doesn't exist for LCPN (each tree node has a
different number of children), forward() returns the readout embedding `g`
rather than logits; callers get a loss/prediction via
`model.cls_head.compute_loss(g, targets)` / `model.cls_head.predict_top_down(g)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from gnn.encoder import MPNNEncoder
from gnn.graph_transformer import GraphTransformer
from gnn.hierarchy import ParsedHierarchy
from gnn.lcpn import LCPNHead
from gnn.readout import MeanReadout
from gnn.resnet import DeepResNetTrunk


@dataclass
class ModelConfig:
    in_dim: int = 64  # raw SegCLR embedding dim (segclr_db's resnet_860b_reshuffled)
    architecture: Literal["graph_transformer", "mpnn", "mean"] = "graph_transformer"
    cls_head_hidden_dim: int | None = None  # None -> plain Linear per LCPN node

    # --- classification head: linear probe vs. the lab's ResNet backbone ---
    # False (default) puts the per-node LCPN heads directly on the readout
    # embedding -- a linear probe. True inserts gnn/resnet.py::DeepResNetTrunk
    # first, shared across all nodes, reproducing the lab's own
    # `local_classifier_resnet_sngp` arrangement (minus SNGP). Applies to every
    # architecture, not just the GraphTransformer.
    cls_head_resnet: bool = False
    cls_resnet_hidden: int = 128  # their configs/local_classifier_sngp.yaml: hidden_size
    cls_resnet_layers: int = 4  # ... hidden_layers
    cls_resnet_dropout: float = 0.0
    cls_resnet_bn: bool = False  # BatchNorm mixes statistics across windows -- off, as theirs is

    # --- gnn/encoder.py::MPNNEncoder (architecture="mpnn") ---
    mpnn_hidden_dim: int = 128
    mpnn_out_dim: int = 128
    mpnn_layers: int = 2
    mpnn_dropout: float = 0.1

    # --- gnn/graph_transformer.py::GraphTransformer (architecture="graph_transformer") ---
    gt_dim: int = 128
    gt_depth: int = 4
    gt_heads: int = 4
    gt_mlp_ratio: int = 4
    gt_pos_dim: int = 8  # must match the pos_dim windows were extracted with
    # (data/dataset_windowed.py's pos_dim / data/geodesic_window.py's
    # DEFAULT_POS_DIM) -- caller's responsibility to keep these in sync, same
    # contract as in_dim already has with the dataset's embedding width.
    gt_use_exp: bool = True  # exp() on predicted gamma, keeps local/global weights positive
    gt_qkv_bias: bool = False
    gt_dropout: float = 0.0
    # Four independent ablation switches, all defaulting to the full model.
    # See gnn/graph_transformer.py's class docstring -- in particular its note
    # that adj_bias is nearly inert under attention_scope="neighborhood",
    # since the hard mask already restricts each row to adj == 1 and a
    # constant added to every surviving logit cancels in the softmax.
    gt_use_lpe: bool = True  # add the per-window Laplacian PE to node embeddings
    # dx, dy, dz AND their norm -- one switch, 4 channels (see
    # GraphTransformer.REL_POS_DIM for why the norm is handed over explicitly)
    gt_use_rel_pos: bool = True
    gt_use_adj_bias: bool = True  # GraphDINO's additive gamma_1 * adj attention bias
    # Replaces the binary adjacency in the bias term with a learned per-head
    # scalar indexed by binned edge length. Off by default: unproven, and
    # keeping it off leaves the three-architecture comparison clean. Requires
    # gt_use_adj_bias. Initialized to reproduce binary adjacency exactly, so
    # turning it on starts from the unmodified model.
    gt_use_dist_bias: bool = False
    gt_attention_scope: Literal["global", "neighborhood"] = "global"
    # Off by default, unlike the four above: it needs
    # data/dendrite_thickness_cache/*.npz ingested AND the dataset built with
    # use_thickness=True. scripts/train_gnn.py drives both from one flag.
    gt_use_thickness: bool = False  # concatenate dendrite shaft radius + measured flag


class WindowClassifier(nn.Module):
    def __init__(self, config: ModelConfig, hierarchy: ParsedHierarchy):
        super().__init__()
        self.config = config

        self.graph_transformer: GraphTransformer | None = None
        self.encoder: MPNNEncoder | None = None
        self.readout: MeanReadout | None = None

        if config.architecture == "graph_transformer":
            self.graph_transformer = GraphTransformer(
                feat_dim=config.in_dim,
                dim=config.gt_dim,
                depth=config.gt_depth,
                num_heads=config.gt_heads,
                mlp_ratio=config.gt_mlp_ratio,
                pos_dim=config.gt_pos_dim,
                use_exp=config.gt_use_exp,
                qkv_bias=config.gt_qkv_bias,
                dropout=config.gt_dropout,
                use_lpe=config.gt_use_lpe,
                use_rel_pos=config.gt_use_rel_pos,
                use_thickness=config.gt_use_thickness,
                use_adj_bias=config.gt_use_adj_bias,
                use_dist_bias=config.gt_use_dist_bias,
                attention_scope=config.gt_attention_scope,
            )
            cls_in_dim = config.gt_dim
        elif config.architecture == "mpnn":
            self.encoder = MPNNEncoder(
                in_dim=config.in_dim,
                hidden_dim=config.mpnn_hidden_dim,
                out_dim=config.mpnn_out_dim,
                num_layers=config.mpnn_layers,
                dropout=config.mpnn_dropout,
            )
            self.readout = MeanReadout()
            cls_in_dim = config.mpnn_out_dim
        elif config.architecture == "mean":
            self.readout = MeanReadout()
            cls_in_dim = config.in_dim  # no encoder -- the mean is over raw embeddings
        else:
            raise ValueError(
                f"unknown architecture {config.architecture!r}; "
                "expected 'graph_transformer', 'mpnn', or 'mean'"
            )

        trunk = None
        if config.cls_head_resnet:
            trunk = DeepResNetTrunk(
                in_dim=cls_in_dim,
                hidden_size=config.cls_resnet_hidden,
                hidden_layers=config.cls_resnet_layers,
                dropout_rate=config.cls_resnet_dropout,
                use_bn=config.cls_resnet_bn,
            )
        self.cls_head = LCPNHead(
            hierarchy=hierarchy,
            in_dim=cls_in_dim,
            hidden_dim=config.cls_head_hidden_dim,
            trunk=trunk,
            trunk_out_dim=config.cls_resnet_hidden if trunk is not None else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        pos_enc: torch.Tensor | None = None,
        rel_pos: torch.Tensor | None = None,
        thickness: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns g: (B, cls_in_dim), one embedding per window, ready for
        self.cls_head. `pos_enc`/`rel_pos`/`thickness`/`edge_attr` are
        consumed only by the GraphTransformer, and only by whichever of its
        switches are on.
        The other two architectures ignore them entirely: the MPNN reads
        structure from `edge_index` alone, and the mean readout sees no
        structure at all."""
        if self.graph_transformer is not None:
            # Only the inputs the configured switches actually consume are
            # required -- an ablated run (gt_use_lpe / gt_use_rel_pos off)
            # legitimately has nothing to pass for the disabled one.
            missing = [
                name
                for name, value, needed in (
                    ("pos_enc", pos_enc, self.graph_transformer.use_lpe),
                    ("rel_pos", rel_pos, self.graph_transformer.use_rel_pos),
                    ("thickness", thickness, self.graph_transformer.use_thickness),
                    ("edge_attr", edge_attr, self.graph_transformer.use_dist_bias),
                )
                if needed and value is None
            ]
            if missing:
                raise ValueError(
                    f"architecture='graph_transformer' requires {' and '.join(missing)} "
                    "(data/geodesic_window.py attaches pos_enc / rel_pos; thickness needs "
                    "WindowedGraphDatasetLCPN(..., use_thickness=True))"
                )
            return self.graph_transformer(
                x, edge_index, batch_index, pos_enc, rel_pos, thickness, edge_attr
            )

        z = self.encoder(x, edge_index) if self.encoder is not None else x
        return self.readout(z, batch_index)
