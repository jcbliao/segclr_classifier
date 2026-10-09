"""Window classifier: one local-neighborhood subgraph in, one hierarchical
cell-type prediction out. Trained end to end on the classification objective
alone -- there is no pretraining stage and no reconstruction path.

Five aggregation methods, selected by `ModelConfig.architecture`, and that
choice is the ONLY thing that differs between them. Same windows, same raw
SegCLR node embeddings, same LCPNHead, same evaluation:

  - "graph_transformer": gnn/graph_transformer.py's AC-attention
    GraphTransformer, a fused encoder + CLS readout producing the graph-level
    embedding directly. Carries five independent feature/ablation switches
    (`gt_use_lpe`, `gt_use_rel_pos`, `gt_use_adj_bias`,
    `gt_attention_scope`, `gt_use_thickness`); the first four default to the
    full model, `gt_use_thickness` defaults off since it needs an extra
    ingested cache.
  - "graph_transformer_teasar": the mixed-node GraphTransformer for new
    skeletons. Embedded nodes receive projected SegCLR features; structural
    nodes receive a learned missing-modality token. Every node still receives
    geometry/LPE and remains active in adjacency and attention.
  - "mpnn": gnn/encoder.py::MPNNEncoder (plain GraphSAGE message passing, no
    attention) followed by MeanReadout.
  - "mpnn_complete": the same MPNNEncoder over a per-window clique instead
    of the skeleton edges -- message passing with the graph structure removed
    but the message passing kept.
  - "pointwise_mlp": gnn/pointwise_mlp.py::PointwiseMLPEncoder, a per-node MLP, followed by
    MeanReadout. z = (1/N) sum_i phi(x_i), then the ordinary head as rho.
    Set-only by construction: no attention, no positional features, no graph.
  - "linear": gnn/linear_encoder.py::LinearEncoder, the same shape with one
    Linear and no nonlinearity, followed by MeanReadout. Embeddings only. A
    node-wise linear map commutes with the mean, so this is `mean` plus a
    linear projection rather than a more expressive model -- it is the control
    that isolates the pointwise MLP's nonlinearity.
  - "mean": gnn/readout.py::MeanReadout over the raw node embeddings, with no
    encoder at all -- the mean-pooling baseline this project exists to beat,
    expressed as a configuration of this same class rather than a separately
    coded pipeline.

Read as a ladder of how much learned mixing happens before the readout: none
("mean"), a per-node linear map that the mean absorbs ("linear"), a learned
nonlinear per-node transform and nothing else ("pointwise_mlp"), fixed
local neighbor averaging over a few hops ("mpnn"), the same averaging with the
skeleton replaced by a clique ("mpnn_complete"), or adjacency-biased global
attention ("graph_transformer").

Classification head is gnn/lcpn.py::LCPNHead (local-classifier-per-node),
not a flat softmax -- see gnn/hierarchy.py for the tree. By default its
per-node heads sit directly on the readout embedding (a linear probe);
`cls_head_resnet=True` inserts the lab's shared ResNet backbone
(gnn/resnet.py) first, matching their own `local_classifier_resnet_sngp`.
That choice is orthogonal to `architecture` -- it applies to every one. Because a single
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

from gnn.linear_encoder import LinearEncoder
from gnn.pointwise_mlp import PointwiseMLPEncoder
from gnn.encoder import MPNNEncoder
from gnn.graph_transformer import GraphTransformer
from gnn.graph_transformer_teasar import GraphTransformerTEASAR
from gnn.hierarchy import ParsedHierarchy
from gnn.flat import FlatHead
from gnn.lcpn import LCPNHead
from gnn.readout import MeanReadout
from gnn.resnet import DeepResNetTrunk


@dataclass
class ModelConfig:
    in_dim: int = 64  # raw SegCLR embedding dim (segclr_db's resnet_860b_reshuffled)
    postsynaptic_dim: int = 0  # Added after pooling, before the classification trunk
    append_presynaptic_mean: bool = False  # Matched-width control using raw embeddings
    architecture: Literal[
        "graph_transformer", "graph_transformer_teasar", "mpnn_complete", "mpnn",
        "pointwise_mlp", "linear", "mean"
    ] = "graph_transformer"
    cls_head_hidden_dim: int | None = None  # None -> plain Linear per LCPN node
    classifier: Literal["lcpn", "flat"] = "lcpn"

    # --- classification head: linear probe vs. the lab's ResNet backbone ---
    # False (default) puts the per-node LCPN heads directly on the readout
    # embedding -- a linear probe. True inserts gnn/resnet.py::DeepResNetTrunk
    # first, shared across all nodes, reproducing the lab's own
    # `local_classifier_resnet_sngp` arrangement (minus SNGP). Applies to every
    # architecture, not just the GraphTransformer.
    cls_head_resnet: bool = True
    cls_resnet_hidden: int = 128  # their configs/local_classifier_sngp.yaml: hidden_size
    cls_resnet_layers: int = 4  # ... hidden_layers
    cls_resnet_dropout: float = 0.0
    cls_resnet_bn: bool = False  # BatchNorm mixes statistics across windows -- off, as theirs is

    # --- spatial node features for "mean" and "mpnn" ---
    # Concatenate the per-window Laplacian PE and the center-relative geometry
    # onto the raw embeddings, giving those two architectures the same node
    # inputs the GraphTransformer already builds from `gt_use_lpe` /
    # `gt_use_rel_pos`. One switch rather than two, because its purpose is a
    # matched-feature cross-architecture sweep: with it on, "mean"/"mpnn" see
    # what a default GraphTransformer sees, and with it off they see raw
    # embeddings alone. The GraphTransformer consumes these inputs internally
    # and keeps its own independent switches, so this must not be combined with
    # architecture="graph_transformer".
    use_spatial_features: bool = False
    use_position: bool = False
    use_lpe: bool = False

    # --- geometry-only control ---
    # False drops the 64-dim SegCLR embedding from the node input entirely,
    # leaving only morphology: the graph, the center-relative offset and the
    # Laplacian PE. This is the control for "how much of the score is the
    # embeddings and how much is the shape they sit on" -- a question the
    # aggregation ladder cannot answer on its own, since every rung of it
    # consumes the embeddings. Applies to every architecture except
    # "pointwise_mlp", which is set-only; for "mean"/"mpnn"/"mpnn_complete" it
    # requires use_spatial_features, which is the only other source of node
    # input they have.
    use_embeddings: bool = True

    # --- gnn/encoder.py::MPNNEncoder (architecture="mpnn") ---
    mpnn_hidden_dim: int = 128
    mpnn_out_dim: int = 128
    mpnn_layers: int = 2
    mpnn_dropout: float = 0.1

    # --- gnn/pointwise_mlp.py::PointwiseMLPEncoder (architecture="pointwise_mlp") ---
    # 64 -> 128 -> 128 at the defaults. No dropout and no normalization knob:
    # this rung of the ladder is deliberately the simplest learned aggregator
    # there is, and every extra option is somewhere for it to stop being that.
    # --- gnn/linear_encoder.py::LinearEncoder (architecture="linear") ---
    # Matches pointwise_mlp_out_dim by default, so the two rungs hand the head
    # the same width and differ only by the nonlinearity between them.
    linear_out_dim: int = 128

    pointwise_mlp_hidden_dim: int = 128
    pointwise_mlp_out_dim: int = 128
    pointwise_mlp_layers: int = 2

    # --- GraphTransformer / GraphTransformerTEASAR ---
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
        self.pointwise_mlp: PointwiseMLPEncoder | None = None
        self.linear: LinearEncoder | None = None
        self.readout: MeanReadout | None = None
        self._aggregator_frozen = False  # see freeze_aggregator()

        # dx, dy, dz, ||(dx,dy,dz)|| + the Laplacian PE -- the same channels,
        # in the same order, GraphTransformer.forward assembles for itself. The
        # norm is derived here rather than cached, exactly as it is there.
        self.use_position = config.use_position or config.use_spatial_features
        self.use_lpe = config.use_lpe or config.use_spatial_features
        self.use_spatial_features = self.use_position or self.use_lpe
        if self.use_spatial_features and config.architecture in (
            "graph_transformer", "graph_transformer_teasar"
        ):
            raise ValueError(
                "use_spatial_features applies to architecture='mean' and 'mpnn' only; "
                "the GraphTransformer builds these inputs itself -- use gt_use_lpe / "
                "gt_use_rel_pos instead"
            )
        if self.use_spatial_features and config.architecture in ("pointwise_mlp", "linear"):
            raise ValueError(
                f"use_spatial_features does not apply to architecture={config.architecture!r}, "
                "which is set-only by construction: no attention, no positional features, no "
                "graph. Use 'mean' or 'mpnn' for a spatially-featured run"
            )
        self.use_embeddings = config.use_embeddings
        if not self.use_embeddings and config.architecture in ("pointwise_mlp", "linear"):
            raise ValueError(
                f"use_embeddings=False with architecture={config.architecture!r} leaves an "
                "empty node input -- the embeddings are its only input, since it takes no "
                "spatial features and no graph"
            )
        if not self.use_embeddings and config.architecture in ("mean", "mpnn", "mpnn_complete") and not self.use_spatial_features:
            raise ValueError(
                "use_embeddings=False with architecture='mean'/'mpnn' requires "
                "use_spatial_features=True -- otherwise the node input is empty"
            )
        self.spatial_dim = (
            (GraphTransformer.REL_POS_DIM if self.use_position else 0)
            + (config.gt_pos_dim if self.use_lpe else 0)
        )
        node_in_dim = (config.in_dim if self.use_embeddings else 0) + self.spatial_dim

        if config.architecture in ("graph_transformer", "graph_transformer_teasar"):
            transformer_cls = (
                GraphTransformerTEASAR
                if config.architecture == "graph_transformer_teasar"
                else GraphTransformer
            )
            self.graph_transformer = transformer_cls(
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
                attention_scope=config.gt_attention_scope,
                use_features=config.use_embeddings,
            )
            cls_in_dim = config.gt_dim
        elif config.architecture in ("mpnn", "mpnn_complete"):
            self.encoder = MPNNEncoder(
                in_dim=node_in_dim,
                hidden_dim=config.mpnn_hidden_dim,
                out_dim=config.mpnn_out_dim,
                num_layers=config.mpnn_layers,
                dropout=config.mpnn_dropout,
            )
            self.readout = MeanReadout()
            cls_in_dim = config.mpnn_out_dim
        elif config.architecture == "pointwise_mlp":
            self.pointwise_mlp = PointwiseMLPEncoder(
                in_dim=config.in_dim,
                hidden_dim=config.pointwise_mlp_hidden_dim,
                out_dim=config.pointwise_mlp_out_dim,
                num_layers=config.pointwise_mlp_layers,
            )
            self.readout = MeanReadout()
            cls_in_dim = config.pointwise_mlp_out_dim
        elif config.architecture == "linear":
            self.linear = LinearEncoder(in_dim=config.in_dim, out_dim=config.linear_out_dim)
            self.readout = MeanReadout()
            cls_in_dim = config.linear_out_dim
        elif config.architecture == "mean":
            self.readout = MeanReadout()
            cls_in_dim = node_in_dim  # no encoder -- the mean is over the node features
        else:
            raise ValueError(
                f"unknown architecture {config.architecture!r}; expected "
                "'graph_transformer', 'graph_transformer_teasar', 'mpnn_complete', 'mpnn', 'pointwise_mlp', "
                "'linear', or 'mean'"
            )

        if config.postsynaptic_dim not in (0, 64):
            raise ValueError('postsynaptic_dim must be 0 or 64')
        if config.append_presynaptic_mean and config.postsynaptic_dim:
            raise ValueError('Presynaptic mean and postsynaptic fusion are mutually exclusive')
        if config.append_presynaptic_mean and config.in_dim != 64:
            raise ValueError('The matched-width control requires 64-dimensional raw embeddings')
        cls_in_dim += config.postsynaptic_dim + (64 if config.append_presynaptic_mean else 0)
        trunk = None
        if config.cls_head_resnet:
            trunk = DeepResNetTrunk(
                in_dim=cls_in_dim,
                hidden_size=config.cls_resnet_hidden,
                hidden_layers=config.cls_resnet_layers,
                dropout_rate=config.cls_resnet_dropout,
                use_bn=config.cls_resnet_bn,
            )
        classifier = getattr(config, "classifier", "lcpn")
        if classifier == "lcpn":
            self.cls_head = LCPNHead(
                hierarchy=hierarchy,
                in_dim=cls_in_dim,
                hidden_dim=config.cls_head_hidden_dim,
                trunk=trunk,
                trunk_out_dim=config.cls_resnet_hidden if trunk is not None else None,
            )
        elif classifier == "flat":
            if config.cls_head_hidden_dim is not None:
                raise ValueError("--cls-hidden-dim is LCPN-only; use the shared ResNet trunk")
            self.cls_head = FlatHead(
                hierarchy=hierarchy, in_dim=cls_in_dim, trunk=trunk,
                trunk_out_dim=config.cls_resnet_hidden if trunk is not None else None,
            )
        else:
            raise ValueError(f"unknown classifier {classifier!r}")

    def freeze_aggregator(self) -> int:
        """Hold the aggregation stage at its initial random weights, training
        only the classification head. Returns the number of frozen parameters.

        This is the random-features control. Mean pooling is a fixed,
        zero-parameter operation, so a mean-pool run puts every parameter and
        every gradient step into the head -- while a GNN run trains the
        aggregation and the head jointly, against each other, from scratch. A
        margin over mean pooling therefore confounds three things: a better
        aggregation, more total capacity, and a different optimization
        problem. Freezing separates the first from the third: whatever a
        frozen random aggregator retains is attributable to its STRUCTURE
        (attention over the local subgraph, the Laplacian PE, the
        center-relative geometry) rather than to anything it learned, and the
        head is once again the only thing training.

        Dropout inside the frozen stage is switched off as well, via the
        `train()` override below -- otherwise the "fixed" features would be
        resampled every forward pass and the control would measure a noisy
        aggregator rather than a frozen one. MPNNEncoder's dropout defaults to
        0.1, so this is not hypothetical.
        """
        modules = self._aggregator_modules()
        if not modules:
            raise ValueError(
                "architecture='mean' has no aggregation parameters to freeze -- "
                "MeanReadout is parameter-free, so a frozen run would be identical "
                "to an ordinary one"
            )
        self._aggregator_frozen = True
        n_frozen = 0
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(False)
                n_frozen += p.numel()
        self.train(self.training)  # apply the eval-mode pin immediately
        return n_frozen

    def train(self, mode: bool = True):
        """Standard nn.Module.train(), except that a frozen aggregator stays
        in eval mode -- the training loop calls model.train() every epoch,
        which would otherwise re-enable its dropout."""
        super().train(mode)
        if getattr(self, "_aggregator_frozen", False):
            for module in self._aggregator_modules():
                module.eval()
        return self

    def _aggregator_modules(self) -> tuple[nn.Module, ...]:
        """Every parameterized aggregation stage this configuration built --
        empty for architecture="mean", whose readout has no parameters.

        Kept as one list because freeze_aggregator() and train() must agree on
        what "the aggregator" is; two separately maintained tuples would drift
        the first time a fourth aggregation method was added, and silently:
        a stage missed by train() keeps its dropout live under a freeze.
        """
        return tuple(
            m for m in (self.graph_transformer, self.encoder, self.pointwise_mlp, self.linear)
            if m is not None
        )

    def forward(self, x, edge_index, batch_index, pos_enc=None, rel_pos=None,
                thickness=None, has_segclr=None, postsynaptic_embedding=None):
        g = self._encode(x, edge_index, batch_index, pos_enc, rel_pos,
                         thickness, has_segclr)
        if self.config.postsynaptic_dim:
            if (postsynaptic_embedding is None or
                    postsynaptic_embedding.shape != (g.shape[0], self.config.postsynaptic_dim)):
                raise ValueError('Expected one postsynaptic embedding per pooled window')
            g = torch.cat((g, postsynaptic_embedding.to(dtype=g.dtype)), dim=-1)
        if self.config.append_presynaptic_mean:
            raw_mean = MeanReadout()(x, batch_index)
            g = torch.cat((g, raw_mean.to(dtype=g.dtype)), dim=-1)
        return g

    def predict_distribution(self, x, edge_index, batch_index, **kwargs):
        """Expose the classifier's logits/probabilities without changing training."""
        hidden=self.forward(x,edge_index,batch_index,**kwargs)
        return self.cls_head.predict_distribution(hidden)

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        pos_enc: torch.Tensor | None = None,
        rel_pos: torch.Tensor | None = None,
        thickness: torch.Tensor | None = None,
        has_segclr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns g: (B, cls_in_dim), one embedding per window, ready for
        self.cls_head.

        `thickness` is consumed only by the GraphTransformer. `pos_enc` and
        `rel_pos` reach the GraphTransformer according to its own switches, and
        reach "mean"/"mpnn" iff `use_spatial_features` is set -- otherwise
        those two see raw embeddings alone, the MPNN reading structure from
        `edge_index` and the mean readout seeing no structure at all.
        "pointwise_mlp" and "linear" read `x` and nothing else, whatever is
        passed alongside them."""
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
                )
                if needed and value is None
            ]
            if missing:
                raise ValueError(
                    f"architecture={self.config.architecture!r} requires {' and '.join(missing)} "
                    "(data/geodesic_window.py attaches pos_enc / rel_pos; thickness needs "
                    "WindowedGraphDatasetLCPN(..., use_thickness=True))"
                )
            return self.graph_transformer(
                x, edge_index, batch_index, pos_enc, rel_pos, thickness, has_segclr
            )

        if self.pointwise_mlp is not None:
            # No edge_index, no pos_enc, no rel_pos: phi sees one node's raw
            # embedding at a time and the mean is the only thing that ever
            # looks across nodes. Those inputs are accepted in the signature
            # and ignored here rather than rejected, exactly as the "mean" and
            # "mpnn" paths ignore what they don't consume -- the constructor is
            # where an attempt to actually switch them on is refused.
            return self.readout(self.pointwise_mlp(x), batch_index)

        if self.linear is not None:
            # Same contract as the pointwise MLP above, one Linear deep.
            return self.readout(self.linear(x), batch_index)

        node_input = self._node_features(x, pos_enc, rel_pos)
        if self.encoder is not None:
            if self.config.architecture == "mpnn_complete":
                # Directed clique (including self loops) independently within
                # each graph in the PyG batch. With at most 40 nodes this is
                # small, and keeps the dataset's skeleton edges intact for GT.
                edges = []
                for graph_id in torch.unique(batch_index):
                    nodes = torch.nonzero(batch_index == graph_id, as_tuple=False).flatten()
                    edges.append(torch.cartesian_prod(nodes, nodes).T)
                edge_index = torch.cat(edges, dim=1)
            z = self.encoder(node_input, edge_index)
        else:
            z = node_input
        return self.readout(z, batch_index)

    def _node_features(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor | None,
        rel_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Raw embeddings, optionally with the spatial channels concatenated.

        Order and content match what GraphTransformer.forward assembles:
        rel_pos's three components, their norm, then the Laplacian PE. The norm
        is handed over rather than left to be learned because a ReLU MLP
        approximates sqrt(dx^2+dy^2+dz^2) poorly.
        """
        if not self.use_spatial_features:
            return x
        missing = []
        if self.use_lpe and pos_enc is None:
            missing.append("pos_enc")
        if self.use_position and rel_pos is None:
            missing.append("rel_pos")
        if missing:
            raise ValueError(
                f"use_spatial_features=True requires {' and '.join(missing)} "
                "(data/geodesic_window.py attaches both to every window)"
            )
        parts = [x] if self.use_embeddings else []
        if self.use_position:
            parts += [rel_pos, torch.linalg.norm(rel_pos, dim=-1, keepdim=True)]
        if self.use_lpe:
            parts.append(pos_enc)
        return torch.cat(parts, dim=-1)
