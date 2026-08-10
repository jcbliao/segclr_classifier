"""Graph Transformer with adjacency-conditioned ("AC") attention -- the
attention mechanism and transformer stack of Weis et al.'s GraphDINO
(https://github.com/marissaweis/ssl_neuron,
`ssl_neuron/graphdino.py`'s `GraphAttention`/`AttentionBlock`/`GraphTransformer`),
used here for supervised classification.

Only the architecture is taken, not GraphDINO's self-distillation
pretraining: no teacher/student encoder pair, no EMA weight update, no
centering, no temperature, no projector head. It is one encoder producing one
graph-level embedding, trained directly by gnn/lcpn.py::LCPNHead. See
gnn/model.py's `ModelConfig.architecture` for how it's wired in.

The bias mechanism is `GraphAttention`'s core trick: attention logits are a
per-node-predicted convex-ish combination of ordinary global dot-product
attention and a fixed adjacency bias --

    attn = gamma_0 * (QK^T / sqrt(d)) + gamma_1 * adj

with `gamma = predict_gamma(x)` (a 2-wide linear head, one weight pair per
query node, optionally passed through exp() so both weights stay positive)
learned per node per layer, so the model can decide per node whether to
behave more like a local message-passer (gamma_1 dominant, attending mostly
to 1-hop neighbors via the adjacency term) or a global transformer (gamma_0
dominant). `adj` includes self-loops, matching the reference's
`neighbors_to_adjacency_torch` convention.

Two deliberate adaptations beyond a literal port, both because this
project's unit is a small (~10-node-average) LOCAL WINDOW around one point,
not a whole skeleton subsampled to a fixed node count like GraphDINO's own
pipeline (`ssl_neuron/datasets.py::_reduce_nodes`/`subsample_graph`):

1. **Padding, not subsampling.** GraphDINO always forces every graph in a
   batch to the exact same `n_nodes` by dropping/duplicating nodes, so it has
   no padding concept at all. This project deliberately never shrinks a
   window (CLAUDE.md: "context window sizes held fixed at the baseline's
   values... only the aggregation/readout changes"), so variable-size
   windows are padded to each batch's own max size instead
   (`torch_geometric.utils.to_dense_batch`/`to_dense_adj`), and
   `GraphAttention` takes an explicit `key_padding_mask` to null out
   attention to padding regardless of what gamma/adj would otherwise say --
   without it, padded (all-zero) key positions could still receive nonzero
   softmax weight from the unconstrained global-attention term.
2. **Positional encoding is precomputed per window**, not batched. GraphDINO
   computes the Laplacian PE (`compute_eig_lapl_torch_batch`) on an already
   uniform-size padded batch. Doing that here -- eigendecomposing one shared
   (B, N, N) matrix built from many different-size real windows padded
   together -- would mix each window's real eigenvectors with spurious ones
   from the zero-adjacency padding block. So the PE is computed once per
   window at extraction time instead (`data/geodesic_window.py`'s
   `_window_laplacian_pos_enc`), on that window's own small real adjacency,
   and just rides along as a per-node `Data.pos_enc` attribute like `x` does.

**Relative position feature.** Each node also carries `rel_pos`, its raw xyz
coordinate minus the window's center node's coordinate (so the center sits at
(0,0,0), every other node at a translation-invariant offset from it) --
computed once per window alongside
`pos_enc` (`data/geodesic_window.py::extract_window_subgraph`, from the
whole-cell `Data.pos` `data/build_dataset_from_store.py` already caches) and
concatenated onto `x` before `to_node_embedding`. Rationale: a SegCLR
embedding's context window is a function of its xyz source position
(CLAUDE.md's project-goal section), but nothing in the embeddings themselves
encodes WHERE within the window a node sits relative to the point being
classified -- this gives the model that directly, on top of whatever the
Laplacian PE's purely structural (topology-only, no metric distances) signal
already provides.
"""

from __future__ import annotations

import torch
from torch import nn

# Bucket boundaries (nm) for the learned distance bias, chosen from the
# measured edge-length distribution over 2.1M real skeleton edges
# (scripts/check_edge_length_distribution.py): p5 529, p50 1867, p95 3865,
# p99 6117. Finer where the mass is, coarser in the long right tail.
DIST_BIAS_BOUNDARIES_NM = (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 6000.0)

# Bucket 0 = "no edge", bucket 1 = self-loop, buckets 2.. = binned edge length
# (torch.bucketize over the boundaries above yields len+1 bins).
N_DIST_BUCKETS = len(DIST_BIAS_BOUNDARIES_NM) + 3


class GraphAttention(nn.Module):
    """Dense attention over one padded batch of windows, with two independent
    switches for how graph structure enters:

    - `use_adj_bias` (default True): the ported GraphDINO mechanism, adding a
      learned multiple of the adjacency matrix to the attention logits --
      `gamma_0 * attn + gamma_1 * adj`, see the module docstring. Turning it
      OFF drops `predict_gamma` entirely and leaves plain scaled dot-product
      attention, not `gamma_0 * attn` with the adjacency term zeroed -- a
      per-node learned temperature is itself part of the mechanism being
      ablated, so it goes too.
    - `attention_scope` ("global" default, or "neighborhood"): whether a node
      may attend anywhere in the window or only to its graph neighbors. The
      adjacency BIAS is soft (every node stays reachable, neighbors just get
      a head start); neighborhood scope is a hard `-inf` mask.

    Note this keeps the reference's unusual head convention: each head
    operates at the FULL `dim` width (not `dim / num_heads`), and heads are
    concatenated (`dim * num_heads`) before a final projection back to
    `dim` -- ported as-is, not "fixed" to conventional multi-head splitting.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        use_exp: bool = True,
        use_adj_bias: bool = True,
        use_dist_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim**-0.5
        self.use_exp = use_exp
        self.use_adj_bias = use_adj_bias

        self.qkv_projection = nn.Linear(dim, dim * num_heads * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim)

        # Per-node trade-off between local (adjacency-biased) and global
        # attention. Initialized close to 0 so exp(gamma) starts close to 1
        # for both terms -- attn and adj contribute roughly equally at init,
        # matching the reference's own init. Not built at all when the
        # adjacency bias is off, so an ablated run carries no dead parameters.
        self.predict_gamma = nn.Linear(dim, 2) if use_adj_bias else None
        if self.predict_gamma is not None:
            self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        # Learned per-head scalar bias indexed by binned edge length, replacing
        # the binary adjacency in the bias term. Initialized so it is EXACTLY
        # the binary adjacency at step 0 -- every bucket 1.0 except "no edge"
        # at 0.0 -- so switching this on starts from the unmodified model and
        # learns away from it, rather than perturbing the initialization.
        self.use_dist_bias = use_dist_bias
        self.dist_bias = nn.Embedding(N_DIST_BUCKETS, num_heads) if use_dist_bias else None
        if self.dist_bias is not None:
            nn.init.ones_(self.dist_bias.weight)
            self.dist_bias.weight.data[0].zero_()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dist_bucket: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, N, dim)
        adj: (B, N, N) -- 0/1 (or float) adjacency, self-loops included.
            Unused when `use_adj_bias` is False, and superseded by
            `dist_bucket` when `use_dist_bias` is True.
        dist_bucket: (B, N, N) long -- per-pair distance bucket index, from
            GraphTransformer._distance_buckets. Required iff `use_dist_bias`.
        key_padding_mask: (B, N) bool -- True where the node is REAL (not
            padding). Applied over the key axis so padded nodes never receive
            attention probability, independent of gamma/adj.
        attn_mask: (B, N, N) bool or None -- True where a query may attend to
            a key. Supplied by GraphTransformer under neighborhood scope; None
            means unrestricted. Its diagonal is always True (see the caller),
            so no query row can end up fully -inf and produce NaN after
            softmax.
        """
        B, N, C = x.shape
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3)  # each (B, H, N, dim)

        attn = (query @ key.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if self.use_adj_bias:
            gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)  # (B, H, N, 2)
            if self.use_exp:
                gamma = torch.exp(gamma)
            if self.use_dist_bias:
                if dist_bucket is None:
                    raise ValueError("use_dist_bias=True but dist_bucket was not provided")
                # (B, N, N, H) -> (B, H, N, N). Per head, unlike gamma, which
                # predict_gamma broadcasts identically across heads -- so this
                # is the only part of the bias term a single head can shape on
                # its own (one head local, another global).
                adj_b = self.dist_bias(dist_bucket).permute(0, 3, 1, 2)
            else:
                adj_b = adj[:, None].repeat(1, self.num_heads, 1, 1)  # (B, H, N, N)
            attn = gamma[:, :, :, 0:1] * attn + gamma[:, :, :, 1:2] * adj_b

        pad = ~key_padding_mask[:, None, None, :]  # (B, 1, 1, N), True at padding
        attn = attn.masked_fill(pad, float("-inf"))
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[:, None], float("-inf"))  # (B, 1, N, N)

        attn = attn.softmax(dim=-1)

        x = (attn @ value).transpose(1, 2).reshape(B, N, -1)  # (B, N, H*dim)
        return self.proj(x)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    """Pre-norm block: norm -> AC-attention -> residual, norm -> MLP ->
    residual. LayerNorm, per the reference's `norm_layer: Any = nn.LayerNorm`
    default -- never BatchNorm, which would mix statistics across unrelated
    nodes/windows in a batch."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        use_exp: bool = True,
        use_adj_bias: bool = True,
        use_dist_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_exp=use_exp,
            use_adj_bias=use_adj_bias, use_dist_bias=use_dist_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim=dim, hidden_dim=dim * mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dist_bucket: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), adj, key_padding_mask, attn_mask, dist_bucket)
        x = x + self.mlp(self.norm2(x))
        return x


class GraphTransformer(nn.Module):
    """Fused encoder + CLS readout for one window: raw SegCLR node features
    (+ optional Laplacian positional encoding) in, one graph-level embedding
    out, feeding gnn/lcpn.py::LCPNHead directly. There is no projector/DINO
    head -- the CLS token's post-block, post-`mlp_head` state IS the returned
    embedding.

    Four independent ablation switches, all defaulting to the full model, so
    each can be turned off without touching the others (see
    gnn/model.py's `gt_use_lpe` / `gt_attention_scope` / `gt_use_adj_bias` /
    `gt_use_rel_pos` and scripts/train_gnn.py's matching flags):

      use_lpe          add the per-window Laplacian PE to the node embedding
      use_rel_pos      concatenate center-relative geometry onto the node
                       features: dx, dy, dz and their norm (4 channels)
      use_thickness    concatenate the spine-corrected dendrite shaft radius
                       (+ its measured flag) onto the node features
      use_adj_bias     GraphDINO's `gamma_1 * adj` additive attention bias
      use_dist_bias    replace that binary `adj` with a learned per-head scalar
                       indexed by binned edge length (requires use_adj_bias)
      attention_scope  "global" (attend anywhere in the window) vs
                       "neighborhood" (hard-masked to graph neighbors)

    `use_thickness` defaults to OFF, unlike the other three -- it needs
    data/dendrite_thickness_cache/*.npz to have been ingested, and the dataset
    has to be constructed with `use_thickness=True` to attach the feature at
    all (scripts/train_gnn.py drives both from one flag so they cannot drift
    apart).

    **Interaction worth knowing before reading an ablation grid:** under
    `attention_scope="neighborhood"` the adjacency bias is nearly inert. The
    hard mask already restricts every node row to exactly the positions where
    `adj == 1`, so adding `gamma_1 * adj` contributes the same constant to
    every surviving logit in the row and cancels in the softmax. What still
    differs is `gamma_0`, the learned per-node temperature that rides along
    with the bias term. So "neighborhood + bias" vs. "neighborhood, no bias"
    is a temperature ablation, not a structure ablation -- the structure
    comparison you probably want is against `attention_scope="global"`.
    """

    # Width of the relative-position feature (data/geodesic_window.py's
    # rel_pos: xyz offset from the window's center node, center-relative not
    # absolute) concatenated onto the raw node features before
    # to_node_embedding -- see forward()'s docstring.
    # dx, dy, dz, ||(dx,dy,dz)||. The norm is a derived 4th channel, not a
    # separate feature: it is computed from the other three in forward()
    # rather than cached, and shares their on/off switch. It is handed over
    # explicitly because a ReLU MLP approximates sqrt(dx^2+dy^2+dz^2) badly --
    # a smooth convex function of three inputs, which a piecewise-linear
    # network can only tile with planes. Direction and distance are
    # complementary rather than redundant: the components keep orientation
    # (apical trunks run pia-ward), the norm makes "how far out in the window"
    # directly legible.
    REL_POS_DIM = 4

    # Width of the dendrite-thickness feature: [normalized shaft radius,
    # measured flag]. Mirrors data/geodesic_window.py::THICKNESS_DIM -- see
    # that constant for why the measured flag is not optional.
    THICKNESS_DIM = 2

    def __init__(
        self,
        feat_dim: int,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        pos_dim: int = 8,
        use_exp: bool = True,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        use_lpe: bool = True,
        use_rel_pos: bool = True,
        use_thickness: bool = False,
        use_adj_bias: bool = True,
        use_dist_bias: bool = False,
        attention_scope: str = "global",
    ):
        super().__init__()
        if attention_scope not in ("global", "neighborhood"):
            raise ValueError(
                f"unknown attention_scope {attention_scope!r}; expected 'global' or 'neighborhood'"
            )
        self.use_lpe = use_lpe
        self.use_rel_pos = use_rel_pos
        self.use_thickness = use_thickness
        self.use_dist_bias = use_dist_bias
        if use_dist_bias and not use_adj_bias:
            # The distance bias IS the bias term's multiplicand -- with
            # use_adj_bias off there is no bias term for it to occupy, so this
            # combination would silently do nothing.
            raise ValueError("use_dist_bias=True requires use_adj_bias=True")
        self.register_buffer(
            "_dist_boundaries", torch.tensor(DIST_BIAS_BOUNDARIES_NM), persistent=False
        )
        self.attention_scope = attention_scope

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        # Must stay in lockstep with the concatenation order in forward().
        node_in_dim = (
            feat_dim
            + (self.REL_POS_DIM if use_rel_pos else 0)
            + (self.THICKNESS_DIM if use_thickness else 0)
        )
        self.to_node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, dim * 2), nn.ReLU(True), nn.Linear(dim * 2, dim)
        )
        # Not built when the LPE is off, so an ablated run carries no dead
        # parameters (same convention as GraphAttention.predict_gamma).
        self.to_pos_embedding = nn.Linear(pos_dim, dim) if use_lpe else None

        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    use_exp=use_exp, use_adj_bias=use_adj_bias, use_dist_bias=use_dist_bias,
                )
                for _ in range(depth)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

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
        """
        x: (N_total, feat_dim) -- raw node features, PyG-batched (concatenated
            across windows along dim 0), same convention every other module
            in this repo uses.
        edge_index: (2, E) -- already-symmetrized local window edges.
        batch_index: (N_total,) -- which window (0..B-1) each node belongs to.
        pos_enc: (N_total, pos_dim) -- precomputed per-window Laplacian PE,
            see data/geodesic_window.py::_window_laplacian_pos_enc. Batched
            the same way x is (a plain per-node attribute PyG concatenates).
            Required iff `use_lpe`; ignored otherwise.
        rel_pos: (N_total, 3) -- precomputed per-window relative xyz position
            (window's center node at (0,0,0), see
            data/geodesic_window.py::extract_window_subgraph), scaled to a
            roughly unit range. Concatenated onto `x` before
            to_node_embedding, not added like pos_enc is -- it's a raw
            geometric feature of the node itself (like the SegCLR embedding
            it rides alongside), not a structural/positional signal meant to
            play the same additive role as the Laplacian PE. Required iff
            `use_rel_pos`; ignored otherwise. Expands to 4 channels here --
            the three components plus their norm.
        thickness: (N_total, THICKNESS_DIM) -- normalized spine-corrected
            dendrite shaft radius plus its measured flag, already NaN-masked
            by data/dataset_windowed.py::load_thickness_features. Concatenated
            onto `x` alongside rel_pos, for the same reason: it is a physical
            property of the node, not a positional signal. Required iff
            `use_thickness`; ignored otherwise.
        edge_attr: (E, 1) or (E,) -- skeleton edge length in nm. Required iff
            `use_dist_bias`; ignored otherwise. Note this is the ONLY consumer
            of edge_attr anywhere in gnn/ -- the binary-adjacency bias and the
            MPNN's SAGEConv both discard it.

        Returns g: (B, dim), one embedding per window.
        """
        from torch_geometric.utils import to_dense_adj, to_dense_batch

        x_dense, node_mask = to_dense_batch(x, batch_index)  # (B, N, feat_dim), (B, N)
        B, N, _ = x_dense.shape

        adj_raw = to_dense_adj(edge_index, batch_index, max_num_nodes=N)  # (B, N, N), 0/1
        self_loops = torch.diag_embed(node_mask.to(adj_raw.dtype))
        adj = torch.maximum(adj_raw, self_loops)  # self-loops on real nodes only

        node_parts = [x_dense]
        if self.use_rel_pos:
            if rel_pos is None:
                raise ValueError("use_rel_pos=True but rel_pos was not provided")
            rel_pos_dense, _ = to_dense_batch(rel_pos, batch_index)  # (B, N, 3)
            # rel_pos is already scaled by REL_POS_SCALE_NM, so its norm is on
            # the same O(1) scale -- no second constant needed. Padding rows
            # are all-zero and stay at distance 0; they are masked out below
            # regardless.
            node_parts.append(rel_pos_dense)
            node_parts.append(torch.linalg.norm(rel_pos_dense, dim=-1, keepdim=True))
        if self.use_thickness:
            if thickness is None:
                raise ValueError(
                    "use_thickness=True but thickness was not provided -- build the dataset "
                    "with WindowedGraphDatasetLCPN(..., use_thickness=True)"
                )
            thickness_dense, _ = to_dense_batch(thickness, batch_index)  # (B, N, 2)
            node_parts.append(thickness_dense)
        node_input = torch.cat(node_parts, dim=-1) if len(node_parts) > 1 else x_dense

        node_emb = self.to_node_embedding(node_input)
        if self.use_lpe:
            if pos_enc is None:
                raise ValueError("use_lpe=True but pos_enc was not provided")
            pos_dense, _ = to_dense_batch(pos_enc, batch_index)  # (B, N, pos_dim)
            node_emb = node_emb + self.to_pos_embedding(pos_dense)
        node_emb = node_emb * node_mask.unsqueeze(-1)  # zero out padding rows

        cls = (self.cls_token + self.cls_pos_embedding).expand(B, -1, -1)
        h = torch.cat([cls, node_emb], dim=1)  # (B, N+1, dim)
        h = self.dropout(h)

        full_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=x.device), node_mask], dim=1
        )  # (B, N+1) -- CLS is always "real"

        # CLS row/col: self-loop only (matches the reference's adj_cls[:,0,0]
        # = 1, rest 0 -- CLS's local/adjacency bias term never singles out
        # any one node; it still reaches every real node through the
        # unconstrained global-attention term).
        adj_full = x.new_zeros(B, N + 1, N + 1)
        adj_full[:, 0, 0] = 1.0
        adj_full[:, 1:, 1:] = adj

        attn_mask = self._neighborhood_mask(adj_full, N, B) if self.attention_scope == "neighborhood" else None

        dist_bucket = None
        if self.use_dist_bias:
            if edge_attr is None:
                raise ValueError("use_dist_bias=True but edge_attr was not provided")
            dist = to_dense_adj(
                edge_index, batch_index, edge_attr=edge_attr.reshape(-1), max_num_nodes=N
            )  # (B, N, N), edge length in nm at adjacent pairs, 0 elsewhere
            dist_bucket = self._distance_buckets(adj_raw, dist, node_mask)

        for block in self.blocks:
            h = block(h, adj_full, full_mask, attn_mask, dist_bucket)

        return self.mlp_head(h[:, 0])

    def _distance_buckets(
        self, adj_raw: torch.Tensor, dist: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, N+1, N+1) long bucket index per pair, laid out on the same
        CLS-prepended grid as adj_full.

        Bucket 0 is "no edge", 1 is a self-loop, and 2.. are binned edge
        lengths. Distinguishing 0 from the shortest length bin is why this
        reads `adj_raw` (pre-self-loop adjacency) rather than inferring
        absence from `dist == 0` -- a genuinely zero-length edge and a missing
        edge are different facts.

        CLS keeps bucket 0 to every node and 1 to itself, matching adj_full's
        convention that CLS gets no adjacency bias toward any particular node
        and reaches them through the unconstrained attention term instead.
        """
        B, N, _ = adj_raw.shape
        bucket = torch.bucketize(dist, self._dist_boundaries) + 2  # (B, N, N)
        bucket = torch.where(adj_raw > 0, bucket, torch.zeros_like(bucket))
        # Self-loops on real nodes only; padding diagonals stay "no edge" so a
        # padding row never claims a bias it should not have (it is masked out
        # by key_padding_mask regardless, but keeping the two consistent means
        # the bucket grid can be read on its own without that caveat).
        bucket.diagonal(dim1=1, dim2=2).copy_(node_mask.long())

        full = bucket.new_zeros(B, N + 1, N + 1)
        full[:, 1:, 1:] = bucket
        full[:, 0, 0] = 1
        return full

    @staticmethod
    def _neighborhood_mask(adj_full: torch.Tensor, N: int, B: int) -> torch.Tensor:
        """Hard attention mask for `attention_scope="neighborhood"`: True
        where a query may attend to a key.

        Node-to-node is restricted to `adj_full` (1-hop neighbors plus each
        node's own self-loop). The CLS row and column are forced fully open,
        which is not cosmetic -- `adj_full` gives CLS a self-loop and nothing
        else, so a purely adjacency-derived mask would leave CLS attending
        only to itself, and since CLS's final state IS the returned
        embedding, every window would collapse to the same constant vector.
        Padding keys stay blocked by `key_padding_mask` regardless of what
        this allows.

        The diagonal is forced True for every position including padding
        rows, so no query row is ever entirely -inf -- an all-masked row
        would come out of softmax as NaN.
        """
        mask = adj_full > 0  # (B, N+1, N+1) bool
        mask[:, 0, :] = True  # CLS reads every position
        mask[:, :, 0] = True  # every node can read CLS
        idx = torch.arange(N + 1, device=adj_full.device)
        mask[:, idx, idx] = True
        return mask
