"""Import + forward-pass smoke test for gnn/* on synthetic data -- catches
shape/API bugs before spending a full GPU allocation on the real pipeline. No
real data needed. Run via sbatch (mit_normal_gpu -- project policy: all
training/eval/inference runs on GPU nodes, so this also confirms the model
actually runs correctly on CUDA, not just CPU).

Covers every aggregation method (--architecture graph_transformer / mpnn /
mpnn_complete / pointwise_mlp / linear / mean, see gnn/model.py) against the real LAB_HIERARCHY_TREE
(gnn/hierarchy.py) rather than a toy tree -- cheap to do and exercises the
actual depth-5, 24-leaf structure the real pipeline uses, not just a 2-level
stand-in that might hide bugs the real tree would trigger (e.g. the
non_neuron/non_neuron/glia repeated-name branch, or depth-padding for the
single-child thalamocortical branch).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from data.dataset_lcpn import HIERARCHY_LEVELS_DROPPED  # noqa: E402
from data.geodesic_window import REL_POS_SCALE_NM, _window_laplacian_pos_enc  # noqa: E402
from gnn.hierarchy import (  # noqa: E402
    LAB_HIERARCHY_TREE,
    parse_hierarchy,
    truncate_hierarchy,
)
from gnn.metrics import summarize  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

GT_POS_DIM = 8  # matches ModelConfig.gt_pos_dim's default


def random_window(n_nodes: int, d: int, seed: int, pos_dim: int = GT_POS_DIM) -> Data:
    """A synthetic window: a chain graph, like a small stretch of skeleton,
    carrying the same per-node attributes data/geodesic_window.py::
    extract_window_subgraph attaches on real windows."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, d, generator=g)
    src = torch.arange(n_nodes - 1)
    dst = torch.arange(1, n_nodes)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    # Edge length in nm on a realistic scale (real p5-p95 is ~530-3900nm).
    # Nothing in gnn/ reads it, but real cached windows carry it, so batching
    # it here keeps the synthetic Data faithful to what the model is handed.
    half = torch.rand(n_nodes - 1, 1, generator=g) * 5000 + 200
    edge_attr = torch.cat([half, half], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # The real per-window Laplacian PE, computed by the same function the real
    # pipeline calls rather than a stand-in.
    data.pos_enc = _window_laplacian_pos_enc(edge_index, n_nodes, pos_dim)
    # Synthetic absolute xyz (like data.pos on a real cached cell) minus the
    # center node's (index 0, same convention as real windows), scaled the
    # same way.
    synthetic_pos = torch.randn(n_nodes, 3, generator=g) * 5000  # nm-scale, like real coords
    data.rel_pos = (synthetic_pos - synthetic_pos[0]).float() / REL_POS_SCALE_NM
    # TEASAR windows contain real routing nodes without SegCLR. The centre is
    # always observed; alternating later nodes exercise both modalities.
    data.has_segclr = torch.arange(n_nodes) % 2 == 0
    # Dendrite thickness, in the same [normalized radius, measured flag] shape
    # data/dataset_windowed.py::load_thickness_features produces. Some nodes
    # are deliberately marked unmeasured (flag 0, radius 0) -- that is the
    # cache's normal state for axon nodes, branch points and mesh holes, so a
    # smoke test with everything measured would not exercise the real case.
    measured = (torch.rand(n_nodes, generator=g) > 0.3).float()
    data.thickness = torch.stack([torch.rand(n_nodes, generator=g) * measured, measured], dim=1)
    return data


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device visible -- this job was expected to run on mit_normal_gpu")

    hierarchy = truncate_hierarchy(
        parse_hierarchy(LAB_HIERARCHY_TREE), HIERARCHY_LEVELS_DROPPED
    )
    print(f"hierarchy: depth={hierarchy.depth}, level sizes={[len(c) for c in hierarchy.level_classes]}")
    granular_labels = sorted(hierarchy.label_paths)

    torch.manual_seed(0)
    d = 64  # raw segclr_db resnet_860b_reshuffled embedding dim

    # Deliberately uneven, small sizes (including a 1-node graph) to exercise
    # the padding/masking path (to_dense_batch pads every graph in the batch
    # to this batch's max size) and _window_laplacian_pos_enc's zero-pad tail
    # for graphs with fewer than gt_pos_dim nontrivial eigenmodes -- both real
    # possibilities for actual small geodesic windows.
    windows = [random_window(n, d, seed=100 + i) for i, n in enumerate([1, 3, 20, 7])]
    for i, w in enumerate(windows):
        label = granular_labels[i % len(granular_labels)]
        path = hierarchy.label_paths[label]
        y_levels = [hierarchy.level_maps[lvl][path[lvl]] for lvl in range(hierarchy.depth)]
        w.y_levels = torch.tensor(y_levels, dtype=torch.long).unsqueeze(0)
    batch = Batch.from_data_list(windows).to(device)
    targets = batch.y_levels  # (B, depth)
    print(
        f"batch: {batch.num_graphs} windows (sizes {[w.x.shape[0] for w in windows]}), "
        f"{batch.num_nodes} nodes total, targets shape={tuple(targets.shape)}"
    )

    # The relative-position convention itself, checked before trusting any
    # model output: each window's own center node (local index 0, i.e.
    # batch.ptr[:-1] after batching) must sit at exactly (0,0,0).
    centers = batch.ptr[:-1]
    assert torch.allclose(batch.rel_pos[centers], torch.zeros_like(batch.rel_pos[centers])), (
        f"center node's rel_pos should be exactly 0 -- got {batch.rel_pos[centers]}"
    )

    preds = None
    for architecture, expected_dim in (
        ("graph_transformer", 32), ("graph_transformer_teasar", 32),
        ("mpnn", 48), ("pointwise_mlp", 40), ("linear", 36),
        ("mean", d),
    ):
        print(f"\n--- architecture={architecture} ---")
        config = ModelConfig(
            in_dim=d, architecture=architecture,
            mpnn_hidden_dim=48, mpnn_out_dim=48, mpnn_layers=2,
            pointwise_mlp_hidden_dim=40, pointwise_mlp_out_dim=40, pointwise_mlp_layers=2,
            linear_out_dim=36,
            gt_dim=32, gt_depth=2, gt_heads=2, gt_pos_dim=GT_POS_DIM,
        )
        model = WindowClassifier(config, hierarchy=hierarchy).to(device)
        if architecture in ("graph_transformer", "graph_transformer_teasar"):
            assert model.readout is None, "graph_transformer should not build a separate readout"
            assert model.encoder is None, "graph_transformer should not build an MPNN encoder"
        else:
            assert model.graph_transformer is None, f"{architecture} should not build a GraphTransformer"
            # Zero-parameter readout in both cases: for "mean" that means every
            # trainable weight belongs to the classification head, which is what
            # makes it a clean baseline; for "mpnn" all the aggregation
            # parameters live in the encoder, not the pooling step.
            assert not list(model.readout.parameters()), "MeanReadout should have no parameters"
            assert (model.encoder is not None) == (architecture == "mpnn"), (
                f"{architecture}: MPNN encoder built={model.encoder is not None}"
            )
            assert (model.pointwise_mlp is not None) == (architecture == "pointwise_mlp"), (
                f"{architecture}: Pointwise MLP phi built={model.pointwise_mlp is not None}"
            )
            assert (model.linear is not None) == (architecture == "linear"), (
                f"{architecture}: linear phi built={model.linear is not None}"
            )

        g = model(
            batch.x, batch.edge_index, batch.batch,
            pos_enc=batch.pos_enc, rel_pos=batch.rel_pos,
            has_segclr=batch.has_segclr,
        )
        assert g.shape == (batch.num_graphs, expected_dim), g.shape
        assert torch.isfinite(g).all(), "non-finite embedding -- likely a padding-mask bug"

        cls_loss = model.cls_head.compute_loss(g, targets)
        preds = model.cls_head.predict_top_down(g)
        assert preds.shape == targets.shape, (preds.shape, targets.shape)
        print(
            f"  embedding shape={tuple(g.shape)}  cls_loss={cls_loss.item():.4f}  "
            f"preds[:, -1]={preds[:, -1].tolist()}"
        )
        cls_loss.backward()
        for aggregator in model._aggregator_modules():
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in aggregator.parameters()
            ), f"{architecture}'s aggregator got no gradient from the classification loss"

        if architecture == "graph_transformer_teasar":
            # Values stored in unavailable rows must be completely gated: the
            # learned missing token, geometry and LPE determine those nodes.
            changed_x = batch.x.detach().clone()
            changed_x[~batch.has_segclr] = 1e6
            model.eval()
            with torch.no_grad():
                changed = model(
                    changed_x, batch.edge_index, batch.batch,
                    pos_enc=batch.pos_enc, rel_pos=batch.rel_pos,
                    has_segclr=batch.has_segclr,
                )
                original = model(
                    batch.x, batch.edge_index, batch.batch,
                    pos_enc=batch.pos_enc, rel_pos=batch.rel_pos,
                    has_segclr=batch.has_segclr,
                )
            assert torch.allclose(changed, original), (
                "unobserved SegCLR values leaked through the availability mask"
            )

    # Flat decoder: one finest-level softmax, with hierarchy-shaped output
    # reconstructed only after the unconstrained leaf argmax for API parity.
    flat = WindowClassifier(
        ModelConfig(
            in_dim=d, architecture="mpnn", classifier="flat",
            mpnn_hidden_dim=48, mpnn_out_dim=48, mpnn_layers=2,
            gt_pos_dim=GT_POS_DIM,
        ),
        hierarchy=hierarchy,
    ).to(device)
    flat_g = flat(
        batch.x, batch.edge_index, batch.batch,
        pos_enc=batch.pos_enc, rel_pos=batch.rel_pos,
    )
    flat_loss = flat.cls_head.compute_loss(flat_g, targets)
    flat_preds = flat.cls_head.predict_top_down(flat_g)
    assert flat_preds.shape == targets.shape
    # Every reconstructed prediction must be one of the hierarchy's valid
    # paths, even though no coarse prediction constrained the leaf argmax.
    valid_paths = {tuple(row) for row in flat.cls_head.leaf_paths.cpu().tolist()}
    assert all(tuple(row) in valid_paths for row in flat_preds.cpu().tolist())
    flat_loss.backward()
    assert flat.cls_head.head.weight.grad is not None
    print(f"\n--- flat decoder ---\n  loss={flat_loss.item():.4f} paths valid")

    # The GraphTransformer path needs pos_enc/rel_pos and must say so loudly
    # rather than silently producing a meaningless embedding.
    gt_model = WindowClassifier(
        ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=2, gt_heads=2),
        hierarchy=hierarchy,
    ).to(device)
    try:
        gt_model(batch.x, batch.edge_index, batch.batch)
    except ValueError as e:
        print(f"\nmissing pos_enc/rel_pos correctly rejected: {e}")
    else:
        raise AssertionError("architecture='graph_transformer' should require pos_enc/rel_pos")

    # --- GraphTransformer ablation switches ---------------------------------
    # Each switch on its own, plus everything off at once, plus neighborhood
    # scope. The 1-node window in this batch is the interesting case for
    # neighborhood scope: its only neighbor is itself, so its attention row
    # would be entirely -inf without the forced diagonal, and CLS would be
    # cut off from every node without the forced CLS row/column.
    print("\n--- graph_transformer ablation switches ---")
    ablations = {
        "full (all on)": {},
        "no LPE": {"gt_use_lpe": False},
        "no rel_pos": {"gt_use_rel_pos": False},
        "no adj bias": {"gt_use_adj_bias": False},
        "neighborhood attention": {"gt_attention_scope": "neighborhood"},
        "neighborhood + no adj bias": {
            "gt_attention_scope": "neighborhood", "gt_use_adj_bias": False,
        },
        "thickness on": {"gt_use_thickness": True},
        "thickness, no rel_pos": {"gt_use_thickness": True, "gt_use_rel_pos": False},
        "everything off": {
            "gt_use_lpe": False, "gt_use_rel_pos": False, "gt_use_adj_bias": False,
        },
    }
    for name, overrides in ablations.items():
        cfg = ModelConfig(
            in_dim=d, architecture="graph_transformer",
            gt_dim=32, gt_depth=2, gt_heads=2, gt_pos_dim=GT_POS_DIM,
            **overrides,
        )
        m = WindowClassifier(cfg, hierarchy=hierarchy).to(device)

        # A disabled switch must drop its parameters, not just skip them at
        # runtime -- otherwise an "ablated" model still carries (and an
        # optimizer still allocates state for) weights that never see gradient.
        assert (m.graph_transformer.to_pos_embedding is not None) == cfg.gt_use_lpe
        for blk in m.graph_transformer.blocks:
            assert (blk.attn.predict_gamma is not None) == cfg.gt_use_adj_bias
        # Input width must track exactly which node features are switched on.
        expected_in = (
            d
            + (4 if cfg.gt_use_rel_pos else 0)  # dx, dy, dz, ||.||
            + (2 if cfg.gt_use_thickness else 0)
        )
        assert m.graph_transformer.to_node_embedding[0].in_features == expected_in, (
            f"{name}: to_node_embedding expects "
            f"{m.graph_transformer.to_node_embedding[0].in_features}, want {expected_in}"
        )

        # Pass both inputs regardless; a disabled switch must simply ignore its
        # input rather than depend on the caller withholding it.
        g = m(batch.x, batch.edge_index, batch.batch,
              pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, thickness=batch.thickness)
        assert g.shape == (batch.num_graphs, 32), g.shape
        assert torch.isfinite(g).all(), f"{name}: non-finite embedding (NaN from an all-masked row?)"

        loss = m.cls_head.compute_loss(g, targets)
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.graph_transformer.parameters()
        ), f"{name}: graph_transformer got no gradient"

        n_params = sum(p.numel() for p in m.graph_transformer.parameters())
        print(f"  {name:<28} loss={loss.item():.4f}  gt_params={n_params}")

    # A disabled switch must not merely ignore its input -- it must not need
    # it at all, so an ablated run can be driven without ever computing it.
    m = WindowClassifier(
        ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=2,
                    gt_heads=2, gt_use_lpe=False, gt_use_rel_pos=False),
        hierarchy=hierarchy,
    ).to(device)
    g = m(batch.x, batch.edge_index, batch.batch)  # no pos_enc, no rel_pos
    assert g.shape == (batch.num_graphs, 32) and torch.isfinite(g).all()
    print("  LPE+rel_pos off runs with neither input supplied")

    # --- MPNN reads the raw embeddings and the graph, nothing else ----------
    # conv0's input width is exactly the embedding dim: no positional encoding,
    # no relative geometry. Structure reaches it only through edge_index.
    print("\n--- mpnn: inputs ---")
    m = WindowClassifier(
        ModelConfig(in_dim=d, architecture="mpnn", mpnn_hidden_dim=48, mpnn_out_dim=48,
                    mpnn_layers=2),
        hierarchy=hierarchy,
    ).to(device)
    conv0 = m.encoder.convs[0]
    assert conv0.in_channels == d, f"conv0 reads {conv0.in_channels}, want {d}"
    # Supplied anyway: they must be ignored, not silently consumed. Compared
    # under eval(), since the encoder's dropout would otherwise make two
    # forward passes differ for reasons that have nothing to do with the inputs.
    m.eval()
    with torch.no_grad():
        g_with = m(batch.x, batch.edge_index, batch.batch,
                   pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
        g_bare = m(batch.x, batch.edge_index, batch.batch)
    # Compared by magnitude, not bitwise: SAGEConv's scatter aggregation uses
    # CUDA atomics, whose summation order varies between calls, so two runs of
    # the same input differ in the last bits. A consumed pos_enc/rel_pos would
    # move the output by O(0.1), not O(1e-7).
    max_diff = (g_with - g_bare).abs().max().item()
    assert max_diff < 1e-5, (
        f"mpnn output changed by {max_diff:.2e} when pos_enc/rel_pos were supplied -- "
        "that is far above scatter nondeterminism, so something is consuming them"
    )
    m.train()

    g = m(batch.x, batch.edge_index, batch.batch)
    assert g.shape == (batch.num_graphs, 48), g.shape
    loss = m.cls_head.compute_loss(g, targets)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters()
    ), "mpnn encoder got no gradient"
    n_params = sum(p.numel() for p in m.encoder.parameters())
    print(f"  conv0 in_channels={conv0.in_channels} (= embedding dim)  "
          f"encoder_params={n_params}  loss={loss.item():.4f}")
    print(f"  pos_enc / rel_pos are ignored: output unchanged with and without them "
          f"(max diff {max_diff:.1e})")

    # --- pointwise_mlp: a set, not a graph -------------------------------------
    # phi is 64 -> 128 -> 128 at the defaults, and the three properties that
    # make this Pointwise MLP rather than "an MLP somewhere in the model" are
    # checked directly: the widths, permutation invariance, and that neither
    # the graph nor the spatial features can reach it.
    print("\n--- pointwise_mlp: phi widths, permutation invariance, set-only inputs ---")
    m = WindowClassifier(
        ModelConfig(in_dim=d, architecture="pointwise_mlp"), hierarchy=hierarchy
    ).to(device)
    linears = [layer for layer in m.pointwise_mlp.phi if isinstance(layer, torch.nn.Linear)]
    widths = [linears[0].in_features] + [layer.out_features for layer in linears]
    assert widths == [d, 128, 128], f"phi is {widths}, want [{d}, 128, 128]"
    # Every Linear is followed by a nonlinearity, the last one included --
    # without it phi's final Linear would commute with the mean and collapse
    # into the head, making the model shallower than the depth claims.
    assert isinstance(m.pointwise_mlp.phi[-1], torch.nn.GELU), "phi's last layer should be GELU"

    m.eval()
    with torch.no_grad():
        g_ref = m(batch.x, batch.edge_index, batch.batch)

        # Permutation invariance, the defining property. Permuting the node
        # order and its batch assignment together leaves every window's SET of
        # embeddings unchanged, so the readout must be unchanged too.
        perm = torch.randperm(batch.num_nodes, device=device)
        g_perm = m(batch.x[perm], batch.edge_index, batch.batch[perm])
        perm_diff = (g_perm - g_ref).abs().max().item()
        assert perm_diff < 1e-5, (
            f"pointwise_mlp is not permutation invariant: output moved by {perm_diff:.2e} "
            "when the node order was shuffled"
        )

        # The graph is not consulted at all, so a completely different (and
        # deliberately nonsensical, window-crossing) edge set must change
        # nothing. This is the check that would catch a future edit
        # accidentally routing pointwise_mlp through the encoder path -- a mere
        # column shuffle would not, since that is the same graph.
        scrambled = torch.randint(
            0, batch.num_nodes, batch.edge_index.shape, device=device, dtype=batch.edge_index.dtype
        )
        g_scrambled = m(batch.x, scrambled, batch.batch)
        edge_diff = (g_scrambled - g_ref).abs().max().item()
        assert edge_diff < 1e-5, (
            f"pointwise_mlp output moved by {edge_diff:.2e} when edge_index was scrambled -- "
            "something is reading the graph"
        )

        # Same for the spatial channels: supplied anyway, they must be ignored
        # rather than silently consumed.
        g_spatial = m(batch.x, batch.edge_index, batch.batch,
                      pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, thickness=batch.thickness)
        spatial_diff = (g_spatial - g_ref).abs().max().item()
        assert spatial_diff < 1e-5, (
            f"pointwise_mlp output moved by {spatial_diff:.2e} when pos_enc/rel_pos were supplied"
        )
    m.train()
    n_params = sum(p.numel() for p in m.pointwise_mlp.parameters())
    print(f"  phi widths={widths}  phi_params={n_params}")
    print(f"  invariant to node order (max diff {perm_diff:.1e}), to edge_index "
          f"({edge_diff:.1e}) and to pos_enc/rel_pos ({spatial_diff:.1e})")

    # Turning the set-only property off is refused at construction, not
    # quietly honoured -- otherwise a sweep script could hand this
    # architecture spatial features and still call the result Pointwise MLP.
    for name, overrides in (
        ("--spatial", {"use_spatial_features": True}),
        ("--no-embeddings", {"use_embeddings": False}),
    ):
        try:
            WindowClassifier(
                ModelConfig(in_dim=d, architecture="pointwise_mlp", **overrides),
                hierarchy=hierarchy,
            )
        except ValueError as e:
            print(f"  {name} with pointwise_mlp correctly rejected: {e}")
        else:
            raise AssertionError(f"{name} with architecture='pointwise_mlp' should raise")

    # --- linear: one Linear, and the mean absorbs it --------------------------
    # The same set-only contract as pointwise_mlp, plus the property that makes
    # this rung a control rather than a model: phi is a single Linear with no
    # activation, so mean(phi(x)) == phi(mean(x)) exactly. If that identity
    # ever stops holding, something nonlinear has been added and the run no
    # longer measures what its name claims.
    print("\n--- linear: single Linear, set-only, commutes with the mean ---")
    m = WindowClassifier(
        ModelConfig(in_dim=d, architecture="linear", linear_out_dim=36), hierarchy=hierarchy
    ).to(device)
    assert m.pointwise_mlp is None and m.encoder is None and m.graph_transformer is None, (
        "linear should build no other aggregation stage"
    )
    assert isinstance(m.linear.phi, torch.nn.Linear), "linear phi should be one nn.Linear"
    assert (m.linear.phi.in_features, m.linear.phi.out_features) == (d, 36), (
        f"linear phi is {m.linear.phi.in_features} -> {m.linear.phi.out_features}, want {d} -> 36"
    )

    m.eval()
    with torch.no_grad():
        g_ref = m(batch.x, batch.edge_index, batch.batch)

        perm = torch.randperm(batch.num_nodes, device=device)
        perm_diff = (
            m(batch.x[perm], batch.edge_index, batch.batch[perm]) - g_ref
        ).abs().max().item()
        assert perm_diff < 1e-5, f"linear is not permutation invariant ({perm_diff:.2e})"

        scrambled = torch.randint(
            0, batch.num_nodes, batch.edge_index.shape, device=device, dtype=batch.edge_index.dtype
        )
        edge_diff = (m(batch.x, scrambled, batch.batch) - g_ref).abs().max().item()
        assert edge_diff < 1e-5, f"linear read the graph ({edge_diff:.2e})"

        spatial_diff = (
            m(batch.x, batch.edge_index, batch.batch,
              pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, thickness=batch.thickness) - g_ref
        ).abs().max().item()
        assert spatial_diff < 1e-5, f"linear read the spatial channels ({spatial_diff:.2e})"

        # phi(mean(x)) computed the other way round, per window.
        pooled = torch.zeros(batch.num_graphs, d, device=device).index_add_(
            0, batch.batch, batch.x
        ) / torch.bincount(batch.batch, minlength=batch.num_graphs).unsqueeze(1)
        commute_diff = (m.linear(pooled) - g_ref).abs().max().item()
        assert commute_diff < 1e-4, (
            f"linear phi does not commute with the mean ({commute_diff:.2e}) -- a "
            "nonlinearity has crept into what is supposed to be one Linear"
        )
    m.train()
    print(f"  phi={d} -> 36, params={sum(p.numel() for p in m.linear.parameters())}")
    print(f"  invariant to node order ({perm_diff:.1e}), edge_index ({edge_diff:.1e}) and "
          f"pos_enc/rel_pos ({spatial_diff:.1e}); mean(phi(x)) == phi(mean(x)) "
          f"({commute_diff:.1e})")

    for name, overrides in (
        ("--spatial", {"use_spatial_features": True}),
        ("--no-embeddings", {"use_embeddings": False}),
    ):
        try:
            WindowClassifier(
                ModelConfig(in_dim=d, architecture="linear", **overrides), hierarchy=hierarchy
            )
        except ValueError as e:
            print(f"  {name} with linear correctly rejected: {e}")
        else:
            raise AssertionError(f"{name} with architecture='linear' should raise")

    # --- --spatial: matched node features for the cross-architecture sweep ---
    # With it on, mean/mpnn must read exactly in_dim + 4 (rel_pos + its norm) +
    # pos_dim, the same channels the GraphTransformer assembles internally.
    print("\n--- mean / mpnn: --spatial ---")
    spatial_dim = 4 + GT_POS_DIM
    for architecture, readout_dim in (("mpnn", 48), ("mean", d + spatial_dim)):
        cfg = ModelConfig(
            in_dim=d, architecture=architecture, mpnn_hidden_dim=48, mpnn_out_dim=48,
            mpnn_layers=2, gt_pos_dim=GT_POS_DIM, use_spatial_features=True,
        )
        m = WindowClassifier(cfg, hierarchy=hierarchy).to(device)
        assert m.spatial_dim == spatial_dim, (architecture, m.spatial_dim)
        if architecture == "mpnn":
            got = m.encoder.convs[0].in_channels
            assert got == d + spatial_dim, f"mpnn conv0 reads {got}, want {d + spatial_dim}"
        # Whatever sits directly on the readout is what consumes its width:
        # the shared ResNet trunk when one is built (the default), otherwise
        # the per-node heads themselves. Asserting on the heads unconditionally
        # measures cls_resnet_hidden instead of the readout, and so says
        # nothing about whether the spatial channels were concatenated.
        if m.cls_head.trunk is not None:
            consumers = [m.cls_head.trunk.input_layer]
        else:
            consumers = [
                head if isinstance(head, torch.nn.Linear) else head[0]
                for head in m.cls_head.heads
            ]
        for consumer in consumers:
            assert consumer.in_features == readout_dim, (
                f"{architecture}: the readout feeds a layer of width "
                f"{consumer.in_features}, want {readout_dim}"
            )

        g = m(batch.x, batch.edge_index, batch.batch,
              pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
        assert g.shape == (batch.num_graphs, readout_dim), g.shape
        assert torch.isfinite(g).all(), f"{architecture} --spatial: non-finite embedding"
        loss = m.cls_head.compute_loss(g, targets)
        loss.backward()
        print(f"  {architecture:<6} spatial_dim={m.spatial_dim}  readout={readout_dim}  "
              f"loss={loss.item():.4f}")

        # On means the inputs are required, not silently skipped.
        try:
            m(batch.x, batch.edge_index, batch.batch)
        except ValueError as e:
            print(f"    missing pos_enc/rel_pos correctly rejected: {e}")
        else:
            raise AssertionError(f"{architecture} --spatial should require pos_enc/rel_pos")

    # The GraphTransformer builds these itself, so the combination is refused
    # rather than silently double-counting the same channels.
    try:
        WindowClassifier(
            ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=1,
                        gt_heads=1, use_spatial_features=True),
            hierarchy=hierarchy,
        )
    except ValueError as e:
        print(f"  --spatial with graph_transformer correctly rejected: {e}")
    else:
        raise AssertionError("use_spatial_features with graph_transformer should raise")

    # --- classification head: linear probe vs. the lab's ResNet trunk --------
    # Orthogonal to `architecture`, so check it composes with every one rather
    # than only with the GraphTransformer it was added alongside.
    print("\n--- classification head: --cls-resnet across architectures ---")
    for architecture, expect_in in (
        ("graph_transformer", 32), ("mpnn", 48), ("pointwise_mlp", 40), ("linear", 36),
        ("mean", d),
    ):
        cfg = ModelConfig(
            in_dim=d, architecture=architecture,
            mpnn_hidden_dim=48, mpnn_out_dim=48, mpnn_layers=2,
            pointwise_mlp_hidden_dim=40, pointwise_mlp_out_dim=40, pointwise_mlp_layers=2,
            linear_out_dim=36,
            gt_dim=32, gt_depth=2, gt_heads=2, gt_pos_dim=GT_POS_DIM,
            cls_head_resnet=True, cls_resnet_hidden=24, cls_resnet_layers=2,
        )
        m = WindowClassifier(cfg, hierarchy=hierarchy).to(device)
        trunk = m.cls_head.trunk
        assert trunk is not None, f"{architecture}: --cls-resnet built no trunk"
        # The trunk must consume the readout width and every per-node head must
        # be sized to the TRUNK's output, not the readout's -- getting that
        # wrong is a silent shape bug only for architectures where the two
        # happen to be equal.
        assert trunk.input_layer.in_features == expect_in, (
            f"{architecture}: trunk expects {trunk.input_layer.in_features}, readout gives {expect_in}"
        )
        for head in m.cls_head.heads:
            first = head if isinstance(head, torch.nn.Linear) else head[0]
            assert first.in_features == 24, f"{architecture}: head reads {first.in_features}, want 24"

        g = m(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc,
              rel_pos=batch.rel_pos)
        loss = m.cls_head.compute_loss(g, targets)
        pr = m.cls_head.predict_top_down(g)
        assert pr.shape == targets.shape, (pr.shape, targets.shape)
        assert torch.isfinite(loss), f"{architecture}: non-finite loss with resnet head"
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk.parameters()), (
            f"{architecture}: resnet trunk got no gradient"
        )
        n = sum(p.numel() for p in m.cls_head.parameters())
        print(f"  {architecture:<18} loss={loss.item():.4f}  cls_head_params={n}")

    # The ResNet trunk is the default head, since every current run uses it;
    # the linear probe is what --no-cls-resnet asks for. Both directions are
    # checked, because a default that silently flipped would move every run's
    # capacity without changing its name.
    assert WindowClassifier(
        ModelConfig(in_dim=d, architecture="mean"), hierarchy=hierarchy
    ).cls_head.trunk is not None, "the ResNet trunk should be the default head"
    assert WindowClassifier(
        ModelConfig(in_dim=d, architecture="mean", cls_head_resnet=False), hierarchy=hierarchy
    ).cls_head.trunk is None, "cls_head_resnet=False should give a linear probe"
    print("  resnet trunk is the default head; cls_head_resnet=False gives the linear probe")

    # Thickness is off by default, so the common mistake is asking the model
    # for it while running a dataset that never attached it. That must fail
    # loudly rather than train on a silently absent feature.
    m = WindowClassifier(
        ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=2,
                    gt_heads=2, gt_use_thickness=True),
        hierarchy=hierarchy,
    ).to(device)
    try:
        m(batch.x, batch.edge_index, batch.batch,
          pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)  # no thickness
    except ValueError as e:
        print(f"  missing thickness correctly rejected: {e}")
    else:
        raise AssertionError("gt_use_thickness=True should require a thickness tensor")

    try:
        ModelConfig(in_dim=d, architecture="graph_transformer", gt_attention_scope="diagonal")
        WindowClassifier(
            ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=1,
                        gt_heads=1, gt_attention_scope="diagonal"),
            hierarchy=hierarchy,
        )
    except ValueError as e:
        print(f"  bad attention scope correctly rejected: {e}")
    else:
        raise AssertionError("an unknown gt_attention_scope should raise")

    print("\nmetrics sanity check (finest level only):")
    n_finest = len(hierarchy.level_classes[-1])
    y_true = targets[:, -1].cpu().numpy()
    y_pred = preds[:, -1].detach().cpu().numpy()
    print(summarize(y_true, y_pred, n_finest, hierarchy.level_classes[-1]))

    print("\nall smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
