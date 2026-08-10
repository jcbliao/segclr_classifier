"""Import + forward-pass smoke test for gnn/* on synthetic data -- catches
shape/API bugs before spending a full GPU allocation on the real pipeline. No
real data needed. Run via sbatch (mit_normal_gpu -- project policy: all
training/eval/inference runs on GPU nodes, so this also confirms the model
actually runs correctly on CUDA, not just CPU).

Covers all three aggregation methods (--architecture graph_transformer / mpnn
/ mean, see gnn/model.py) against the real LAB_HIERARCHY_TREE
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

from data.geodesic_window import REL_POS_SCALE_NM, _window_laplacian_pos_enc  # noqa: E402
from gnn.hierarchy import LAB_HIERARCHY_TREE, parse_hierarchy  # noqa: E402
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
    # Edge length in nm, spanning the buckets DIST_BIAS_BOUNDARIES_NM defines
    # (real p5-p95 is ~530-3900nm) so the distance-bias lookup is exercised
    # across several buckets rather than landing in one.
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

    hierarchy = parse_hierarchy(LAB_HIERARCHY_TREE)
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
    for architecture, expected_dim in (("graph_transformer", 32), ("mpnn", 48), ("mean", d)):
        print(f"\n--- architecture={architecture} ---")
        config = ModelConfig(
            in_dim=d, architecture=architecture,
            mpnn_hidden_dim=48, mpnn_out_dim=48, mpnn_layers=2,
            gt_dim=32, gt_depth=2, gt_heads=2, gt_pos_dim=GT_POS_DIM,
        )
        model = WindowClassifier(config, hierarchy=hierarchy).to(device)
        if architecture == "graph_transformer":
            assert model.readout is None, "graph_transformer should not build a separate readout"
            assert model.encoder is None, "graph_transformer should not build an MPNN encoder"
        else:
            assert model.graph_transformer is None, f"{architecture} should not build a GraphTransformer"
            # Zero-parameter readout in both cases: for "mean" that means every
            # trainable weight belongs to the classification head, which is what
            # makes it a clean baseline; for "mpnn" all the aggregation
            # parameters live in the encoder, not the pooling step.
            assert not list(model.readout.parameters()), "MeanReadout should have no parameters"
            if architecture == "mpnn":
                assert model.encoder is not None, "mpnn should build an MPNN encoder"
            else:
                assert model.encoder is None, "mean should not build an encoder"

        g = model(
            batch.x, batch.edge_index, batch.batch,
            pos_enc=batch.pos_enc, rel_pos=batch.rel_pos,
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
        aggregator = model.graph_transformer if architecture == "graph_transformer" else model.encoder
        if aggregator is not None:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in aggregator.parameters()
            ), f"{architecture}'s aggregator got no gradient from the classification loss"

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
        "dist bias": {"gt_use_dist_bias": True},
        "dist bias + neighborhood": {
            "gt_use_dist_bias": True, "gt_attention_scope": "neighborhood",
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
            assert (blk.attn.dist_bias is not None) == cfg.gt_use_dist_bias
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
              pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, thickness=batch.thickness,
              edge_attr=batch.edge_attr)
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

    # --- classification head: linear probe vs. the lab's ResNet trunk --------
    # Orthogonal to `architecture`, so check it composes with all three rather
    # than only with the GraphTransformer it was added alongside.
    print("\n--- classification head: --cls-resnet across architectures ---")
    for architecture, expect_in in (("graph_transformer", 32), ("mpnn", 48), ("mean", d)):
        cfg = ModelConfig(
            in_dim=d, architecture=architecture,
            mpnn_hidden_dim=48, mpnn_out_dim=48, mpnn_layers=2,
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
              rel_pos=batch.rel_pos, edge_attr=batch.edge_attr)
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

    # Linear probe stays the default -- a run that does not ask for the trunk
    # must not get one.
    assert WindowClassifier(
        ModelConfig(in_dim=d, architecture="mean"), hierarchy=hierarchy
    ).cls_head.trunk is None, "linear probe should be the default head"
    print("  linear probe is still the default")

    # The distance bias claims to be initialized as an EXACT reproduction of
    # binary adjacency. That is the whole reason it is safe to switch on, so
    # verify it rather than trusting the init code: same weights everywhere
    # else, the two models must agree bit-for-bit at step 0.
    base = WindowClassifier(
        ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=2, gt_heads=2),
        hierarchy=hierarchy,
    ).to(device)
    withbias = WindowClassifier(
        ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=2,
                    gt_heads=2, gt_use_dist_bias=True),
        hierarchy=hierarchy,
    ).to(device)
    # Copy every shared weight across rather than reusing a seed. Building the
    # dist_bias embedding consumes RNG, so two same-seeded constructions
    # diverge in ALL their other parameters -- which makes a seeded comparison
    # test nothing. strict=False leaves dist_bias.weight at its ones/zeros init
    # (base's state_dict has no such key).
    withbias.load_state_dict(base.state_dict(), strict=False)
    kw = dict(pos_enc=batch.pos_enc, rel_pos=batch.rel_pos, edge_attr=batch.edge_attr)
    with torch.no_grad():
        g_base = base(batch.x, batch.edge_index, batch.batch, **kw)
        g_bias = withbias(batch.x, batch.edge_index, batch.batch, **kw)
    assert torch.allclose(g_base, g_bias, atol=1e-5), (
        "dist bias is not a no-op at init -- max abs diff "
        f"{(g_base - g_bias).abs().max().item():.2e}"
    )
    print(f"\n  dist bias reproduces binary adjacency at init "
          f"(max diff {(g_base - g_bias).abs().max().item():.1e})")

    # And it must fail loudly when combined with the switch that removes the
    # bias term it lives in, rather than silently doing nothing.
    try:
        WindowClassifier(
            ModelConfig(in_dim=d, architecture="graph_transformer", gt_dim=32, gt_depth=1,
                        gt_heads=1, gt_use_dist_bias=True, gt_use_adj_bias=False),
            hierarchy=hierarchy,
        )
    except ValueError as e:
        print(f"  dist bias without adj bias correctly rejected: {e}")
    else:
        raise AssertionError("gt_use_dist_bias without gt_use_adj_bias should raise")

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
