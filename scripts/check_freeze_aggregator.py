"""Verifies --freeze-aggregator does what it claims, on synthetic windows.

Freezing is the kind of change that fails silently in both directions: a
missed `requires_grad_(False)` trains the "frozen" stage anyway, and a
forgotten eval() leaves its dropout resampling the supposedly fixed features
every forward pass. Either one turns the random-features control into
something else without erroring, and the run's metrics would look perfectly
reasonable.

Four assertions per architecture, after a real forward + backward + optimizer
step:

  1. every aggregator parameter has requires_grad False, and none of them
     received a gradient;
  2. their VALUES are bit-identical before and after the step (the check that
     actually matters -- requires_grad False is what should cause this, but
     weight decay or a stray optimizer group could still move them);
  3. head parameters did move, so the run is training something;
  4. two forward passes in train() mode give identical output, i.e. the
     frozen stage is deterministic (MPNNEncoder's dropout defaults to 0.1).

Runs on GPU per CLAUDE.md's rule for model code; takes seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

IN_DIM, POS_DIM, N_WINDOWS = 64, 8, 32


def synthetic_batch(hierarchy, device):
    """Small random windows, shaped like data/geodesic_window.py's output.

    Targets are derived from a real granular label's full path, exactly as
    data/dataset_windowed.py does. Drawing each level independently would
    produce paths the tree does not contain, and LCPN's global-to-local remap
    returns -1 for a child that is not under the sampled parent -- which
    surfaces as an opaque device-side assert inside cross_entropy rather than
    as anything resembling "your labels are inconsistent".
    """
    g = torch.Generator().manual_seed(0)
    labels = sorted(hierarchy.label_paths)
    items = []
    for _ in range(N_WINDOWS):
        n = int(torch.randint(2, 12, (1,), generator=g))
        # A path graph: connected, so the Laplacian PE and adjacency are real.
        src = torch.arange(n - 1)
        edge_index = torch.cat(
            [torch.stack([src, src + 1]), torch.stack([src + 1, src])], dim=1
        )
        path = hierarchy.label_paths[labels[int(torch.randint(0, len(labels), (1,), generator=g))]]
        y = torch.tensor(
            [hierarchy.level_maps[lvl][path[lvl]] for lvl in range(hierarchy.depth)],
            dtype=torch.long,
        ).unsqueeze(0)
        items.append(Data(
            x=torch.randn(n, IN_DIM, generator=g),
            edge_index=edge_index,
            pos_enc=torch.randn(n, POS_DIM, generator=g),
            rel_pos=torch.randn(n, 3, generator=g),
            y_levels=y,
        ))
    return next(iter(DataLoader(items, batch_size=N_WINDOWS))).to(device)


def check(architecture: str, hierarchy, device) -> None:
    print(f"\n=== architecture={architecture} ===")
    config = ModelConfig(
        in_dim=IN_DIM, architecture=architecture, gt_pos_dim=POS_DIM,
        use_spatial_features=architecture == "mpnn",
    )
    torch.manual_seed(0)
    model = WindowClassifier(config, hierarchy=hierarchy).to(device)
    n_frozen = model.freeze_aggregator()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  frozen {n_frozen:,} parameters, {n_train:,} trainable")
    assert n_frozen > 0 and n_train > 0

    agg_modules = [m for m in (model.graph_transformer, model.encoder) if m is not None]
    agg_params = {n: p for m in agg_modules for n, p in m.named_parameters()}
    head_params = dict(model.cls_head.named_parameters())
    before = {n: p.detach().clone() for n, p in agg_params.items()}
    head_before = {n: p.detach().clone() for n, p in head_params.items()}

    assert not any(p.requires_grad for p in agg_params.values()), "aggregator param wants grad"

    batch = synthetic_batch(hierarchy, device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=1e-2,
    )
    model.train()

    # (4) the eval-mode pin, checked two ways.
    #
    # Directly first: every stochastic submodule of the frozen stage must
    # report training=False even though model.train() was just called. This is
    # the actual claim, and it is exact.
    live = [
        f"{type(m).__name__}(p={getattr(m, 'p', '?')})"
        for mod in agg_modules for m in mod.modules()
        if isinstance(m, torch.nn.Dropout) and m.training
    ]
    assert not live, f"frozen stage still has dropout in train mode: {live}"
    n_dropout = sum(
        isinstance(m, torch.nn.Dropout) for mod in agg_modules for m in mod.modules()
    )
    print(f"  eval-mode pin holds: {n_dropout} Dropout module(s) inactive under model.train()")

    # Then behaviourally, with a tolerance rather than torch.equal. Live
    # dropout at p=0.1 would move outputs by far more than this; what a strict
    # equality check would ALSO catch is SAGEConv's scatter reduction, whose
    # CUDA accumulation order is nondeterministic and differs in the last bits
    # between identical calls. That is a property of the conv, not a leak in
    # the freeze, so it must not fail this test.
    with torch.no_grad():
        a = model(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
        b = model(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
    max_diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), (
        f"frozen aggregator output varies between identical calls by {max_diff:.3e} -- "
        "too large for float accumulation order, so something stochastic is still live"
    )
    print(f"  two train()-mode forwards agree to {max_diff:.2e} (float accumulation order only)")

    for _ in range(3):
        opt.zero_grad()
        g = model(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
        loss = model.cls_head.compute_loss(g, batch.y_levels)
        loss.backward()
        opt.step()
    print(f"  3 optimizer steps taken, final loss {loss.item():.4f}")

    got_grad = [n for n, p in agg_params.items() if p.grad is not None]
    assert not got_grad, f"frozen parameters received gradients: {got_grad[:3]}"
    moved = [n for n, p in agg_params.items() if not torch.equal(p.detach(), before[n])]
    assert not moved, f"frozen parameters CHANGED VALUE: {moved[:3]}"
    print(f"  all {len(agg_params)} aggregator tensors bit-identical after the steps")

    head_moved = [n for n, p in head_params.items() if not torch.equal(p.detach(), head_before[n])]
    assert head_moved, "no head parameter moved -- nothing is training"
    print(f"  {len(head_moved)}/{len(head_params)} head tensors updated")

    # The mean architecture must refuse rather than silently no-op.
    if architecture == "mpnn":
        mean_model = WindowClassifier(
            ModelConfig(in_dim=IN_DIM, architecture="mean", gt_pos_dim=POS_DIM), hierarchy
        ).to(device)
        try:
            mean_model.freeze_aggregator()
        except ValueError as e:
            print(f"  architecture='mean' correctly refuses: {e}")
        else:
            raise AssertionError("architecture='mean' should refuse to freeze")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    hierarchy = load_hierarchy(load_manifest())
    print(f"hierarchy depth {hierarchy.depth}, {len(hierarchy.level_classes[-1])} classes")
    for architecture in ("graph_transformer", "mpnn"):
        check(architecture, hierarchy, device)
    print("\nall freeze checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
