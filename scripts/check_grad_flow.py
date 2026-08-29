"""Measures how much gradient actually reaches the aggregator, with a linear
probe head versus the shared ResNet trunk.

The concern this answers: inserting gnn/resnet.py::DeepResNetTrunk between the
readout and the per-node LCPN heads puts four residual blocks between the loss
and the aggregation stage, and it is not obvious by inspection whether the
aggregator still receives a usable training signal through them.

Two distinct questions, and only the second is interesting:

  1. Does gradient reach the aggregator AT ALL under --cls-resnet? This is a
     yes/no about the autograd graph, and a non-finite or exactly-zero norm
     would mean a real bug (a detach, an in-place break, a dead branch).
  2. Is it ATTENUATED relative to the probe? A deep randomly-initialised head
     can shrink or scramble the signal arriving upstream, which shows up as
     the aggregator learning more slowly even though nothing is broken. The
     ratio of aggregator-to-head gradient norm is the readable version of
     that, since it says where the optimizer's effort is actually going.

Reports both, for both parameterised architectures, at initialisation and
again after a few steps -- attenuation at step 0 is a property of the random
init, whereas attenuation that persists is a property of training.
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

IN_DIM, POS_DIM, N_WINDOWS = 64, 8, 256


def synthetic_batch(hierarchy, device):
    g = torch.Generator().manual_seed(0)
    labels = sorted(hierarchy.label_paths)
    items = []
    for _ in range(N_WINDOWS):
        n = int(torch.randint(2, 12, (1,), generator=g))
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
            x=torch.randn(n, IN_DIM, generator=g), edge_index=edge_index,
            pos_enc=torch.randn(n, POS_DIM, generator=g),
            rel_pos=torch.randn(n, 3, generator=g), y_levels=y,
        ))
    return next(iter(DataLoader(items, batch_size=N_WINDOWS))).to(device)


def grad_norm(params) -> float:
    """L2 norm over a parameter group's gradients, treating absent grads as 0."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total**0.5


def step_and_measure(model, batch, opt=None) -> dict:
    model.zero_grad(set_to_none=True)
    g = model(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
    loss = model.cls_head.compute_loss(g, batch.y_levels)
    loss.backward()

    agg = [p for m in (model.graph_transformer, model.encoder) if m is not None
           for p in m.parameters()]
    trunk = list(model.cls_head.trunk.parameters()) if model.cls_head.trunk is not None else []
    heads = list(model.cls_head.heads.parameters())

    out = {
        "loss": loss.item(),
        "agg": grad_norm(agg),
        "trunk": grad_norm(trunk),
        "heads": grad_norm(heads),
        # The first aggregator layer is the furthest point from the loss, so
        # it is where attenuation through the head would show up first.
        "agg_first": grad_norm(agg[:2]),
    }
    if opt is not None:
        opt.step()
    return out


def run(architecture: str, use_resnet: bool, hierarchy, device, n_steps: int) -> None:
    config = ModelConfig(
        in_dim=IN_DIM, architecture=architecture, gt_pos_dim=POS_DIM,
        use_spatial_features=architecture == "mpnn", cls_head_resnet=use_resnet,
    )
    torch.manual_seed(0)  # same init for both head choices, so norms are comparable
    model = WindowClassifier(config, hierarchy=hierarchy).to(device)
    model.train()
    batch = synthetic_batch(hierarchy, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    head = "resnet4x128" if use_resnet else "probe"
    first = step_and_measure(model, batch, opt)
    for _ in range(n_steps - 1):
        last = step_and_measure(model, batch, opt)

    for tag, m in (("step0", first), (f"step{n_steps - 1}", last)):
        finite = "OK" if m["agg"] > 0 and torch.isfinite(torch.tensor(m["agg"])) else "DEAD"
        print(
            f"  {head:<12} {tag:<8} loss={m['loss']:.4f}  |grad| agg={m['agg']:.3e} "
            f"trunk={m['trunk']:.3e} heads={m['heads']:.3e}  "
            f"agg/heads={m['agg'] / max(m['heads'], 1e-12):.3f}  "
            f"agg_layer0={m['agg_first']:.3e}  [{finite}]"
        )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchy = load_hierarchy(load_manifest())
    print(f"device: {device}, hierarchy depth {hierarchy.depth}, "
          f"{len(hierarchy.level_classes[-1])} classes\n")
    for architecture in ("graph_transformer", "mpnn"):
        print(f"=== {architecture} ===")
        for use_resnet in (False, True):
            run(architecture, use_resnet, hierarchy, device, n_steps=20)
    print(
        "\nagg=0 or non-finite would mean the trunk breaks the autograd path. "
        "A much smaller agg/heads ratio under resnet4x128 than under probe means the "
        "signal still arrives but the head is absorbing the error."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
