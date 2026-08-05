"""Import + forward-pass smoke test for gnn/* on synthetic data -- catches
shape/API bugs before spending a full GPU allocation on the real pipeline. No
real data needed. Run via sbatch (mit_normal_gpu -- project policy: all
training/eval/inference runs on GPU nodes, so this also confirms the model
actually runs correctly on CUDA, not just CPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

from gnn.losses import (  # noqa: E402
    classification_loss,
    compute_class_weights,
    cosine_reconstruction_loss,
    joint_loss,
    masked_reconstruction_loss,
)
from gnn.metrics import summarize  # noqa: E402
from gnn.model import GraphAutoEncoderClassifier, ModelConfig  # noqa: E402


def random_graph(n_nodes: int, d: int, seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, d, generator=g)
    # a simple chain, like a skeleton
    src = torch.arange(n_nodes - 1)
    dst = torch.arange(1, n_nodes)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    edge_attr = torch.rand(edge_index.shape[1], 1, generator=g) * 500
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device visible -- this job was expected to run on mit_normal_gpu")

    torch.manual_seed(0)
    d, num_classes = 64, 3
    graphs = [random_graph(n, d, seed=i) for i, n in enumerate([20, 35, 12])]
    for i, g in enumerate(graphs):
        g.y = torch.tensor([i % num_classes])
    batch = Batch.from_data_list(graphs).to(device)
    print(f"batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes total")

    for conv_type in ("sage", "gat", "transformer"):
        print(f"\n--- conv_type={conv_type} ---")
        config = ModelConfig(
            in_dim=d, hidden_dim=32, encoder_out_dim=32, num_encoder_layers=2,
            conv_type=conv_type, encoder_heads=2, num_classes=num_classes, mask_prob=0.3,
        )
        model = GraphAutoEncoderClassifier(config).to(device)

        def replacement_source(n):
            return torch.randn(n, d, device=device)

        out = model(
            batch.x, batch.edge_index, batch.batch, batch.edge_attr,
            mode="pretrain", replacement_source=replacement_source,
        )
        assert out["x_hat"].shape == out["target"].shape, (out["x_hat"].shape, out["target"].shape)
        rec_loss = masked_reconstruction_loss(out["x_hat"], out["target"])
        print(f"  pretrain: mask sum={int(out['mask'].sum())}  rec_loss={rec_loss.item():.4f}")
        rec_loss.backward()

        model.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, mode="classify")
        assert out["logits"].shape == (batch.num_graphs, num_classes), out["logits"].shape
        weights = compute_class_weights(batch.y, num_classes)
        cls_loss = classification_loss(out["logits"], batch.y, weights)
        print(f"  classify: logits shape={tuple(out['logits'].shape)}  cls_loss={cls_loss.item():.4f}")
        cls_loss.backward()

        model.zero_grad()
        out = model(
            batch.x, batch.edge_index, batch.batch, batch.edge_attr,
            mode="joint", replacement_source=replacement_source,
        )
        j_loss = joint_loss(
            classification_loss(out["logits"], batch.y, weights),
            masked_reconstruction_loss(out["x_hat"], out["target"]),
            lambda_rec=1.0,
        )
        print(f"  joint: loss={j_loss.item():.4f}")
        j_loss.backward()

    print("\nmetrics sanity check:")
    y_true = batch.y.cpu().numpy()
    y_pred = out["logits"].argmax(1).detach().cpu().numpy()
    print(summarize(y_true, y_pred, num_classes, [f"class{i}" for i in range(num_classes)]))

    print("\nall smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
