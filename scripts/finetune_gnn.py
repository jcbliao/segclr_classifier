"""Classification training on top of GraphAutoEncoderClassifier (spec steps
5-6), in four modes:

  --mode scratch   classification only, no pretraining, no masking -- the
                    "does pretraining even help" ablation baseline for the GNN.
  --mode frozen     encoder loaded from --pretrained-ckpt and frozen; only the
                    readout + cls_head train.
  --mode finetune   encoder loaded from --pretrained-ckpt; all weights train.
  --mode joint      L_joint = L_cls + lambda_rec * L_mask, masking active
                    throughout (optionally warm-started from
                    --pretrained-ckpt).

Run via sbatch (mit_normal_gpu):
    python scripts/finetune_gnn.py --mode finetune --pretrained-ckpt results/pretrain_random/checkpoint_e99.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.dataset import ReplacementPool, SegCLRGraphDataset, load_manifest  # noqa: E402
from gnn.losses import (  # noqa: E402
    classification_loss,
    compute_class_weights,
    joint_loss,
    masked_reconstruction_loss,
)
from gnn.metrics import summarize  # noqa: E402
from gnn.model import GraphAutoEncoderClassifier, ModelConfig  # noqa: E402


def load_pretrained(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.edge_attr, mode="classify")
        preds.append(out["logits"].argmax(1).cpu().numpy())
        labels.append(data.y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest()
    train_ds = SegCLRGraphDataset(manifest, "train", depth=args.depth)
    val_ds = SegCLRGraphDataset(manifest, "val", depth=args.depth)
    test_ds = SegCLRGraphDataset(manifest, "test", depth=args.depth)
    classes = train_ds.classes
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    config = ModelConfig(
        in_dim=train_ds[0].x.shape[1],
        hidden_dim=args.hidden_dim,
        encoder_out_dim=args.hidden_dim,
        num_encoder_layers=args.num_layers,
        conv_type=args.conv_type,
        num_classes=len(classes),
        mask_prob=args.mask_prob,
    )
    model = GraphAutoEncoderClassifier(config).to(device)

    if args.pretrained_ckpt:
        load_pretrained(model, args.pretrained_ckpt, device)
        print(f"loaded pretrained weights from {args.pretrained_ckpt}")

    if args.mode == "frozen":
        model.encoder.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    train_labels = torch.tensor([int(train_ds[i].y.item()) for i in range(len(train_ds))])
    class_weights = compute_class_weights(train_labels, len(classes)).to(device)

    pool = (
        ReplacementPool(train_ds, strategy=args.replace_strategy, seed=args.seed, device=device)
        if args.mode == "joint"
        else None
    )

    best_val_bacc, best_state = -1.0, None
    epoch_bar = tqdm(range(args.epochs), desc="finetune", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss, n = 0.0, 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch}", unit="batch", leave=False)
        for data in batch_bar:
            data = data.to(device)
            opt.zero_grad()
            if args.mode == "joint":
                label = int(data.y.item())
                out = model(
                    data.x, data.edge_index, data.batch, data.edge_attr,
                    mode="joint", replacement_source=pool.sampler_for(label),
                )
                cls = classification_loss(out["logits"], data.y, class_weights)
                rec = (
                    masked_reconstruction_loss(out["x_hat"], out["target"])
                    if out["mask"].sum()
                    else torch.zeros((), device=device)
                )
                loss = joint_loss(cls, rec, lambda_rec=args.lambda_rec)
            else:
                out = model(data.x, data.edge_index, data.batch, data.edge_attr, mode="classify")
                loss = classification_loss(out["logits"], data.y, class_weights)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.num_graphs
            n += data.num_graphs
            batch_bar.set_postfix(loss=f"{total_loss / max(1, n):.4f}")

        val_labels, val_preds = evaluate(model, val_loader, device)
        val_metrics = summarize(val_labels, val_preds, len(classes), classes)
        if val_metrics["balanced_accuracy"] > best_val_bacc:
            best_val_bacc = val_metrics["balanced_accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epoch_bar.set_postfix(
            train_loss=f"{total_loss / max(1, n):.4f}",
            val_bacc=f"{val_metrics['balanced_accuracy']:.3f}",
            val_acc=f"{val_metrics['accuracy']:.3f}",
        )
        if epoch % max(1, args.epochs // 20) == 0:
            tqdm.write(
                f"epoch {epoch:4d}  train_loss={total_loss / max(1, n):.4f}  "
                f"val_bacc={val_metrics['balanced_accuracy']:.3f}  val_acc={val_metrics['accuracy']:.3f}"
            )

    model.load_state_dict(best_state)
    test_labels, test_preds = evaluate(model, test_loader, device)
    test_metrics = summarize(test_labels, test_preds, len(classes), classes)
    print("=== test metrics (GNN) ===")
    print(json.dumps(test_metrics, indent=2))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    # Tag by source pretrain run (e.g. "pretrain_random_mask0.1") so frozen/
    # finetune results from different masking ratios don't collide on the
    # same two filenames -- mode+depth alone doesn't distinguish them.
    if args.pretrained_ckpt:
        ckpt_tag = Path(args.pretrained_ckpt).parent.name
        out_path = out_dir / f"gnn_{args.mode}_{ckpt_tag}_depth{args.depth}.json"
    else:
        out_path = out_dir / f"gnn_{args.mode}_depth{args.depth}.json"
    out_path.write_text(
        json.dumps({"args": vars(args), "test_metrics": test_metrics, "classes": classes}, indent=2)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="scratch", choices=["scratch", "frozen", "finetune", "joint"])
    p.add_argument("--pretrained-ckpt", default=None)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--conv-type", default="sage", choices=["sage", "gat", "transformer"])
    p.add_argument("--mask-prob", type=float, default=0.3)
    p.add_argument(
        "--replace-strategy", default="random", choices=["random", "diff_class", "same_class"]
    )
    p.add_argument("--lambda-rec", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.mode in ("frozen", "finetune") and not args.pretrained_ckpt:
        raise SystemExit(f"--mode {args.mode} requires --pretrained-ckpt")
    if args.mode == "joint" and args.batch_size != 1:
        print("mode=joint requires batch_size=1 (per-graph masking); overriding")
        args.batch_size = 1
    if args.depth == 0:
        args.depth = None
    main(args)
