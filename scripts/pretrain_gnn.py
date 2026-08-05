"""Masked-autoencoder pretraining of the GNN encoder (spec steps 1-4).

Loss is cosine reconstruction over the masked set M, starting point per
explicit project direction regardless of what scripts/norm_diagnostic.py
finds (--use-smooth-l1 is available but off by default).

Run via sbatch (mit_normal_gpu):
    python scripts/pretrain_gnn.py --replace-strategy random
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from data.dataset import ReplacementPool, SegCLRGraphDataset, load_manifest  # noqa: E402
from gnn.losses import masked_reconstruction_loss  # noqa: E402
from gnn.model import GraphAutoEncoderClassifier, ModelConfig  # noqa: E402


def run_epoch(model, loader, pool, device, opt=None, accum_steps=1, use_smooth_l1=False, lambda_mag=1.0):
    """opt=None -> eval mode, no gradient step."""
    train_mode = opt is not None
    model.train(train_mode)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        if train_mode:
            opt.zero_grad()
        for step, data in enumerate(loader):
            data = data.to(device)
            label = int(data.y.item())
            replacement_source = pool.sampler_for(label)
            out = model(
                data.x, data.edge_index, data.batch, data.edge_attr,
                mode="pretrain", replacement_source=replacement_source,
            )
            if out["mask"].sum() == 0:
                continue
            loss = masked_reconstruction_loss(
                out["x_hat"], out["target"], use_smooth_l1=use_smooth_l1, lambda_mag=lambda_mag
            )
            if train_mode:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    opt.step()
                    opt.zero_grad()
            total_loss += loss.item()
            n += 1
        if train_mode and (step + 1) % accum_steps != 0:
            opt.step()
            opt.zero_grad()
    return total_loss / max(1, n)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest()
    train_ds = SegCLRGraphDataset(manifest, "train", depth=args.depth)
    val_ds = SegCLRGraphDataset(manifest, "val", depth=args.depth)
    print(f"train={len(train_ds)} val={len(val_ds)} classes={train_ds.classes}")

    # ReplacementPool.sampler_for(label) is bound to ONE graph's class, so
    # every draw within a masked graph uses that graph's own class -- correct
    # only if each forward pass is one graph. batch_size=1 + gradient
    # accumulation is therefore not just a memory concession (cells run up to
    # ~20k nodes, per CLAUDE.md's p99) but required for same_class/diff_class
    # replacement to mean what the spec says.
    pool = ReplacementPool(train_ds, strategy=args.replace_strategy, seed=args.seed, device=device)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    config = ModelConfig(
        in_dim=train_ds[0].x.shape[1],
        hidden_dim=args.hidden_dim,
        encoder_out_dim=args.hidden_dim,
        num_encoder_layers=args.num_layers,
        conv_type=args.conv_type,
        num_classes=len(train_ds.classes),
        mask_prob=args.mask_prob,
    )
    model = GraphAutoEncoderClassifier(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(__file__).resolve().parent.parent / "results" / f"pretrain_{args.replace_strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, pool, device, opt, args.accum_steps, args.use_smooth_l1, args.lambda_mag
        )
        val_loss = run_epoch(
            model, val_loader, pool, device, None, use_smooth_l1=args.use_smooth_l1, lambda_mag=args.lambda_mag
        )
        print(f"epoch {epoch:4d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt_path = out_dir / f"checkpoint_e{epoch}.pt"
            torch.save(
                {"model_state": model.state_dict(), "config": config, "epoch": epoch}, ckpt_path
            )
            print(f"  saved {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--conv-type", default="sage", choices=["sage", "gat", "transformer"])
    p.add_argument("--mask-prob", type=float, default=0.3)
    p.add_argument(
        "--replace-strategy", default="random", choices=["random", "diff_class", "same_class"]
    )
    p.add_argument(
        "--use-smooth-l1", action="store_true",
        help="off by default -- cosine-only is the starting default per project direction",
    )
    p.add_argument("--lambda-mag", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.depth == 0:
        args.depth = None
    main(args)
