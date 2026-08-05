"""Trains + evaluates the mean-pool baseline (baseline/mean_pool_classifier.py)
on the same cells/splits/labels the GNN uses. Run via sbatch (mit_normal_gpu,
per project policy that all training/eval/inference runs on GPU nodes, even
though this particular model is small enough that CPU would also be fine).

    python scripts/train_baseline.py --depth 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from baseline.mean_pool_classifier import WINDOW_NM, MeanPoolClassifier, build_feature_matrix  # noqa: E402
from data.dataset import load_manifest  # noqa: E402
from gnn.losses import classification_loss, compute_class_weights  # noqa: E402
from gnn.metrics import summarize  # noqa: E402


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    manifest = load_manifest()
    Xtr, ytr, _, classes = build_feature_matrix(manifest, "train", args.depth, args.window_nm)
    Xval, yval, _, _ = build_feature_matrix(manifest, "val", args.depth, args.window_nm)
    Xtest, ytest, _, _ = build_feature_matrix(manifest, "test", args.depth, args.window_nm)
    print(f"train={len(ytr)} val={len(yval)} test={len(ytest)} classes={classes}")

    # Standardize using train statistics only -- val/test never touch this.
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sigma, dtype=torch.float32, device=device)
    Xval_t = torch.tensor((Xval - mu) / sigma, dtype=torch.float32, device=device)
    Xtest_t = torch.tensor((Xtest - mu) / sigma, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)

    class_weights = compute_class_weights(ytr_t, len(classes)).to(device)
    model = MeanPoolClassifier(Xtr.shape[1], len(classes), hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_bacc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = classification_loss(logits, ytr_t, class_weights)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t).argmax(1).cpu().numpy()
        val_metrics = summarize(yval, val_pred, len(classes), classes)
        if val_metrics["balanced_accuracy"] > best_val_bacc:
            best_val_bacc = val_metrics["balanced_accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 10) == 0:
            print(
                f"epoch {epoch:4d}  loss={loss.item():.4f}  "
                f"val_bacc={val_metrics['balanced_accuracy']:.3f}  val_acc={val_metrics['accuracy']:.3f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xtest_t).argmax(1).cpu().numpy()
    test_metrics = summarize(ytest, test_pred, len(classes), classes)
    print("=== test metrics (baseline: geodesic-mean pooling) ===")
    print(json.dumps(test_metrics, indent=2))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"baseline_depth{args.depth}.json"
    out_path.write_text(
        json.dumps({"args": vars(args), "test_metrics": test_metrics, "classes": classes}, indent=2)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--depth", type=int, default=2,
        help="label hierarchy depth (2=neuron/glia, 3=+E/I/glia-subtype, 0=finest)",
    )
    p.add_argument("--window-nm", type=float, default=WINDOW_NM)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()
    if args.depth == 0:
        args.depth = None
    train(args)
