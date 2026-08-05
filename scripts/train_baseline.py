"""Trains + evaluates the mean-pool baseline (baseline/mean_pool_classifier.py)
on the same cells/splits/labels the GNN uses. Classification happens per
25um-windowed point, with cell-level accuracy from majority-voting per-point
predictions -- matching the SegCLR paper's classifier and this lab's own
replication (see baseline/mean_pool_classifier.py's docstring for the two
independent sources). Run via sbatch (mit_normal_gpu, per project policy that
all training/eval/inference runs on GPU nodes).

    python scripts/train_baseline.py --depth 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from baseline.mean_pool_classifier import WINDOW_NM, MeanPoolClassifier, build_node_feature_matrix  # noqa: E402
from data.dataset import load_manifest  # noqa: E402
from gnn.losses import classification_loss, compute_class_weights  # noqa: E402
from gnn.metrics import majority_vote_by_group, summarize  # noqa: E402


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    manifest = load_manifest()
    Xtr, ytr, rtr, classes = build_node_feature_matrix(manifest, "train", args.depth, args.window_nm)
    Xval, yval, rval, _ = build_node_feature_matrix(manifest, "val", args.depth, args.window_nm)
    Xtest, ytest, rtest, _ = build_node_feature_matrix(manifest, "test", args.depth, args.window_nm)
    print(
        f"train: {len(ytr)} points / {len(set(rtr.tolist()))} cells   "
        f"val: {len(yval)} points / {len(set(rval.tolist()))} cells   "
        f"test: {len(ytest)} points / {len(set(rtest.tolist()))} cells   classes={classes}"
    )

    # Standardize using train POINT statistics only -- val/test never touch this.
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sigma, dtype=torch.float32, device=device)
    Xval_t = torch.tensor((Xval - mu) / sigma, dtype=torch.float32, device=device)
    Xtest_t = torch.tensor((Xtest - mu) / sigma, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)

    # Class weights from training POINTS, same "training data only" rule as
    # everywhere else -- a point is still training data, just more granular.
    class_weights = compute_class_weights(ytr_t, len(classes)).to(device)
    model = MeanPoolClassifier(Xtr.shape[1], len(classes), hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_bacc, best_state = -1.0, None
    pbar = tqdm(range(args.epochs), desc="train", unit="epoch")
    for epoch in pbar:
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = classification_loss(logits, ytr_t, class_weights)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred_points = model(Xval_t).argmax(1).cpu().numpy()
        val_cell_true, val_cell_pred = majority_vote_by_group(rval, yval, val_pred_points)
        val_metrics = summarize(val_cell_true, val_cell_pred, len(classes), classes)
        if val_metrics["balanced_accuracy"] > best_val_bacc:
            best_val_bacc = val_metrics["balanced_accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            val_bacc_cell=f"{val_metrics['balanced_accuracy']:.3f}",
            val_acc_cell=f"{val_metrics['accuracy']:.3f}",
        )
        if epoch % max(1, args.epochs // 10) == 0:
            tqdm.write(
                f"epoch {epoch:4d}  loss={loss.item():.4f}  "
                f"val_bacc_cell={val_metrics['balanced_accuracy']:.3f}  val_acc_cell={val_metrics['accuracy']:.3f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_points = model(Xtest_t).argmax(1).cpu().numpy()

    # Report both granularities: per-point (what the classifier actually
    # sees/decides on each 25um window) and cell-level via majority vote
    # (the number that's actually comparable to "does the GNN beat this").
    point_metrics = summarize(ytest, test_pred_points, len(classes), classes)
    cell_true, cell_pred = majority_vote_by_group(rtest, ytest, test_pred_points)
    cell_metrics = summarize(cell_true, cell_pred, len(classes), classes)

    print("=== test metrics, per-point (each 25um window classified independently) ===")
    print(json.dumps(point_metrics, indent=2))
    print("\n=== test metrics, cell-level (majority vote over a cell's point predictions) ===")
    print(json.dumps(cell_metrics, indent=2))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"baseline_depth{args.depth}.json"
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "point_test_metrics": point_metrics,
                "cell_test_metrics": cell_metrics,
                "classes": classes,
            },
            indent=2,
        )
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
