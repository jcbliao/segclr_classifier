"""Embedding-norm diagnostic: how variable are SegCLR embedding norms, do
they predict anything, and can they be ablated. Informative only -- nothing
in the training pipeline branches on its output.

Uses the local dataset from data/build_dataset.py (data/manifest.json +
data/graph_cache/*.pt) -- run build_dataset.py first. Run via sbatch
(mit_quicktest is plenty; this is pure numpy/scipy over already-cached data).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from data.dataset import label_vocab, load_manifest  # noqa: E402

GRAPH_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "graph_cache"


def load_cell_arrays(manifest: dict, depth: int = 2):
    """Returns (node_norms, node_class_ids, cell_mean_x, cell_class_ids, classes)."""
    import torch

    classes, class_to_idx = label_vocab(manifest, depth=depth)
    node_norms, node_class_ids = [], []
    cell_means, cell_class_ids = [], []

    for root_id, info in manifest["cells"].items():
        path = GRAPH_CACHE_DIR / f"{root_id}.pt"
        if not path.exists():
            continue
        data = torch.load(path, weights_only=False)
        x = data.x.numpy()
        cid = class_to_idx["-".join(info["cell_type"].split("-")[:depth])]
        norms = np.linalg.norm(x, axis=1)
        node_norms.append(norms)
        node_class_ids.append(np.full(len(norms), cid))
        cell_means.append(x.mean(axis=0))
        cell_class_ids.append(cid)

    return (
        np.concatenate(node_norms),
        np.concatenate(node_class_ids),
        np.stack(cell_means),
        np.array(cell_class_ids),
        classes,
    )


def nearest_centroid_accuracy(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Simple, dependency-light probe: 70/30 split, class centroids from the
    70%, cosine-nearest-centroid accuracy on the 30%. Used twice -- once on
    raw X, once on L2-normalized X -- to answer "can the norm be ablated
    without hurting classification".
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    classes = np.unique(y)
    centroids = np.stack([X[train_idx][y[train_idx] == c].mean(axis=0) for c in classes])
    centroids_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    Xt = X[test_idx]
    Xt_n = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
    sims = Xt_n @ centroids_n.T  # cosine similarity to each centroid
    pred = classes[np.argmax(sims, axis=1)]
    return float((pred == y[test_idx]).mean())


def main() -> int:
    manifest = load_manifest()
    print("=" * 70)
    print("1. norm variability")
    print("=" * 70)
    node_norms, node_cls, cell_means, cell_cls, classes = load_cell_arrays(manifest, depth=2)
    print(f"node-level norms: n={len(node_norms)} mean={node_norms.mean():.2f} "
          f"std={node_norms.std():.2f} cv={node_norms.std() / node_norms.mean():.3f}")
    for c, name in enumerate(classes):
        m = node_cls == c
        if m.sum():
            print(f"  {name:8s} n={m.sum():6d}  norm mean={node_norms[m].mean():7.2f}  "
                  f"std={node_norms[m].std():6.2f}")

    print()
    print("=" * 70)
    print("2. does norm predict class? (one-way ANOVA, node-level norm vs coarse class)")
    print("=" * 70)
    groups = [node_norms[node_cls == c] for c in range(len(classes)) if (node_cls == c).any()]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"F={f_stat:.2f}  p={p_value:.2e}  "
          f"({'norm differs significantly by class' if p_value < 0.01 else 'no strong evidence norm differs by class'})")

    print()
    print("=" * 70)
    print("3. ablation: nearest-centroid accuracy, raw vs L2-normalized cell-mean embeddings")
    print("=" * 70)
    acc_raw = nearest_centroid_accuracy(cell_means, cell_cls)
    acc_norm = nearest_centroid_accuracy(
        cell_means / (np.linalg.norm(cell_means, axis=1, keepdims=True) + 1e-8), cell_cls
    )
    print(f"raw:            {acc_raw:.3f}")
    print(f"L2-normalized:  {acc_norm:.3f}")
    drop = acc_raw - acc_norm
    print(f"drop from ablating norm: {drop:+.3f}  "
          f"({'norm appears to carry real signal -- consider L_cos + SmoothL1 later' if drop > 0.03 else 'norm ablation costs little -- cosine-only looks sufficient'})")

    print()
    print("(reminder: pretraining starts with cosine loss regardless of the above,")
    print(" per explicit project direction -- this is informative only.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
