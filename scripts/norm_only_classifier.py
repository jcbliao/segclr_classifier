"""Deeper follow-up to scripts/norm_diagnostic.py, on the real 365-cell
dataset: (1) how much do embedding norms actually vary, broken down by the
FULL fine-grained cell_type (not just the coarse neuron/glia split the first
diagnostic used), with a proper variance-decomposition (eta^2) instead of
just an ANOVA F-stat, which is not very interpretable at n~1.8M (everything
looks "significant"); (2) can norms ALONE (no direction information at all)
predict cell type -- trains an actual classifier (sklearn LogisticRegression)
on per-cell norm-summary features, using the same train/val/test split as
scripts/train_baseline.py, at three label depths.

Run via sbatch. CPU only -- sklearn's LogisticRegression has no CUDA path, so
there is no GPU to move this to; norms are also scalar-derived per-cell
features (5-8 numbers per cell), nowhere near large enough to need one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from data.dataset import label_vocab, load_manifest  # noqa: E402
from gnn.metrics import summarize  # noqa: E402

GRAPH_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "graph_cache"


def eta_squared(values: np.ndarray, group_ids: np.ndarray) -> float:
    """Fraction of total variance explained by group membership -- SS_between
    / SS_total. 0 = groups explain nothing, 1 = groups explain everything.
    Interpretable regardless of n, unlike an ANOVA F-stat/p-value."""
    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    ss_between = 0.0
    for g in np.unique(group_ids):
        vals = values[group_ids == g]
        ss_between += len(vals) * (vals.mean() - grand_mean) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def norm_feature_vector(norms: np.ndarray) -> np.ndarray:
    """Per-cell summary of its node-level embedding norms -- deliberately
    ONLY magnitude information, no direction, so any predictive power here
    is specifically about norm, not a proxy for the full embedding."""
    return np.array(
        [
            norms.mean(),
            norms.std(),
            np.median(norms),
            np.percentile(norms, 10),
            np.percentile(norms, 90),
            norms.min(),
            norms.max(),
        ]
    )


def load_data(manifest: dict):
    """Returns per-cell: node-level norms, fine cell_type, split."""
    cells = {}
    for root_id, info in manifest["cells"].items():
        path = GRAPH_CACHE_DIR / f"{root_id}.pt"
        if not path.exists():
            continue
        data = torch.load(path, weights_only=False)
        norms = np.linalg.norm(data.x.numpy(), axis=1)
        cells[int(root_id)] = {
            "norms": norms,
            "cell_type": info["cell_type"],
            "split": info["split"],
        }
    return cells


def main() -> int:
    manifest = load_manifest()
    cells = load_data(manifest)
    print(f"{len(cells)} cells loaded")

    all_norms = np.concatenate([c["norms"] for c in cells.values()])
    node_cell_type = np.concatenate(
        [np.full(len(c["norms"]), c["cell_type"]) for c in cells.values()]
    )

    print("=" * 70)
    print("1. how much do norms vary?")
    print("=" * 70)
    print(
        f"overall node-level norm: n={len(all_norms)} mean={all_norms.mean():.2f} "
        f"std={all_norms.std():.2f} cv={all_norms.std() / all_norms.mean():.3f}"
    )
    pct = np.percentile(all_norms, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f"percentiles [1,5,10,25,50,75,90,95,99]: {np.round(pct, 1).tolist()}")

    print("\nper fine-grained cell_type (sorted by mean norm):")
    rows = []
    for ct in np.unique(node_cell_type):
        vals = all_norms[node_cell_type == ct]
        rows.append((ct, len(vals), vals.mean(), vals.std(), vals.std() / vals.mean()))
    rows.sort(key=lambda r: -r[2])
    print(f"{'cell_type':16s} {'n':>9s} {'mean':>8s} {'std':>8s} {'cv':>6s}")
    for ct, n, mean, std, cv in rows:
        print(f"{ct:16s} {n:9d} {mean:8.2f} {std:8.2f} {cv:6.3f}")

    print("\nvariance explained by cell type (eta^2 = SS_between / SS_total):")
    for depth, label in [(2, "coarse (neuron/glia)"), (3, "mid (+E/I/glia-subtype)"), (None, "fine (full cell_type)")]:
        group = (
            node_cell_type
            if depth is None
            else np.array(["-".join(ct.split("-")[:depth]) for ct in node_cell_type])
        )
        eta2 = eta_squared(all_norms, group)
        print(f"  depth={depth!s:>5s} ({label:24s}): eta^2 = {eta2:.4f}  "
              f"({eta2:.1%} of norm variance is between-class; rest is within-class/noise)")

    print()
    print("=" * 70)
    print("2. can norms ALONE predict cell type? (LogisticRegression on per-cell norm summary)")
    print("=" * 70)
    for depth in (2, 3, None):
        classes, class_to_idx = label_vocab(manifest, depth)

        def label_id(ct, depth=depth):
            key = ct if depth is None else "-".join(ct.split("-")[:depth])
            return class_to_idx[key]

        Xtr, ytr, Xval, yval, Xtest, ytest = [], [], [], [], [], []
        for c in cells.values():
            feat = norm_feature_vector(c["norms"])
            label = label_id(c["cell_type"])
            if c["split"] == "train":
                Xtr.append(feat)
                ytr.append(label)
            elif c["split"] == "val":
                Xval.append(feat)
                yval.append(label)
            else:
                Xtest.append(feat)
                ytest.append(label)
        Xtr, ytr = np.stack(Xtr), np.array(ytr)
        Xtest, ytest = np.stack(Xtest), np.array(ytest)

        mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_n, Xtest_n = (Xtr - mu) / sigma, (Xtest - mu) / sigma

        # class_weight="balanced" since several fine classes have <10 examples
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_n, ytr)
        pred = clf.predict(Xtest_n)
        metrics = summarize(ytest, pred, len(classes), classes)

        print(f"\n--- depth={depth} ({len(classes)} classes) ---")
        print(f"train={len(ytr)} test={len(ytest)}")
        print(f"accuracy={metrics['accuracy']:.3f}  balanced_accuracy={metrics['balanced_accuracy']:.3f}  "
              f"macro_f1={metrics['macro_f1']:.3f}")
        print(f"per-class recall: {metrics['per_class_recall']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
