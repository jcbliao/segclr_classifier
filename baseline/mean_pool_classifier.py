"""Baseline to beat: geodesic-window mean pooling of per-node embeddings, then
a classifier head (Elabbady et al. 2023's aggregation method). Computed via
segclr_db.aggregate.geodesic_mean/build_csr, imported as pure functions --
not through segclr_db's Store/Writer/Database (see data/cave_skeletons.py for
why). window_nm=25000 (25um) matches the paper and the public release's own
"_agg25um" bucket naming.

Classification happens **per aggregated point**, not on a single whole-cell-
averaged vector -- confirmed against two independent sources: the original
classifier gist (train_embeddings.extend(e), flattening every windowed node
embedding in a cell into its own training example) and this lab's own
replication (github.mit.edu/collina/segCLR_cell_classification's
aggregation_study/03_train_evaluate.py, which reports "per-point accuracy"
and "cell-level majority-vote accuracy" via cell_majority_vote_accuracy(),
never embedding averaging). An earlier version of this module averaged a
cell's windowed embeddings into one vector before classifying once --
deprecated, see results/deprecated_wholecell_baseline/README.md for why that
measures something different from "aggregation radius" as the paper means it.

Consumes the SAME cells, splits, and label vocabulary as the GNN (built by
data/build_dataset_from_store.py) -- only the aggregation/readout differs,
which is the whole point of the comparison (CLAUDE.md "Project goal").
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from segclr_db.aggregate import geodesic_mean  # noqa: E402

from data import cave_skeletons as cs  # noqa: E402
from data.dataset import label_vocab, load_manifest  # noqa: E402

WINDOW_NM = 25_000
GRAPH_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "graph_cache"


def node_level_features(root_id: int, window_nm: float = WINDOW_NM) -> np.ndarray:
    """geodesic_mean at every covered node, kept per-node -- NOT collapsed to
    a single whole-cell vector. Returns (n_covered_nodes, D)."""
    data = torch.load(GRAPH_CACHE_DIR / f"{root_id}.pt", weights_only=False)
    skeleton = cs.load_cached(root_id)
    if skeleton is None:
        raise FileNotFoundError(f"no cached skeleton for {root_id} -- run the dataset build first")
    result = geodesic_mean(skeleton, data.orig_node_ids.numpy(), data.x.numpy(), window_nm)
    return result.embeddings


def build_node_feature_matrix(manifest: dict, split: str, depth: int, window_nm: float = WINDOW_NM):
    """Stacks EVERY covered node's windowed embedding across every cell in the
    split into one big (N_nodes_total, D) matrix, with a parallel per-node
    label (the parent cell's label, repeated) and root_id (for majority-vote
    aggregation back to cell level at eval time)."""
    classes, class_to_idx = label_vocab(manifest, depth)
    X, y, root_ids = [], [], []
    for root_id_str, info in manifest["cells"].items():
        if info["split"] != split:
            continue
        root_id = int(root_id_str)
        feats = node_level_features(root_id, window_nm)
        ct = info["cell_type"] if depth is None else "-".join(info["cell_type"].split("-")[:depth])
        label = class_to_idx[ct]
        X.append(feats)
        y.append(np.full(len(feats), label, dtype=np.int64))
        root_ids.append(np.full(len(feats), root_id, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y), np.concatenate(root_ids), classes


class MeanPoolClassifier(nn.Module):
    """Same-capacity classifier head as gnn.model.GraphAutoEncoderClassifier's
    cls_head -- an MLP option is offered too since a single linear layer may
    be an unfairly weak comparison. Try both; report whichever the baseline
    does better with, since the point is the best baseline this aggregation
    choice can produce, not the weakest one. Operates per-node here (see
    module docstring) -- the "many nodes -> one cell decision" collapse
    happens via majority vote on predictions, not by averaging features.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
