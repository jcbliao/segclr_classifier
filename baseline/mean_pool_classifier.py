"""Baseline to beat: geodesic-window mean pooling of per-node embeddings, then
a classifier head (Elabbady et al. 2023's aggregation method). Computed via
segclr_db.aggregate.geodesic_mean/build_csr, imported as pure functions --
not through segclr_db's Store/Writer/Database (see data/cave_skeletons.py for
why). window_nm=25000 (25um) matches the paper and the public release's own
"_agg25um" bucket naming.

Consumes the SAME cells, splits, and label vocabulary as the GNN (built once
by data/build_dataset.py) -- only the aggregation/readout differs, which is
the whole point of the comparison (CLAUDE.md "Project goal"). Per-cell feature
vectors come from geodesic_mean over data.orig_node_ids/data.x (the identical
covered-node embeddings the GNN trains on), then a mean over nodes for one
vector per cell -- the same "many node vectors -> one graph-level vector"
collapse the GNN's CLS-attention readout does, just with a mean instead of
learned attention. That is deliberately the only degree of freedom this
baseline gets: everything downstream (the classifier head) matches the GNN's
head as closely as possible so an accuracy gap reflects the aggregation
choice, not incidental architecture differences.
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


def pooled_feature(root_id: int, window_nm: float = WINDOW_NM) -> np.ndarray:
    """One (D,) feature vector per cell: geodesic_mean over window_nm, then
    mean over the resulting per-node vectors."""
    data = torch.load(GRAPH_CACHE_DIR / f"{root_id}.pt", weights_only=False)
    skeleton = cs.load_cached(root_id)
    if skeleton is None:
        raise FileNotFoundError(f"no cached skeleton for {root_id} -- run data/build_dataset.py first")
    result = geodesic_mean(skeleton, data.orig_node_ids.numpy(), data.x.numpy(), window_nm)
    return result.embeddings.mean(axis=0)


def build_feature_matrix(manifest: dict, split: str, depth: int, window_nm: float = WINDOW_NM):
    classes, class_to_idx = label_vocab(manifest, depth)
    X, y, root_ids = [], [], []
    for root_id_str, info in manifest["cells"].items():
        if info["split"] != split:
            continue
        root_id = int(root_id_str)
        X.append(pooled_feature(root_id, window_nm))
        ct = "-".join(info["cell_type"].split("-")[:depth])
        y.append(class_to_idx[ct])
        root_ids.append(root_id)
    return np.stack(X), np.array(y, dtype=np.int64), root_ids, classes


class MeanPoolClassifier(nn.Module):
    """Same-capacity classifier head as gnn.model.GraphAutoEncoderClassifier's
    cls_head (a linear layer on a d-dim graph vector) -- an MLP is offered too
    since the baseline's "graph vector" here is a mean, not something an
    encoder already nonlinearly transformed, so a single linear layer may be
    an unfairly weak comparison. Try both; report whichever the baseline does
    better with, since the point is the best baseline this aggregation choice
    can produce, not the weakest one.
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
