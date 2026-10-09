"""Flat finest-level classification head for controlled LCPN comparisons."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gnn.hierarchy import ParsedHierarchy


def compute_flat_class_weights(
    hierarchy: ParsedHierarchy, window_counts_by_label: dict[str, float]
) -> torch.Tensor:
    """Inverse-frequency weights over finest classes, from training windows."""
    counts = torch.zeros(len(hierarchy.level_classes[-1]), dtype=torch.float32)
    for label, count in window_counts_by_label.items():
        if label not in hierarchy.label_paths:
            continue
        leaf_name = hierarchy.label_paths[label][-1]
        counts[hierarchy.level_maps[-1][leaf_name]] += float(count)
    if (counts <= 0).any():
        raise ValueError("flat loss weighting found a finest class with no training windows")
    weights = counts.sum() / (len(counts) * counts)
    return weights / weights.mean()


class FlatHead(nn.Module):
    """One softmax over the active finest-level classes.

    ``predict_top_down`` reconstructs the unique valid hierarchy path for the
    predicted leaf solely to preserve the existing evaluation/export API. No
    coarse prediction is used to select or constrain the leaf prediction.
    """

    def __init__(self, hierarchy: ParsedHierarchy, in_dim: int,
                 trunk: nn.Module | None = None,
                 trunk_out_dim: int | None = None):
        super().__init__()
        self.hierarchy = hierarchy
        self.n_levels = hierarchy.depth
        self.trunk = trunk
        if trunk is not None and trunk_out_dim is None:
            raise ValueError("FlatHead(trunk=...) also requires trunk_out_dim")
        head_in = in_dim if trunk is None else trunk_out_dim
        self.head = nn.Linear(head_in, len(hierarchy.level_classes[-1]))
        self.register_buffer("class_weight", None, persistent=False)

        # Each active finest class must correspond to one hierarchy path. It
        # may represent several original granular labels, but their retained
        # path is identical after hierarchy pruning/truncation.
        paths: dict[int, tuple[int, ...]] = {}
        for path_names in hierarchy.label_paths.values():
            indices = tuple(
                hierarchy.level_maps[level][name]
                for level, name in enumerate(path_names)
            )
            leaf = indices[-1]
            if leaf in paths and paths[leaf] != indices:
                raise ValueError(f"finest class {leaf} has non-unique hierarchy paths")
            paths[leaf] = indices
        expected = set(range(len(hierarchy.level_classes[-1])))
        if set(paths) != expected:
            raise ValueError("not every finest class has a hierarchy path")
        self.register_buffer(
            "leaf_paths",
            torch.tensor([paths[i] for i in range(len(paths))], dtype=torch.long),
            persistent=False,
        )

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self.register_buffer(
            "class_weight", weights.to(next(self.parameters()).device), persistent=False
        )

    def _features(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden if self.trunk is None else self.trunk(hidden)

    def compute_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.head(self._features(hidden))
        return F.cross_entropy(logits, targets[:, -1], weight=self.class_weight)

    def predict_top_down(self, hidden: torch.Tensor) -> torch.Tensor:
        leaves = self.head(self._features(hidden)).argmax(dim=-1)
        return self.leaf_paths[leaves]

    def predict_finest(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(hidden)).argmax(dim=-1)
