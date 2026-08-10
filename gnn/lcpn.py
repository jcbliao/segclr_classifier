"""Local-classifier-per-node (LCPN) classification head -- replaces the flat
`nn.Linear(d, num_classes)` head this project originally called for, per
explicit user direction to adopt the lab's own hierarchical, node-based
classification approach for both the baseline and the GNN.

Mechanics (loss + top-down inference) are a direct port of
`segCLR_cell_classification`'s `LocalClassifierSNGPTrainer`
(`lcpn` branch, `src/training/trainer.py`) with the SNGP uncertainty
machinery stripped out -- this project's existing cls_head was a plain
`nn.Linear` with no SNGP anywhere else in gnn/, so LCPNHead uses plain
(optionally 1-hidden-layer) linear heads per node instead of their
spectral-norm + random-feature-GP heads. The hierarchy tree itself
(gnn/hierarchy.py::LAB_HIERARCHY_TREE) is unchanged from theirs.

Unlike the flat head, a single (B, num_classes) logits tensor doesn't exist
here (each node has a different number of children) -- callers get the
readout embedding `g` from gnn/model.py::WindowClassifier.forward() and call
`compute_loss`/`predict_top_down` on this module directly, rather than
reading a logits tensor off the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gnn.hierarchy import ParsedHierarchy, get_local_classifier_nodes


class LCPNHead(nn.Module):
    def __init__(
        self,
        hierarchy: ParsedHierarchy,
        in_dim: int,
        hidden_dim: int | None = None,
        trunk: nn.Module | None = None,
        trunk_out_dim: int | None = None,
    ):
        """
        trunk: optional module applied to the readout embedding BEFORE the
            per-node heads, shared across every node. Passing
            gnn/resnet.py::DeepResNetTrunk here reproduces the lab's own
            `local_classifier_resnet_sngp` arrangement (shared ResNet backbone,
            one head per hierarchy node) instead of the default linear probe
            straight off the readout. `trunk_out_dim` is required with it,
            since the heads are sized from the trunk's output, not `in_dim`.
        """
        super().__init__()
        self.hierarchy = hierarchy
        self.n_levels = hierarchy.depth
        self.nodes = get_local_classifier_nodes(hierarchy)

        self.trunk = trunk
        if trunk is not None and trunk_out_dim is None:
            raise ValueError("LCPNHead(trunk=...) also requires trunk_out_dim")
        head_in = in_dim if trunk is None else trunk_out_dim

        def _make_head(n_children: int) -> nn.Module:
            if hidden_dim is None:
                return nn.Linear(head_in, n_children)
            return nn.Sequential(
                nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_children)
            )

        self.heads = nn.ModuleList([_make_head(node["n_children"]) for node in self.nodes])

        # (parent_level, parent_global_idx) -> index into self.nodes/self.heads
        self.node_lookup: dict[tuple[int, int], int] = {
            (node["parent_level"], node["parent_global_idx"]): i for i, node in enumerate(self.nodes)
        }

        # Passthrough table for single-child nodes skipped by
        # get_local_classifier_nodes(skip_single_child=True): (parent_level,
        # parent_global_idx) -> child_global_idx at child_level.
        self.passthrough: dict[tuple[int, int], int] = {}
        if (-1, -1) not in self.node_lookup:
            assert len(hierarchy.level_classes[0]) == 1, (
                "root has >1 level-0 class but no root node -- get_local_classifier_nodes bug"
            )
            self.passthrough[(-1, -1)] = 0
        for k in range(self.n_levels - 1):
            for parent_name in hierarchy.level_classes[k]:
                parent_idx = hierarchy.level_maps[k][parent_name]
                key = (k, parent_idx)
                if key not in self.node_lookup:
                    child_names = {
                        path[k + 1] for path in hierarchy.label_paths.values() if path[k] == parent_name
                    }
                    child_name = next(iter(child_names))
                    self.passthrough[key] = hierarchy.level_maps[k + 1][child_name]

        # Per-node global-class-idx -> local (0..n_children-1) remap, as
        # buffers so model.to(device) moves them along with the parameters.
        for i, node in enumerate(self.nodes):
            n_at_child = len(hierarchy.level_classes[node["child_level"]])
            g2l = torch.full((n_at_child,), -1, dtype=torch.long)
            for local_idx, global_idx in enumerate(node["child_global_indices"]):
                g2l[global_idx] = local_idx
            self.register_buffer(f"g2l_{i}", g2l, persistent=False)

        # Unweighted by default (matches segCLR_cell_classification's own
        # LocalClassifierSNGPTrainer.compute_loss, which never passes
        # weight= either) -- call set_class_weights() to enable per-node
        # inverse-frequency weighting once real training showed this
        # matters: with severe class imbalance (e.g. L4IT ~2.45M windows vs.
        # singleton classes), unweighted per-node CE let val_cell_bacc sit
        # at ~chance (~1/24) while val_cell_acc sat well above chance --
        # the model learns to just predict populous classes since they
        # dominate the (sum-reduced) loss, with no gradient pressure to
        # learn rare ones. Neither of the lab's own weight_imbalanced_classes
        # settings fixes this for THIS trainer type: "loss" only affects
        # their flat/hierarchy-single-head trainers (which do read
        # dataset.class_weights), and LocalClassifierSNGPTrainer never
        # consumes class_weights at all regardless of the config value --
        # their only real lever for this trainer is "sample" (a
        # WeightedRandomSampler at the data level). This project's fix is
        # loss-level weighting instead, applied directly here.
        for i in range(len(self.nodes)):
            self.register_buffer(f"class_weight_{i}", None, persistent=False)

    def _g2l(self, i: int) -> torch.Tensor:
        return getattr(self, f"g2l_{i}")

    def _class_weight(self, i: int) -> torch.Tensor | None:
        return getattr(self, f"class_weight_{i}")

    def set_class_weights(self, node_weights: dict[int, torch.Tensor]) -> None:
        """node_weights[i]: (n_children,) inverse-frequency weight per local
        child class of node i, computed from TRAINING data only (see
        compute_node_class_weights below) -- caller's responsibility; this
        module has no way to enforce it.
        """
        device = next(self.parameters()).device
        for i, w in node_weights.items():
            self.register_buffer(f"class_weight_{i}", w.to(device), persistent=False)

    @property
    def num_classes_per_level(self) -> list[int]:
        return [len(m) for m in self.hierarchy.level_maps]

    def compute_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Sum of per-node CE losses, normalised by total active-sample count
        (not active-node count) so gradient scale doesn't vary implicitly
        with how many nodes happen to be active in a batch.

        targets: (B, n_levels) long -- global class index at every level,
        e.g. from ParsedHierarchy.level_maps applied to each sample's
        label_paths entry.
        """
        hidden = hidden if self.trunk is None else self.trunk(hidden)
        total_loss = hidden.new_zeros(())
        n_samples = 0
        for i, node in enumerate(self.nodes):
            if node["parent_level"] == -1:
                mask = torch.ones(targets.shape[0], dtype=torch.bool, device=hidden.device)
            else:
                mask = targets[:, node["parent_level"]] == node["parent_global_idx"]
            if not mask.any():
                continue

            logits = self.heads[i](hidden[mask])
            global_tgts = targets[mask, node["child_level"]]
            local_tgts = self._g2l(i)[global_tgts]
            total_loss = total_loss + F.cross_entropy(
                logits, local_tgts, weight=self._class_weight(i), reduction="sum"
            )
            n_samples += int(mask.sum().item())

        return total_loss / max(n_samples, 1)

    def predict_top_down(self, hidden: torch.Tensor) -> torch.Tensor:
        """Cascade each sample through the tree from the root, routing at
        each level by the previous level's prediction. Returns (B, n_levels)
        global class indices, one column per hierarchy level.
        """
        hidden = hidden if self.trunk is None else self.trunk(hidden)
        B = hidden.shape[0]
        preds = torch.zeros(B, self.n_levels, dtype=torch.long, device=hidden.device)
        current_preds = torch.full((B,), -1, dtype=torch.long, device=hidden.device)

        for next_level in range(self.n_levels):
            current_level = next_level - 1
            next_preds = torch.zeros(B, dtype=torch.long, device=hidden.device)

            for parent_idx_t in current_preds.unique():
                parent_idx = int(parent_idx_t.item())
                mask = current_preds == parent_idx_t
                key = (current_level, parent_idx)

                if key in self.node_lookup:
                    node_idx = self.node_lookup[key]
                    node = self.nodes[node_idx]
                    logits = self.heads[node_idx](hidden[mask])
                    local_preds = logits.argmax(dim=-1)
                    child_indices = torch.tensor(
                        node["child_global_indices"], dtype=torch.long, device=hidden.device
                    )
                    next_preds[mask] = child_indices[local_preds]
                else:
                    next_preds[mask] = self.passthrough[key]

            preds[:, next_level] = next_preds
            current_preds = next_preds

        return preds

    def predict_finest(self, hidden: torch.Tensor) -> torch.Tensor:
        """Convenience: just the finest (last) level's predictions, (B,)."""
        return self.predict_top_down(hidden)[:, -1]


def compute_node_class_weights(
    hierarchy: ParsedHierarchy, window_counts_by_label: dict[str, float], eps: float = 1.0,
) -> dict[int, torch.Tensor]:
    """Per-LCPN-node inverse-frequency class weights, for LCPNHead.set_class_weights.

    window_counts_by_label: {granular_label: total WINDOW count in train split}
    -- window count, not cell count. A window inherits its cell's label (every
    window from one cell shares that cell's granular label), so a 20,000-node
    cell contributes 20,000x more windows of its class than a 500-node cell of
    the same class -- weighting from cell counts alone would systematically
    under-correct for exactly the imbalance that matters at the granularity
    training actually happens at. Compute window_counts_by_label from
    data/manifest.json's train-split n_nodes_covered per cell, not by
    materializing every window.

    Weight_c = N_node / (n_children * count_c) -- ordinary inverse-frequency
    weighting, applied separately per LCPN node since each node's children
    are a different, smaller label set than the full 24-class vocabulary.
    """
    nodes = get_local_classifier_nodes(hierarchy)
    weights_by_node: dict[int, torch.Tensor] = {}
    for i, node in enumerate(nodes):
        child_counts = torch.zeros(node["n_children"])
        for label, count in window_counts_by_label.items():
            path = hierarchy.label_paths[label]
            if node["parent_level"] == -1 or path[node["parent_level"]] == node["parent_name"]:
                child_name = path[node["child_level"]]
                local_idx = node["children"].index(child_name)
                child_counts[local_idx] += count
        child_counts = child_counts.clamp(min=eps)
        weights_by_node[i] = child_counts.sum() / (node["n_children"] * child_counts)
    return weights_by_node
