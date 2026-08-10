"""Local-classifier-per-node (LCPN) hierarchy support, ported from the lab's
`segCLR_cell_classification` repo (`lcpn` branch, `src/data/hierarchy.py`) --
`parse_hierarchy`/`get_local_classifier_nodes` below are a direct port of that
module's algorithm (docstrings trimmed, otherwise unchanged), not a
reimplementation from scratch, so this module and the baseline's use of their
actual `HierarchyCellTypingDataset`/`LocalClassifierSNGPTrainer` agree on
exactly the same tree semantics.

`LAB_HIERARCHY_TREE` is copied verbatim from their
`configs/local_classifier_sngp.yaml` (`lcpn` branch) `data.hierarchy` block --
this is "their hierarchical clustering (the node based approach)" the user
asked to adopt, not a taxonomy we invented. See CLAUDE.md for the pivot
history (label source moved to their pre-built
`all_cells_aggregated_1718.h5`, which is where the granular labels below
actually come from).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedHierarchy:
    """Result of parsing a hierarchy config tree.

    Attributes:
        depth: Number of classification levels (= number of LCPN heads' levels).
        label_paths: Maps each granular dataset label to a list of length
            `depth` giving the class name at every level. Branches shorter
            than the maximum depth are padded by repeating the granular label.
        drop_labels: Granular labels excluded from training.
        level_classes: `level_classes[i]` is a sorted list of distinct class
            names at level `i`.
        level_maps: `level_maps[i][class_name]` is the integer index of
            `class_name` at level `i`.
    """

    depth: int
    label_paths: dict[str, list[str]]
    drop_labels: set[str]
    level_classes: list[list[str]]
    level_maps: list[dict[str, int]]


def parse_hierarchy(tree: dict) -> ParsedHierarchy:
    """Parse a nested hierarchy dict into a ParsedHierarchy.

    Tree format::

        hierarchy:
          top_group:
            mid_group:
              fine_group: [granular_label1, granular_label2]
              single_member: null          # treated as [single_member]
          drop: [label_to_exclude, ...]    # special reserved key

    Interior dict nodes are classification groups at their depth; a list
    value gives the granular labels belonging to the parent group; `null`
    means the key itself is the granular label; `drop` at the top level
    lists granular labels excluded entirely. The number of heads equals the
    maximum nesting depth across all branches; shorter branches are padded
    by repeating the granular label.
    """
    drop_labels: set[str] = set(tree.get("drop") or [])
    subtree = {k: v for k, v in tree.items() if k != "drop"}

    label_paths: dict[str, list[str]] = {}
    max_depth = [0]

    def _recurse(node, path: list[str]) -> None:
        if isinstance(node, str):
            label_paths[node] = path.copy()
            max_depth[0] = max(max_depth[0], len(path))
        elif node is None:
            granular = path[-1]
            label_paths[granular] = path.copy()
            max_depth[0] = max(max_depth[0], len(path))
        elif isinstance(node, list):
            for label in node:
                _recurse(label, path + [label])
        elif isinstance(node, dict):
            for key, value in node.items():
                _recurse(value, path + [key])

    _recurse(subtree, [])

    if not label_paths:
        raise ValueError("Hierarchy parsed to an empty label set -- check the tree structure.")

    depth = max_depth[0]
    for label in label_paths:
        path = label_paths[label]
        if len(path) < depth:
            label_paths[label] = path + [label] * (depth - len(path))

    level_class_sets: list[set[str]] = [set() for _ in range(depth)]
    for path in label_paths.values():
        for i, cls in enumerate(path):
            level_class_sets[i].add(cls)

    level_classes = [sorted(s) for s in level_class_sets]
    level_maps = [{cls: idx for idx, cls in enumerate(classes)} for classes in level_classes]

    return ParsedHierarchy(
        depth=depth,
        label_paths=label_paths,
        drop_labels=drop_labels,
        level_classes=level_classes,
        level_maps=level_maps,
    )


def get_local_classifier_nodes(
    hierarchy: ParsedHierarchy, skip_single_child: bool = True
) -> list[dict]:
    """Extract internal nodes for local-classifier-per-parent training.

    Returns a list of node dicts, root-first then level by level. Each dict:
        parent_level:          -1 for the root node (classifies into level 0)
        child_level:            0 for root, else parent_level + 1
        parent_name:            '_root_' for the root node
        parent_global_idx:      -1 for root; global index in level_maps[parent_level]
        children:                list[str] -- sorted child class names at child_level
        child_global_indices:   list[int] -- indices in level_maps[child_level]
        n_children:              len(children)

    When `skip_single_child` is True (default), nodes with exactly one child
    are omitted -- inference routes through them via a passthrough table
    instead (see gnn/lcpn.py::LCPNHead).
    """
    nodes: list[dict] = []

    level0_classes = hierarchy.level_classes[0]
    if not (skip_single_child and len(level0_classes) <= 1):
        nodes.append(
            {
                "parent_level": -1,
                "child_level": 0,
                "parent_name": "_root_",
                "parent_global_idx": -1,
                "children": list(level0_classes),
                "child_global_indices": [hierarchy.level_maps[0][c] for c in level0_classes],
                "n_children": len(level0_classes),
            }
        )

    for k in range(hierarchy.depth - 1):
        for parent_name in hierarchy.level_classes[k]:
            parent_global_idx = hierarchy.level_maps[k][parent_name]

            seen: set[str] = set()
            children: list[str] = []
            for path in hierarchy.label_paths.values():
                if path[k] == parent_name:
                    child = path[k + 1]
                    if child not in seen:
                        seen.add(child)
                        children.append(child)
            children.sort()

            if skip_single_child and len(children) <= 1:
                continue

            nodes.append(
                {
                    "parent_level": k,
                    "child_level": k + 1,
                    "parent_name": parent_name,
                    "parent_global_idx": parent_global_idx,
                    "children": children,
                    "child_global_indices": [hierarchy.level_maps[k + 1][c] for c in children],
                    "n_children": len(children),
                }
            )

    return nodes


# Verbatim from segCLR_cell_classification (lcpn branch)
# configs/local_classifier_sngp.yaml, data.hierarchy.
LAB_HIERARCHY_TREE: dict = {
    "neuron": {
        "excitatory": {
            "pyramidal": {
                "ET": ["L5ET"],
                "IT": ["L2IT", "L3IT", "L4IT", "L6IT", "L5IT"],
                "NP": ["L5NP"],
            },
            "corticothalamic": ["L6CT"],
            "thalamocortical": ["thalamocortical"],
        },
        "inhibitory": {
            "putative_somatostatin": ["MC", "NMC", "DTC"],
            "putative_parvalbumin": ["PV", "ChC"],
            "putative_cge": ["ITC", "ITCperi", "NGC", "AltBasket", "AltDTC", "L1"],
        },
    },
    "non_neuron": {
        "non_neuron": {
            "glia": ["astrocyte", "oligo", "microglia", "OPC"],
        },
    },
}
