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


#: Reserved key naming a bucket of granular labels that terminate at their
#: parent group instead of each earning a level of its own. Matches
#: segclr_db's `LabelHierarchy.from_tree` leaf_key, so a tree registered in
#: the store parses to the same levels here as it does there.
LEAF_BUCKET_KEY = "_labels_"


def parse_hierarchy(tree: dict) -> ParsedHierarchy:
    """Parse a nested hierarchy dict into a ParsedHierarchy.

    Tree format::

        hierarchy:
          top_group:
            mid_group:
              fine_group: [granular_label1, granular_label2]
              single_member: null          # treated as [single_member]
              coarse_group:
                _labels_: [lumped1, lumped2]   # stop at coarse_group
          drop: [label_to_exclude, ...]    # special reserved key

    Interior dict nodes are classification groups at their depth; a list
    value gives the granular labels belonging to the parent group; `null`
    means the key itself is the granular label; `_labels_` collects granular
    labels whose path ENDS at the parent group, so they stay addressable by
    the label the dataset carries without becoming classes of their own;
    `drop` at the top level lists granular labels excluded entirely. The
    number of heads equals the maximum nesting depth across all branches;
    shorter branches are padded by repeating their last path element (for
    everything but a `_labels_` bucket, that element is the granular label
    itself).
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
                if key == LEAF_BUCKET_KEY:
                    for label in value or []:
                        label_paths[str(label)] = path.copy()
                        max_depth[0] = max(max_depth[0], len(path))
                else:
                    _recurse(value, path + [key])

    _recurse(subtree, [])

    if not label_paths:
        raise ValueError("Hierarchy parsed to an empty label set -- check the tree structure.")

    depth = max_depth[0]
    for label in label_paths:
        path = label_paths[label]
        if len(path) < depth:
            # Repeat the last element, which is the granular label on every
            # branch except a `_labels_` bucket, where it is the parent group
            # the bucket's labels stop at.
            label_paths[label] = path + [path[-1]] * (depth - len(path))

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


def with_dropped_labels(hierarchy: ParsedHierarchy, labels) -> ParsedHierarchy:
    """Add `labels` to the hierarchy's drop set, leaving everything else alone.

    The `drop` key parse_hierarchy understands is the lab's own way of saying
    this, but it lives inside the tree -- and the trees here are pinned
    verbatim to the rows registered in the store, so a `drop` key added to one
    would make it differ from what the store holds. This is the same
    instruction applied from outside the tree instead, for callers that pin
    the tree (see data/dataset_lcpn.py::DROP_LABELS).

    A dropped label is removed from `label_paths` outright, and every level
    class that no surviving label reaches is pruned with it -- so dropping OPC
    at a level where OPC is its own class yields an 8-class problem, not a
    9-class one with an empty slot.

    This deviates from segCLR_cell_classification, whose
    HierarchyCellTypingDataset filters samples by drop_labels but still builds
    level_classes from every label in the tree. Their way leaves a class with
    no examples, which here would mean an LCPN output unit that never receives
    a positive target, a per-class recall column that is undefined rather than
    zero, and a macro average whose denominator disagrees with the number of
    classes actually being learned. Class indices shift when a class is
    pruned, which is safe only because every consumer -- LCPN nodes, class
    weights, per-window targets, reported class names -- derives from this one
    object; a checkpoint from before a drop cannot be resumed across it.
    """
    drop = set(hierarchy.drop_labels) | set(labels)
    surviving = {
        label: list(path) for label, path in hierarchy.label_paths.items() if label not in drop
    }
    if not surviving:
        raise ValueError(f"dropping {sorted(drop)} leaves no labels at all")

    level_class_sets: list[set[str]] = [set() for _ in range(hierarchy.depth)]
    for path in surviving.values():
        for i, cls in enumerate(path):
            level_class_sets[i].add(cls)
    level_classes = [sorted(s) for s in level_class_sets]

    return ParsedHierarchy(
        depth=hierarchy.depth,
        label_paths=surviving,
        drop_labels=drop,
        level_classes=level_classes,
        level_maps=[{cls: i for i, cls in enumerate(cs)} for cs in level_classes],
    )


def truncate_hierarchy(hierarchy: ParsedHierarchy, levels: int = 1) -> ParsedHierarchy:
    """Drop the `levels` finest classification levels.

    Granular labels remain the keys of `label_paths`, so callers still look a
    cell up by the label the dataset actually carries; each path now ends at
    the coarser level, and several granular labels can share one path. Every
    consumer of a ParsedHierarchy -- the LCPN nodes, their class weights, the
    per-window targets, and the class list metrics are reported over -- is
    derived from these fields, so they all move together.
    """
    depth = hierarchy.depth - levels
    if depth < 1:
        raise ValueError(
            f"cannot drop {levels} level(s) from a depth-{hierarchy.depth} hierarchy"
        )
    return ParsedHierarchy(
        depth=depth,
        label_paths={label: path[:depth] for label, path in hierarchy.label_paths.items()},
        drop_labels=set(hierarchy.drop_labels),
        level_classes=[list(c) for c in hierarchy.level_classes[:depth]],
        level_maps=[dict(m) for m in hierarchy.level_maps[:depth]],
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


# Verbatim from the `hierarchy_v2` row of the shared v3 store's
# label_hierarchies table (db.hierarchy("hierarchy_v2").tree, content_hash
# 50e5c4d2dc6f7ac788dd1c74912d2d276a1bc40852569c80d6e5c87c1f95f303).
# scripts/check_registered_hierarchies.py prints it; the store is the
# authority and scripts/check_hierarchy_v2_parse.py asserts that parsing this
# copy reproduces the level_classes the store reports, so a drifted
# transcription fails loudly rather than training on a different taxonomy.
#
# Differs from LAB_HIERARCHY_TREE in three ways, all of which change what the
# levels mean: the interneuron families are `_labels_` buckets, so their
# subtypes stop at putative_{cge,parvalbumin,somatostatin} rather than each
# becoming a class; the four glia are separate classes from level 2 down
# instead of one `glia` group; and corticothalamic is gone from level 2, L6CT
# reaching pyramidal -> CT instead. Level sizes: [2, 3, 9, 12, 16].
HIERARCHY_V2_TREE: dict = {
    "neuron": {
        "excitatory": {
            "pyramidal": {
                "CT": ["L6CT"],
                "ET": ["L5ET"],
                "IT": ["L2IT", "L3IT", "L4IT", "L6IT", "L5IT"],
                "NP": ["L5NP"],
            },
            "thalamocortical": ["thalamocortical"],
        },
        "inhibitory": {
            "putative_cge": {
                LEAF_BUCKET_KEY: ["ITC", "ITCperi", "NGC", "AltBasket", "AltDTC", "L1"],
            },
            "putative_parvalbumin": {LEAF_BUCKET_KEY: ["PV", "ChC"]},
            "putative_somatostatin": {LEAF_BUCKET_KEY: ["MC", "NMC", "DTC"]},
        },
    },
    "non_neuron": {
        "non_neuron": ["astrocyte", "oligo", "microglia", "OPC"],
    },
}
